# app/graph.py
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun

from agents.hypothesis_generator import HypothesisGenerator
from agents.evidence_evaluator import EvidenceEvaluatorAgent
from agents.risk_assessor import RiskAssessorAgent
from agents.critic import CriticAgent
from agents.synthesizer import SynthesizerAgent
from tools.medical_retriever import StaticRetriever

class CmarState(TypedDict):
    patient_scenario: Dict
    specialty_groups: Dict[str, List[Dict]]
    critic_feedback: Dict
    critic_history: List[Dict]
    refinement_loop_count: int
    final_report: Dict
    ground_truth: str 

def build_graph(llm_client: ChatGoogleGenerativeAI, embeddings_client: HuggingFaceEmbeddings, optimization_config: dict = None, benchmark_mode: bool = False):
    if optimization_config is None: optimization_config = {}
    
    enable_critic = optimization_config.get('enable_critic_loop', True)
    max_loops = optimization_config.get('max_refinement_loops', 3)
    batch_evidence = optimization_config.get('batch_evidence_retrieval', True)

    # --- BENCHMARK CONFIGURATION ---
    hypo_prompt_override = None
    if benchmark_mode:
        print("-> ⚠️ BENCHMARK MODE ACTIVE: Using Option Selection Prompt")
        # Strong Prompt for Option Selection
        hypo_prompt_override = """You are taking the USMLE medical board exam.
        
        **INPUT DATA:**
        You will be provided with a 'Patient Scenario' and a list of 'Candidate Diagnoses (Options)'.

        **TASK:**
        1.  Analyze the clinical vignette carefully.
        2.  Evaluate EACH of the provided Options against the patient's symptoms.
        3.  **SELECT THE CORRECT OPTION.**
        4.  Output the top 3 most likely options from the provided list, ranked by probability.
        5.  **DO NOT** invent new diagnoses. You MUST stick to the provided Options.
        """

    # --- AGENT INITIALIZATION ---
    retriever_tool = StaticRetriever(embeddings=embeddings_client).get_retriever()
    web_search_tool = DuckDuckGoSearchRun()
    
    # Initialize Agents
    hypothesis_agent = HypothesisGenerator(llm_client, prompt_override=hypo_prompt_override)
    
    evidence_evaluator_agent = EvidenceEvaluatorAgent(
        llm=llm_client,
        retriever_tool=retriever_tool, 
        web_search_tool=web_search_tool, 
        enable_batching=batch_evidence
    )
    
    risk_assessor_agent = RiskAssessorAgent(llm_client, benchmark_mode=benchmark_mode)
    critic_agent = CriticAgent(llm_client, benchmark_mode=benchmark_mode)
    synthesizer_agent = SynthesizerAgent(llm_client, embeddings_model=embeddings_client) 

    # --- NODE FUNCTIONS ---
    def run_hypothesis_generator(state: CmarState):
        print("\n--- Node: Generate Hypotheses ---")
        patient_scenario = state['patient_scenario']
        specialty_groups = hypothesis_agent.run(patient_scenario['summary'])
        return {"specialty_groups": specialty_groups, "refinement_loop_count": 0, "critic_history": state.get('critic_history', [])}

    def run_evidence_evaluation(state: CmarState):
        """Parallelized Evidence Evaluation"""
        print("\n--- Node: Evidence Evaluation (Parallel) ---")
        patient_summary = state['patient_scenario']['summary']
        specialty_groups = state['specialty_groups']
        critic_feedback = state.get('critic_feedback', {})
        target_specialty = critic_feedback.get('target_specialty')
        
        updated_groups = {}
        tasks = []
        
        # Identify tasks
        for specialty, hypotheses in specialty_groups.items():
            is_target = (specialty == target_specialty)
            # Evaluate if new (no evidence yet) OR if it's the critic's specific target
            to_evaluate = [h for h in hypotheses if 'evidence' not in h]
            
            if to_evaluate or (is_target and hypotheses):
                # If targeted, re-evaluate all to be safe, or just the new ones
                eval_list = hypotheses if is_target else to_evaluate
                if eval_list:
                    tasks.append((specialty, hypotheses, eval_list))
            else:
                updated_groups[specialty] = hypotheses

        # Execute in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_spec = {
                executor.submit(
                    evidence_evaluator_agent.evaluate_hypotheses_for_specialty,
                    patient_summary, 
                    task[2] # list to evaluate
                ): (task[0], task[1]) # (specialty, original_list)
                for task in tasks
            }
            
            for future in as_completed(future_to_spec):
                specialty, original_list = future_to_spec[future]
                try:
                    updated_list = future.result()
                    # Merge logic: update original list with new results
                    updated_map = {h['hypothesis']: h for h in updated_list}
                    merged = [updated_map.get(h['hypothesis'], h) for h in original_list]
                    updated_groups[specialty] = merged
                    print(f"  -> Completed evaluation for {specialty}")
                except Exception as e:
                    print(f"  -> Error in {specialty}: {e}")
                    
                    # --- FIX START: Mark failed items with error message instead of leaving empty ---
                    failed_list = []
                    for h in original_list:
                        if 'evidence' not in h:
                            h_copy = h.copy()
                            h_copy['evidence'] = f"⚠️ Evaluation Failed: {str(e)}"
                            failed_list.append(h_copy)
                        else:
                            failed_list.append(h)
                    updated_groups[specialty] = failed_list
                    # --- FIX END ---

        return {"specialty_groups": updated_groups}
    
    def run_risk_assessment(state: CmarState):
        print("\n--- Node: Risk Assessment ---")
        patient_summary = state['patient_scenario']['summary']
        specialty_groups = state['specialty_groups']
        feedback = state.get('critic_feedback', {})
        target_specialty = feedback.get('target_specialty')
        updated_groups = {}
        for specialty, hypotheses in specialty_groups.items():
            needs_assessment = any('severity' not in h for h in hypotheses)
            is_target = (specialty == target_specialty)
            if needs_assessment or is_target:
                challenge = feedback.get('feedback', 'Initial') if is_target else 'Initial'
                updated_groups[specialty] = risk_assessor_agent.assess_risk_for_specialty(patient_summary, hypotheses, challenge)
            else: updated_groups[specialty] = hypotheses
        return {"specialty_groups": updated_groups}

    def run_critic(state: CmarState):
        print("\n--- Node: Critic Review ---")
        decision = critic_agent.run(state['patient_scenario']['summary'], state['specialty_groups'], state.get('critic_history', []))
        loop_count = state.get('refinement_loop_count', 0) + 1
        return {"critic_feedback": decision, "critic_history": state.get('critic_history', []) + [{**decision, 'iteration': loop_count}], "refinement_loop_count": loop_count}

    def update_hypotheses(state: CmarState):
        print("\n--- Node: Update Hypotheses ---")
        feedback = state.get('critic_feedback', {})
        decision = feedback.get('decision')
        specialty_groups = state['specialty_groups']
        target = feedback.get('target_specialty')
        if decision == "ADD_HYPOTHESIS" and target:
            if target not in specialty_groups: specialty_groups[target] = []
            specialty_groups[target].append({"hypothesis": feedback.get('new_hypothesis_name'), "specialty": target})
        elif decision == "DISCARD_HYPOTHESIS" and target and target in specialty_groups:
            specialty_groups[target] = [h for h in specialty_groups[target] if h['hypothesis'] != feedback.get('hypothesis_to_discard')]
        return {"specialty_groups": specialty_groups}

    def run_synthesizer(state: CmarState):
        print("\n--- Node: Synthesizer ---")
        return {"final_report": synthesizer_agent.run(state['patient_scenario']['summary'], state['specialty_groups'], state.get('ground_truth'))}

    def decide_next_step(state: CmarState):
        decision = state['critic_feedback']['decision']
        if decision == "ASK_HUMAN": return END
        if not enable_critic or state.get('refinement_loop_count', 0) >= max_loops or decision == 'APPROVE': return "synthesize"
        return "refine"

    # --- WIRING ---
    workflow = StateGraph(CmarState)
    workflow.add_node("hypothesis_generator", run_hypothesis_generator)
    workflow.add_node("evidence_evaluator", run_evidence_evaluation)
    workflow.add_node("risk_assessor", run_risk_assessment)
    workflow.add_node("synthesizer", run_synthesizer)
    
    if enable_critic:
        workflow.add_node("critic", run_critic)
        workflow.add_node("update_hypotheses", update_hypotheses)
        workflow.set_entry_point("hypothesis_generator")
        workflow.add_edge("hypothesis_generator", "evidence_evaluator")
        workflow.add_edge("evidence_evaluator", "risk_assessor")
        workflow.add_edge("risk_assessor", "critic")
        workflow.add_conditional_edges("critic", decide_next_step, {"refine": "update_hypotheses", "synthesize": "synthesizer", END: END})
        workflow.add_edge("update_hypotheses", "evidence_evaluator")
    else:
        workflow.set_entry_point("hypothesis_generator")
        workflow.add_edge("hypothesis_generator", "evidence_evaluator")
        workflow.add_edge("evidence_evaluator", "risk_assessor")
        workflow.add_edge("risk_assessor", "synthesizer")
    
    workflow.add_edge("synthesizer", END)
    return workflow.compile()