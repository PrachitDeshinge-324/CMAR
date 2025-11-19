# app/graph.py
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from concurrent.futures import ThreadPoolExecutor, as_completed

# LLM and Tool Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun

# Agent Imports
from agents.hypothesis_generator import HypothesisGenerator
from agents.evidence_evaluator import EvidenceEvaluatorAgent
from agents.risk_assessor import RiskAssessorAgent
from agents.critic import CriticAgent
from agents.synthesizer import SynthesizerAgent

# Custom Tool Imports
from tools.medical_retriever import StaticRetriever

# 1. DEFINE THE GRAPH'S STATE
class CmarState(TypedDict):
    patient_scenario: Dict
    specialty_groups: Dict[str, List[Dict]]
    critic_feedback: Dict
    critic_history: List[Dict]
    refinement_loop_count: int
    final_report: Dict
    ground_truth: str  # Optional ground truth for evaluation

# 2. DEFINE THE GRAPH BUILDER
def build_graph(llm_client: ChatGoogleGenerativeAI, embeddings_client: HuggingFaceEmbeddings, optimization_config: dict = None):
    if optimization_config is None: optimization_config = {}
    
    enable_critic = optimization_config.get('enable_critic_loop', True)
    max_loops = optimization_config.get('max_refinement_loops', 3)
    batch_evidence = optimization_config.get('batch_evidence_retrieval', True)

    # --- AGENT INITIALIZATION ---
    retriever_tool = StaticRetriever(embeddings=embeddings_client).get_retriever()
    web_search_tool = DuckDuckGoSearchRun()
    
    hypothesis_agent = HypothesisGenerator(llm_client)
    
    # --- FIX IS HERE: Pass llm_client ---
    evidence_evaluator_agent = EvidenceEvaluatorAgent(
        llm=llm_client,           # <--- REQUIRED ARGUMENT ADDED
        retriever_tool=retriever_tool, 
        web_search_tool=web_search_tool,
        enable_batching=batch_evidence
    )
    
    risk_assessor_agent = RiskAssessorAgent(llm_client)
    critic_agent = CriticAgent(llm_client)
    synthesizer_agent = SynthesizerAgent(llm_client, embeddings_model=embeddings_client) 

    # --- NODE FUNCTION DEFINITIONS ---

    def run_hypothesis_generator(state: CmarState):
        print("\n--- Node: Generate Hypotheses ---")
        patient_scenario = state['patient_scenario']
        specialty_groups = hypothesis_agent.run(patient_scenario['summary'])
        return {
            "specialty_groups": specialty_groups, 
            "refinement_loop_count": 0,
            "critic_history": []
        }

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
                    updated_groups[specialty] = original_list

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
                challenge = feedback.get('feedback', 'Initial assessment') if is_target else 'Initial assessment'
                updated_hypotheses = risk_assessor_agent.assess_risk_for_specialty(
                    patient_summary, hypotheses, challenge
                )
                updated_groups[specialty] = updated_hypotheses
            else:
                updated_groups[specialty] = hypotheses
                
        return {"specialty_groups": updated_groups}

    def run_critic(state: CmarState):
        print("\n--- Node: Critic Review ---")
        patient_summary = state['patient_scenario']['summary']
        specialty_groups = state['specialty_groups']
        critic_history = state.get('critic_history', [])
        
        decision = critic_agent.run(patient_summary, specialty_groups, critic_history)
        
        # Update history
        loop_count = state.get('refinement_loop_count', 0) + 1
        new_history = critic_history + [{**decision, 'iteration': loop_count}]
        
        return {
            "critic_feedback": decision, 
            "critic_history": new_history,
            "refinement_loop_count": loop_count
        }
        
    def update_hypotheses(state: CmarState):
        """Updates hypothesis list based on Critic's decision."""
        print("\n--- Node: Update Hypotheses ---")
        feedback = state.get('critic_feedback', {})
        decision = feedback.get('decision')
        specialty_groups = state['specialty_groups']
        target = feedback.get('target_specialty')

        if decision == "ADD_HYPOTHESIS" and target:
            new_hypo = feedback.get('new_hypothesis_name')
            if new_hypo:
                if target not in specialty_groups: specialty_groups[target] = []
                specialty_groups[target].append({
                    "hypothesis": new_hypo, 
                    "specialty": target
                }) 
                
        elif decision == "DISCARD_HYPOTHESIS" and target:
            discard = feedback.get('hypothesis_to_discard')
            if discard and target in specialty_groups:
                specialty_groups[target] = [h for h in specialty_groups[target] if h['hypothesis'] != discard]

        return {"specialty_groups": specialty_groups}
    
    def run_synthesizer(state: CmarState):
        print("\n--- Node: Synthesizer ---")
        patient_summary = state['patient_scenario']['summary']
        specialty_groups = state['specialty_groups']
        ground_truth = state.get('ground_truth')
        
        report = synthesizer_agent.run(patient_summary, specialty_groups, ground_truth=ground_truth)
        return {"final_report": report}

    # --- EDGES ---
    def decide_next_step(state: CmarState):
        decision = state['critic_feedback']['decision']
        
        # HITL CHECK
        if decision == "ASK_HUMAN":
            print("-> 🗣️ Critic requires human input. Stopping graph.")
            return END
            
        if not enable_critic: return "synthesize"
        
        if state.get('refinement_loop_count', 0) >= max_loops:
            return "synthesize"
        
        if decision == 'APPROVE':
            return "synthesize"
        else:
            return "refine"

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
        
        workflow.add_conditional_edges(
            "critic",
            decide_next_step,
            {
                "refine": "update_hypotheses",
                "synthesize": "synthesizer",
                END: END
            }
        )
        workflow.add_edge("update_hypotheses", "evidence_evaluator")
    else:
        workflow.set_entry_point("hypothesis_generator")
        workflow.add_edge("hypothesis_generator", "evidence_evaluator")
        workflow.add_edge("evidence_evaluator", "risk_assessor")
        workflow.add_edge("risk_assessor", "synthesizer")
    
    workflow.add_edge("synthesizer", END)

    return workflow.compile()