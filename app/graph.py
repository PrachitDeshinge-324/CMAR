# app/graph.py
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from concurrent.futures import ThreadPoolExecutor # We need this back!

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
    """
    Builds the CMAR LangGraph with advanced, targeted counterfactual loops.
    
    Args:
        llm_client: The LLM client to use
        embeddings_client: The embeddings client to use
        optimization_config: Optional dict with optimization settings:
            - enable_critic_loop: bool (default True)
            - max_refinement_loops: int (default 3)
            - batch_risk_assessment: bool (default False)
    """
    # Apply optimization defaults
    if optimization_config is None:
        optimization_config = {}
    
    enable_critic = optimization_config.get('enable_critic_loop', True)
    max_loops = optimization_config.get('max_refinement_loops', 3)
    batch_risk = optimization_config.get('batch_risk_assessment', False)
    
    print(f"-> Graph optimization settings:")
    print(f"   - Critic loop: {'enabled' if enable_critic else 'DISABLED (saves ~3 API calls/patient)'}")
    print(f"   - Max refinement loops: {max_loops}")
    print(f"   - Batch risk assessment: {'enabled (saves ~9 API calls/patient)' if batch_risk else 'disabled'}")

    # --- AGENT AND TOOL INITIALIZATION ---
    retriever_tool = StaticRetriever(embeddings=embeddings_client).get_retriever()
    web_search_tool = DuckDuckGoSearchRun()
    hypothesis_agent = HypothesisGenerator(llm_client)
    evidence_evaluator_agent = EvidenceEvaluatorAgent(retriever_tool=retriever_tool, web_search_tool=web_search_tool)
    risk_assessor_agent = RiskAssessorAgent(llm_client)
    critic_agent = CriticAgent(llm_client)
    synthesizer_agent = SynthesizerAgent(llm_client) 

    # --- NODE FUNCTION DEFINITIONS ---

    def run_hypothesis_generator(state: CmarState):
        print("\n--- Executing Node: Generate and Group Hypotheses ---")
        patient_scenario = state['patient_scenario']
        specialty_groups = hypothesis_agent.run(patient_scenario['summary'])
        return {
            "specialty_groups": specialty_groups, 
            "refinement_loop_count": 0,
            "critic_history": []
        }

    def run_evidence_evaluation(state: CmarState):
        print("\n--- Executing Node: Sequential Evidence Evaluation ---")
        patient_summary = state['patient_scenario']['summary']
        specialty_groups = state['specialty_groups']
        critic_feedback = state.get('critic_feedback', {})
        targeted_specialty = critic_feedback.get('target_specialty')
        
        updated_groups = {}
        
        for specialty, hypotheses in specialty_groups.items():
            # Check if this specialty was targeted by the critic
            is_critic_target = (specialty == targeted_specialty)
            
            # Re-evaluate if:
            # 1. There are new hypotheses without evidence, OR
            # 2. This specialty was targeted by the critic (to get fresh evidence)
            hypotheses_to_evaluate = [h for h in hypotheses if 'evidence' not in h]
            
            if hypotheses_to_evaluate or is_critic_target:
                if is_critic_target and not hypotheses_to_evaluate:
                    # Critic targeted this specialty - re-evaluate all hypotheses
                    print(f"  -> Re-evaluating specialty: {specialty} (targeted by critic)")
                    hypotheses_to_evaluate = hypotheses
                else:
                    print(f"  -> Evaluating specialty: {specialty} ({len(hypotheses_to_evaluate)} hypotheses)")
                
                updated_hypotheses = evidence_evaluator_agent.evaluate_hypotheses_for_specialty(
                    patient_summary, hypotheses_to_evaluate
                )
                # Merge back with already evaluated hypotheses
                evaluated_map = {h['hypothesis']: h for h in updated_hypotheses}
                final_hypotheses = [evaluated_map.get(h['hypothesis'], h) for h in hypotheses]
                updated_groups[specialty] = final_hypotheses
            else:
                # Skip if no new hypotheses and not critic target
                updated_groups[specialty] = hypotheses
        
        return {"specialty_groups": updated_groups}

    def run_risk_assessment(state: CmarState):
        """Node 3: BATCH risk assessment - all specialties in ONE API call to save quota."""
        print("\n--- Executing Node: Batch Risk Assessment (1 API Call) ---")
        patient_summary = state['patient_scenario']['summary']
        specialty_groups = state['specialty_groups']
        feedback = state.get('critic_feedback', {})
        target_specialty = feedback.get('target_specialty')
        challenge = feedback.get('feedback', "None. Initial assessment.")

        # Combine ALL specialties into a single API call
        all_hypotheses = []
        specialty_map = {}  # Track which hypotheses belong to which specialty
        
        for specialty, hypotheses in specialty_groups.items():
            for hypo in hypotheses:
                hypo['_original_specialty'] = specialty  # Temporary marker
                all_hypotheses.append(hypo)
        
        # Single batched API call for all hypotheses
        try:
            print(f"  -> Assessing {len(all_hypotheses)} hypotheses across {len(specialty_groups)} specialties in 1 API call...")
            updated_all = risk_assessor_agent.assess_risk_for_specialty(
                patient_summary, 
                all_hypotheses, 
                f"Target specialty: {target_specialty}. Challenge: {challenge}" if target_specialty else "None. Initial assessment."
            )
            
            # Re-group by specialty
            updated_groups = {}
            for hypo in updated_all:
                specialty = hypo.pop('_original_specialty')  # Remove temporary marker
                if specialty not in updated_groups:
                    updated_groups[specialty] = []
                updated_groups[specialty].append(hypo)
            
            print(f"  ✅ Batch risk assessment completed successfully")
            
        except Exception as exc:
            print(f"  ⚠️ Batch risk assessment encountered an error: {str(exc)[:200]}")
            # Fallback to default scores
            updated_groups = {}
            for specialty, hypotheses in specialty_groups.items():
                fallback_hypotheses = []
                for hypo in hypotheses:
                    if 'severity' not in hypo:
                        hypo['severity'] = 5
                    if 'likelihood' not in hypo:
                        hypo['likelihood'] = 5
                    if 'risk_justification' not in hypo:
                        hypo['risk_justification'] = f"Risk assessment failed: {str(exc)[:100]}"
                    fallback_hypotheses.append(hypo)
                updated_groups[specialty] = fallback_hypotheses
        
        return {"specialty_groups": updated_groups}

    def run_critic(state: CmarState):
        print("\n--- Executing Node: Critic Review ---")
        patient_summary = state['patient_scenario']['summary']
        specialty_groups = state['specialty_groups']
        critic_history = state.get('critic_history', [])
        
        # Pass critic history to avoid repeating same critiques
        decision = critic_agent.run(patient_summary, specialty_groups, critic_history)
        
        print(f"-> Critic's Directive: {decision['decision']}")
        if decision['decision'] != 'APPROVE':
            print(f"-> Target: {decision.get('target_specialty')}")
            print(f"-> Justification: {decision.get('feedback')}")
        
        loop_count = state.get('refinement_loop_count', 0) + 1
        
        # Add iteration number to the decision and append to history
        critic_feedback_with_iteration = {
            **decision,
            'iteration': loop_count
        }
        
        critic_history = state.get('critic_history', [])
        critic_history.append(critic_feedback_with_iteration)
        
        return {
            "critic_feedback": decision, 
            "critic_history": critic_history,
            "refinement_loop_count": loop_count
        }
        
    # --- NEW NODE FOR DYNAMIC HYPOTHESIS MANAGEMENT ---
    def update_hypotheses(state: CmarState):
        """Node that acts on the critic's directive to add or remove hypotheses."""
        print("\n--- Executing Node: Update Hypotheses List ---")
        feedback = state.get('critic_feedback', {})
        decision = feedback.get('decision')
        specialty_groups = state['specialty_groups']
        target_specialty = feedback.get('target_specialty')

        if decision == "ADD_HYPOTHESIS":
            new_hypo_name = feedback.get('new_hypothesis_name')
            if target_specialty and new_hypo_name and target_specialty in specialty_groups:
                print(f"-> Adding '{new_hypo_name}' to {target_specialty}.")
                specialty_groups[target_specialty].append({
                    "hypothesis": new_hypo_name, "specialty": target_specialty
                })
        elif decision == "DISCARD_HYPOTHESIS":
            hypo_to_discard = feedback.get('hypothesis_to_discard')
            if target_specialty and hypo_to_discard and target_specialty in specialty_groups:
                print(f"-> Discarding '{hypo_to_discard}' from {target_specialty}.")
                specialty_groups[target_specialty] = [
                    h for h in specialty_groups[target_specialty] if h['hypothesis'] != hypo_to_discard
                ]
        else:
            print("-> No changes to hypothesis list required.")

        return {"specialty_groups": specialty_groups}
    
    def run_synthesizer(state: CmarState):
        """Node 5: Ranks the final hypotheses and generates the final report."""
        print("\n--- Executing Node: Synthesizer ---")
        patient_summary = state['patient_scenario']['summary']
        specialty_groups = state['specialty_groups']
        ground_truth = state.get('ground_truth')  # Get ground truth if provided
        
        # Pass ground truth to synthesizer for validation
        report = synthesizer_agent.run(
            patient_summary, 
            specialty_groups,
            ground_truth=ground_truth
        )
        
        print("\n--- FINAL REPORT GENERATED ---")
        return {"final_report": report}

    # --- CONDITIONAL EDGE LOGIC ---
    def decide_next_step_after_critic(state: CmarState):
        print("--- Checking Critic's Directive ---")
        
        # Check if critic loop is disabled
        if not enable_critic:
            print("-> Critic loop DISABLED. Proceeding directly to Synthesizer.")
            return "synthesize"
        
        if state.get('refinement_loop_count', 0) >= max_loops:
            print(f"-> Maximum refinement loops ({max_loops}) reached. Proceeding to Synthesizer.")
            return "synthesize"
        
        if state['critic_feedback']['decision'] != 'APPROVE':
            print("-> Directive is a refinement. Looping back.")
            return "refine"
        else:
            print("-> Directive is APPROVE. Proceeding to Synthesizer.")
            return "synthesize"
        
    # --- GRAPH WIRING ---
    workflow = StateGraph(CmarState)
    workflow.add_node("hypothesis_generator", run_hypothesis_generator)
    workflow.add_node("evidence_evaluator", run_evidence_evaluation)
    workflow.add_node("risk_assessor", run_risk_assessment)
    workflow.add_node("synthesizer", run_synthesizer)
    
    workflow.set_entry_point("hypothesis_generator")
    workflow.add_edge("hypothesis_generator", "evidence_evaluator")
    workflow.add_edge("evidence_evaluator", "risk_assessor")
    
    if enable_critic:
        # Full workflow with critic loop
        workflow.add_node("critic", run_critic)
        workflow.add_node("update_hypotheses", update_hypotheses)
        workflow.add_edge("risk_assessor", "critic")
        workflow.add_conditional_edges(
            "critic",
            decide_next_step_after_critic,
            {
                "refine": "update_hypotheses",
                "synthesize": "synthesizer"
            }
        )
        workflow.add_edge("update_hypotheses", "evidence_evaluator")
    else:
        # Simplified workflow: skip critic, go straight to synthesizer
        workflow.add_edge("risk_assessor", "synthesizer")
    
    workflow.add_edge("synthesizer", END)

    return workflow.compile()