# app/graph.py
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END

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
    critic_history: List[Dict]  # Track all critic feedback across iterations
    refinement_loop_count: int
    final_report: Dict

# 2. DEFINE THE GRAPH BUILDER
def build_graph(llm_client: ChatGoogleGenerativeAI, embeddings_client: HuggingFaceEmbeddings):
    """
    Builds the CMAR LangGraph with advanced, targeted counterfactual loops.
    """

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
        specialty_groups = hypothesis_agent.run(patient_scenario)
        return {
            "specialty_groups": specialty_groups, 
            "refinement_loop_count": 0,
            "critic_history": []  # Initialize critic history
        }

    def run_evidence_evaluation(state: CmarState):
        print("\n--- Executing Node: Sequential Evidence Evaluation ---")
        patient_summary = state['patient_scenario']['summary']
        specialty_groups = state['specialty_groups']
        updated_groups = {}
        for specialty, hypotheses in specialty_groups.items():
            print(f"  -> Evaluating specialty: {specialty}")
            # We only re-evaluate if the hypothesis doesn't already have evidence
            # or if the critic has modified this specialty group.
            hypotheses_to_evaluate = [h for h in hypotheses if 'evidence' not in h]
            if hypotheses_to_evaluate:
                updated_hypotheses = evidence_evaluator_agent.evaluate_hypotheses_for_specialty(patient_summary, hypotheses_to_evaluate)
                # Merge back with already evaluated hypotheses
                evaluated_map = {h['hypothesis']: h for h in updated_hypotheses}
                final_hypotheses = [evaluated_map.get(h['hypothesis'], h) for h in hypotheses]
                updated_groups[specialty] = final_hypotheses
            else:
                updated_groups[specialty] = hypotheses # No change needed
        return {"specialty_groups": updated_groups}

    def run_risk_assessment(state: CmarState):
        print("\n--- Executing Node: Sequential Risk Assessment (Rate-Limited) ---")
        patient_summary = state['patient_scenario']['summary']
        specialty_groups = state['specialty_groups']
        feedback = state.get('critic_feedback', {})
        target_specialty = feedback.get('target_specialty')
        challenge = feedback.get('feedback', "None. Initial assessment.")
        
        updated_groups = {}
        for specialty, hypotheses in specialty_groups.items():
            # Apply the challenge only to the targeted specialty
            current_challenge = challenge if specialty == target_specialty else "None. Initial assessment."
            try:
                print(f"  -> Assessing risk for specialty: {specialty}")
                updated_hypotheses = risk_assessor_agent.assess_risk_for_specialty(
                    patient_summary, hypotheses, current_challenge
                )
                updated_groups[specialty] = updated_hypotheses
                print(f"  -> Finished risk assessment for specialty: {specialty}")
            except Exception as exc:
                print(f"  -> Risk assessment for {specialty} generated an exception: {exc}")
                updated_groups[specialty] = specialty_groups[specialty]
        
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
        
        report = synthesizer_agent.run(patient_summary, specialty_groups)
        
        print("\n--- FINAL REPORT GENERATED ---")
        return {"final_report": report}

    # --- CONDITIONAL EDGE LOGIC ---
    def decide_next_step_after_critic(state: CmarState):
        print("--- Checking Critic's Directive ---")
        if state.get('refinement_loop_count', 0) >= 3:
            print("-> Maximum refinement loops reached. Proceeding to Synthesizer.")
            return "synthesize" # Change "end" to "synthesize"
        
        if state['critic_feedback']['decision'] != 'APPROVE':
            print("-> Directive is a refinement. Looping back.")
            return "refine"
        else:
            print("-> Directive is APPROVE. Proceeding to Synthesizer.")
            return "synthesize" # Change "end" to "synthesize"
        
    # --- GRAPH WIRING ---
    workflow = StateGraph(CmarState)
    workflow.add_node("hypothesis_generator", run_hypothesis_generator)
    workflow.add_node("evidence_evaluator", run_evidence_evaluation)
    workflow.add_node("risk_assessor", run_risk_assessment)
    workflow.add_node("critic", run_critic)
    workflow.add_node("update_hypotheses", update_hypotheses)
    workflow.add_node("synthesizer", run_synthesizer) # <-- Add new synthesizer node

    workflow.set_entry_point("hypothesis_generator")
    workflow.add_edge("hypothesis_generator", "evidence_evaluator")
    workflow.add_edge("evidence_evaluator", "risk_assessor")
    workflow.add_edge("risk_assessor", "critic")

    workflow.add_conditional_edges(
        "critic",
        decide_next_step_after_critic,
        {
            "refine": "update_hypotheses",
            "synthesize": "synthesizer" # <-- The new final path
        }
    )
    workflow.add_edge("update_hypotheses", "evidence_evaluator")
    workflow.add_edge("synthesizer", END) # <-- The synthesizer is the new end point

    return workflow.compile()