# app/graph.py
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from concurrent.futures import ThreadPoolExecutor

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI

from agents.hypothesis_generator import HypothesisGenerator
from agents.evidence_evaluator import EvidenceEvaluatorAgent # <-- Import new agent
from tools.medical_retriever import StaticRetriever       # <-- Import new retriever tool

class CmarState(TypedDict):
    patient_scenario: Dict
    specialty_groups: Dict[str, List[Dict]]

def build_graph(llm_client: ChatGoogleGenerativeAI, embeddings_client: HuggingFaceEmbeddings):
    """Builds the LangGraph workflow."""
    
    # 1. Instantiate Tools and Agents
    retriever_tool = StaticRetriever(embeddings=embeddings_client).get_retriever()
    web_search_tool = DuckDuckGoSearchRun()
    
    hypothesis_agent = HypothesisGenerator(llm_client)
    evidence_evaluator_agent = EvidenceEvaluatorAgent(
        retriever_tool=retriever_tool,
        web_search_tool=web_search_tool
    )

    # 2. Define Node Functions
    def run_hypothesis_generator(state: CmarState):
        # (This function is unchanged)
        print("\n--- Executing Node: Generate and Group Hypotheses ---")
        patient_scenario = state['patient_scenario']
        specialty_groups = hypothesis_agent.run(patient_scenario)
        print("\n-> Generated and grouped hypotheses.")
        return {"specialty_groups": specialty_groups}

    def run_evidence_evaluation(state: CmarState):
        """
        Runs the evidence evaluation for each specialty in parallel.
        """
        print("\n--- Executing Node: Parallel Evidence Evaluation ---")
        patient_summary = state['patient_scenario']['summary']
        specialty_groups = state['specialty_groups']
        
        # Use a ThreadPoolExecutor to run evaluations concurrently
        with ThreadPoolExecutor() as executor:
            # Create a future for each specialty evaluation
            future_to_specialty = {
                executor.submit(
                    evidence_evaluator_agent.evaluate_hypotheses_for_specialty,
                    patient_summary,
                    hypotheses
                ): specialty
                for specialty, hypotheses in specialty_groups.items()
            }
            
            updated_groups = {}
            for future in future_to_specialty:
                specialty = future_to_specialty[future]
                try:
                    # Get the result (the list of updated hypotheses)
                    updated_hypotheses = future.result()
                    updated_groups[specialty] = updated_hypotheses
                    print(f"  -> Finished evaluation for specialty: {specialty}")
                except Exception as exc:
                    print(f"  -> Evaluation for {specialty} generated an exception: {exc}")
                    updated_groups[specialty] = specialty_groups[specialty] # Keep original on error

        return {"specialty_groups": updated_groups}

    # 3. Define the Graph Structure
    workflow = StateGraph(CmarState)
    workflow.add_node("hypothesis_generator", run_hypothesis_generator)
    workflow.add_node("evidence_evaluator", run_evidence_evaluation)

    workflow.set_entry_point("hypothesis_generator")
    workflow.add_edge("hypothesis_generator", "evidence_evaluator")
    workflow.add_edge("evidence_evaluator", END)

    return workflow.compile()