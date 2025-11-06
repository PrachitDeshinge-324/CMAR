# app/graph.py
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from agents.hypothesis_generator import HypothesisGenerator
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Update the state. We no longer need the intermediate 'hypotheses' list.
class CmarState(TypedDict):
    patient_scenario: Dict
    specialty_groups: Dict[str, List[Dict]]

# 2. Build the simplified graph
def build_graph(llm_client: ChatGoogleGenerativeAI):
    """Builds the simplified LangGraph workflow."""
    # Instantiate the single agent needed for this phase
    hypothesis_agent = HypothesisGenerator(llm_client)

    # Define the node function
    def run_hypothesis_generator(state: CmarState):
        """Generates and groups hypotheses in a single step."""
        print("\n--- Executing Node: Generate and Group Hypotheses ---")
        patient_scenario = state['patient_scenario']
        
        specialty_groups = hypothesis_agent.run(patient_scenario)
        
        print("\n-> Generated and grouped hypotheses into specialties:")
        for specialty, group in specialty_groups.items():
            print(f"  - {specialty}: {[h['hypothesis'] for h in group]}")
            
        return {"specialty_groups": specialty_groups}

    # Define the graph structure
    workflow = StateGraph(CmarState)

    workflow.add_node("hypothesis_generator", run_hypothesis_generator)
    workflow.set_entry_point("hypothesis_generator")
    workflow.add_edge("hypothesis_generator", END)

    # Compile and return the graph
    app = workflow.compile()
    return app