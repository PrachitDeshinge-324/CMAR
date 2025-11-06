# agents/triage.py
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict

# Define the desired output structure
class SpecialtyGroup(BaseModel):
    specialty: str = Field(description="The medical specialty relevant to the hypotheses (e.g., Cardiology, Gastroenterology).")
    hypotheses: List[str] = Field(description="A list of hypothesis names that fall under this specialty.")

class TriageResult(BaseModel):
    groups: List[SpecialtyGroup]

class TriageAgent:
    """
    An agent that groups medical hypotheses by the relevant specialty
    to prepare them for parallel evaluation.
    """
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
        self.system_prompt = """You are an expert medical dispatcher. Your task is to analyze a list of differential diagnoses (hypotheses) and group them by the most relevant medical specialty.

        Common specialties include:
        - Cardiology
        - Gastroenterology
        - Pulmonology
        - Musculoskeletal
        - Neurology
        - Psychiatry

        Group the provided list of hypotheses into their respective specialties.
        """
        self.parser = PydanticOutputParser(pydantic_object=TriageResult)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Please group the following hypotheses:\n{hypotheses_list}\n\n{format_instructions}")
        ])
        self.chain = self.prompt | self.llm | self.parser

    def run(self, hypotheses: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Takes a list of hypothesis objects and returns a dictionary
        grouping them by specialty.
        """
        print("-> Running Triage Agent...")
        # Create a simple list of hypothesis names for the LLM
        hypothesis_names = [h['hypothesis'] for h in hypotheses]
        
        result = self.chain.invoke({
            "hypotheses_list": "\n".join(f"- {name}" for name in hypothesis_names),
            "format_instructions": self.parser.get_format_instructions()
        })
        
        # Re-structure the data for our state, mapping original hypothesis objects
        # to their new specialty groups.
        original_hypotheses_map = {h['hypothesis']: h for h in hypotheses}
        grouped_results = {}
        for group in result.groups:
            specialty = group.specialty
            # Collect the full hypothesis objects for this specialty
            grouped_results[specialty] = [
                original_hypotheses_map[name] for name in group.hypotheses if name in original_hypotheses_map
            ]
            
        return grouped_results