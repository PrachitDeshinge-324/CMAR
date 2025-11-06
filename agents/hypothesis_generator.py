# agents/hypothesis_generator.py
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict
from collections import defaultdict

# Define the new desired structure for each item
class Hypothesis(BaseModel):
    hypothesis: str = Field(description="The name of the potential medical condition or diagnosis.")
    specialty: str = Field(description="The medical specialty most relevant to this hypothesis (e.g., Cardiology, Gastroenterology).")

# The top-level list remains the same
class HypothesisList(BaseModel):
    hypotheses: List[Hypothesis]

class HypothesisGenerator:
    """
    An agent that generates an initial list of differential diagnoses
    and groups them by the relevant medical specialty.
    """
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
        self.system_prompt = """You are an expert medical diagnostician. Your role is to analyze a patient scenario and generate a broad list of potential differential diagnoses.

        For each hypothesis, you must also identify the primary medical specialty that would handle it. Common specialties include Cardiology, Gastroenterology, Pulmonology, Musculoskeletal, Psychiatry, Neurology, Gynecology, Dermatology, Emergency Medicine, and more.

        Based on the user-provided patient summary, generate a list of possible hypotheses, each with its corresponding specialty.
        """
        self.parser = PydanticOutputParser(pydantic_object=HypothesisList)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Patient Scenario:\n{patient_summary}\n\n{format_instructions}")
        ])
        self.chain = self.prompt | self.llm | self.parser

    def run(self, patient_scenario: Dict) -> Dict[str, List[Dict]]:
        """
        Analyzes the patient scenario and returns a dictionary of hypotheses
        grouped by specialty.
        """
        print("-> Running Combined Hypothesis Generator & Triage...")
        result = self.chain.invoke({
            "patient_summary": patient_scenario['summary'],
            "format_instructions": self.parser.get_format_instructions()
        })
        
        # Perform the grouping logic right here
        grouped_hypotheses = defaultdict(list)
        for hypothesis_obj in result.hypotheses:
            # Convert the Pydantic object to a dictionary
            hypothesis_dict = hypothesis_obj.dict()
            specialty = hypothesis_dict['specialty']
            grouped_hypotheses[specialty].append(hypothesis_dict)
            
        # Convert defaultdict back to a regular dict for the state
        return dict(grouped_hypotheses)