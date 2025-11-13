# agents/hypothesis_generator.py
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict
from collections import defaultdict

# --- (Hypothesis and HypothesisList classes are unchanged) ---
class Hypothesis(BaseModel):
    hypothesis: str = Field(description="The name of the potential medical condition or diagnosis.")
    specialty: str = Field(description="The medical specialty most relevant to this hypothesis (e.g., Cardiology, Pulmonology).")

class HypothesisList(BaseModel):
    hypotheses: List[Hypothesis]


class HypothesisGenerator:
    """
    An agent that generates an initial list of differential diagnoses
    and groups them by the relevant medical specialty.
    """
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
        # --- THIS IS THE KEY CHANGE ---
        self.system_prompt = """You are an expert medical diagnostician. Your role is to analyze a patient scenario and generate a broad list of potential differential diagnoses.

        **CRITICAL INSTRUCTIONS:**
        1.  **Think Broadly:** Consider possibilities from all relevant medical specialties.
        2.  **Consider Both Acute and Chronic Conditions:** A patient's history (like 'history of alcohol abuse') can suggest long-term, chronic diseases (like Cirrhosis or Chronic Bronchitis) in addition to acute emergencies (like Pancreatitis or Overdose). You MUST include both types of possibilities in your differential.
        3.  **Assign Specialty:** For each hypothesis, identify the primary medical specialty.
        """
        self.parser = PydanticOutputParser(pydantic_object=HypothesisList)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Patient Scenario:\n{patient_summary}\n\n{format_instructions}")
        ])
        self.chain = self.prompt | self.llm | self.parser

    def run(self, patient_summary: str) -> Dict[str, List[Dict]]: # Changed input to be a simple string
        """
        Analyzes the patient scenario and returns a dictionary of hypotheses
        grouped by specialty.
        """
        print("-> Running Combined Hypothesis Generator & Triage...")
        result = self.chain.invoke({
            "patient_summary": patient_summary, # Pass the string directly
            "format_instructions": self.parser.get_format_instructions()
        })
        
        grouped_hypotheses = defaultdict(list)
        for hypothesis_obj in result.hypotheses:
            hypothesis_dict = hypothesis_obj.dict()
            specialty = hypothesis_dict['specialty']
            grouped_hypotheses[specialty].append(hypothesis_dict)
            
        return dict(grouped_hypotheses)