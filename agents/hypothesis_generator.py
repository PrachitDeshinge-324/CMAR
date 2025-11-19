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
        self.system_prompt = """You are an expert medical diagnostician. Your role is to analyze a patient scenario and generate a focused list of the MOST LIKELY differential diagnoses.

        **CRITICAL INSTRUCTIONS:**
        1.  **Focus on Most Relevant:** Generate 10-15 total hypotheses across the MOST relevant specialties (aim for 6-8 specialties maximum).
        2.  **Prioritize by Likelihood:** Focus on diagnoses that best match the patient's symptoms and context.
        3.  **Consider BOTH Acute AND Chronic/Lifestyle Conditions:**
            - ACUTE: Immediate life-threatening or emergency conditions
            - CHRONIC: Long-term conditions that may be exacerbating (e.g., COPD, Chronic Bronchitis, Varicocele)
            - LIFESTYLE-RELATED: Conditions related to patient's history (alcohol → cirrhosis, smoking → COPD)
        4.  **Pay Special Attention to Patient History:**
            - "Alcohol/drug abuse" → consider chronic liver disease, withdrawal, COPD, chronic bronchitis
            - "Smoker" → consider COPD, chronic bronchitis, lung cancer
            - "Diabetes" → consider neuropathy, nephropathy, vascular disease
        5.  **Assign Specialty:** For each hypothesis, identify the primary medical specialty (e.g., Cardiology, Neurology, Internal Medicine).
        6.  **Balanced Differential:** Include mix of high-severity acute AND high-likelihood chronic conditions.
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