# agents/hypothesis_generator.py
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict, Optional
from collections import defaultdict

class Hypothesis(BaseModel):
    hypothesis: str = Field(description="The exact text of the selected option.")
    specialty: str = Field(description="The relevant medical specialty.")

class HypothesisList(BaseModel):
    hypotheses: List[Hypothesis]

class HypothesisGenerator:
    def __init__(self, llm: ChatGoogleGenerativeAI, prompt_override: Optional[str] = None):
        self.llm = llm
        
        # --- DEFAULT (Real World) ---
        default_prompt = """You are an expert medical diagnostician. 
        Analyze the patient scenario and generate 5-8 differential diagnoses based on the Chief Complaint.
        """
        
        # --- BENCHMARK (Strict Option Picker) ---
        # Only used if passed via prompt_override (controlled by graph.py)
        self.system_prompt = prompt_override if prompt_override else default_prompt
        
        self.parser = PydanticOutputParser(pydantic_object=HypothesisList)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Patient Scenario:\n{patient_summary}\n\n{format_instructions}")
        ])
        self.chain = self.prompt | self.llm | self.parser

    def run(self, patient_summary: str) -> Dict[str, List[Dict]]:
        print("-> Running Hypothesis Generator...")
        try:
            result = self.chain.invoke({
                "patient_summary": patient_summary,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            grouped_hypotheses = defaultdict(list)
            for hypothesis_obj in result.hypotheses:
                hypothesis_dict = hypothesis_obj.dict()
                specialty = hypothesis_dict['specialty']
                grouped_hypotheses[specialty].append(hypothesis_dict)
                
            return dict(grouped_hypotheses)
            
        except Exception as e:
            print(f"❌ Error in Hypothesis Generator: {e}")
            return {}