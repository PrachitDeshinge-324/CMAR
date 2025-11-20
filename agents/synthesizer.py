# agents/synthesizer.py
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict, Optional

class FinalDiagnosis(BaseModel):
    rank: int = Field(description="Rank")
    diagnosis: str = Field(description="Name of the option/diagnosis")
    severity: int = Field(description="Score")
    likelihood: int = Field(description="Score")
    justification: str = Field(description="Summary")

class FinalReport(BaseModel):
    patient_summary: str = Field(description="Summary")
    differential_diagnoses: List[FinalDiagnosis]
    overall_assessment: str = Field(description="Conclusion")
    ground_truth_validation: Optional[Dict] = Field(default=None, description="Ignore for this task")

class SynthesizerAgent:
    def __init__(self, llm: ChatGoogleGenerativeAI, embeddings_model=None):
        self.llm = llm
        
        self.system_prompt = """You are a final adjudicator.
        Review the ranked list of options/diagnoses.
        Output a Final Report formatted exactly as requested.
        
        **CRITICAL:** - You will receive 'Likelihood' and 'Severity' scores for each item. 
        - You MUST transcribe these numbers EXACTLY as provided. Do not invent or change them.
        """
        
        self.parser = PydanticOutputParser(pydantic_object=FinalReport)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """
            **Patient Scenario:**
            {patient_summary}

            **Ranked Options:**
            {ranked_diagnoses}

            {format_instructions}
            """)
        ])
        self.chain = self.prompt | self.llm | self.parser

    def _rank_hypotheses(self, specialty_groups: Dict) -> List[Dict]:
        all_hypotheses = []
        for specialty in specialty_groups:
            all_hypotheses.extend(specialty_groups[specialty])

        # Filter valid
        valid_hypotheses = [h for h in all_hypotheses if 'likelihood' in h]
        
        # RANKING LOGIC:
        # Sort by Likelihood primarily (Accuracy), then Severity (Urgency) as tie-breaker
        ranked_list = sorted(
            valid_hypotheses,
            key=lambda x: (x['likelihood'], x.get('severity', 0)),
            reverse=True
        )
        return ranked_list

    def _format_ranked_list_for_prompt(self, ranked_list: List[Dict]) -> str:
        formatted_string = ""
        for i, hypo in enumerate(ranked_list):
            formatted_string += f"\n--- Rank {i+1} ---\n"
            formatted_string += f"Diagnosis: {hypo['hypothesis']}\n"
            # --- FIX IS HERE: Explicitly include BOTH scores ---
            formatted_string += f"Likelihood: {hypo['likelihood']}/10\n"
            formatted_string += f"Severity: {hypo.get('severity', 0)}/10\n" 
            formatted_string += f"Notes: {hypo.get('risk_justification', 'N/A')}\n"
        return formatted_string

    def run(self, patient_summary: str, specialty_groups: Dict, ground_truth: Optional[str] = None) -> Dict:
        print("-> Synthesizing final report...")
        
        ranked_diagnoses_data = self._rank_hypotheses(specialty_groups)
        ranked_diagnoses_prompt_str = self._format_ranked_list_for_prompt(ranked_diagnoses_data[:10])
        
        try:
            report = self.chain.invoke({
                "patient_summary": patient_summary,
                "ranked_diagnoses": ranked_diagnoses_prompt_str,
                "format_instructions": self.parser.get_format_instructions(),
            })
            return report.dict()
        except Exception as e:
            print(f"❌ Synthesizer Error: {e}")
            return {}