# agents/synthesizer.py
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict

# Define the structured output for the final report
class FinalDiagnosis(BaseModel):
    rank: int = Field(description="The final rank of the diagnosis, ordered by clinical urgency.")
    diagnosis: str = Field(description="The name of the potential medical condition.")
    severity: int = Field(description="The assessed severity score (1-10).")
    likelihood: int = Field(description="The assessed likelihood score (1-10).")
    justification: str = Field(description="A concise summary of the justification for this ranking, combining evidence and risk assessment.")

class FinalReport(BaseModel):
    patient_summary: str = Field(description="A brief summary of the patient's initial presentation.")
    differential_diagnoses: List[FinalDiagnosis]
    overall_assessment: str = Field(description="A concluding summary of the diagnostic process and key findings.")

class SynthesizerAgent:
    """
    An agent that synthesizes the complete analysis into a final, ranked report
    ordered by clinical urgency.
    """
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
        self.system_prompt = """You are a senior medical scribe and rapporteur. Your task is to synthesize a complex, multi-specialty diagnostic analysis into a single, clear, and actionable final report.

        The final list of differential diagnoses has already been ranked by a deterministic algorithm based on clinical urgency (severity first, then likelihood).

        Your main responsibilities are:
        1.  Accurately transcribe the provided ranked list of diagnoses.
        2.  For each diagnosis, write a concise, professional justification that synthesizes the key findings from the risk assessment.
        3.  Write a brief, high-level "Overall Assessment" that summarizes the most critical findings and explains why the top-ranked diagnoses are the most urgent.
        """
        self.parser = PydanticOutputParser(pydantic_object=FinalReport)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """
            Please synthesize the following ranked diagnostic data into the final report format.

            **Patient Scenario:**
            {patient_summary}

            **Ranked and Validated Differential Diagnoses:**
            {ranked_diagnoses}

            {format_instructions}
            """)
        ])
        self.chain = self.prompt | self.llm | self.parser

    def _rank_hypotheses(self, specialty_groups: Dict) -> List[Dict]:
        """
        Flattens and ranks all hypotheses based on clinical urgency.
        Ranking Logic: Severity (descending), then Likelihood (descending).
        """
        all_hypotheses = []
        for specialty in specialty_groups:
            all_hypotheses.extend(specialty_groups[specialty])

        # Filter out any hypotheses that might have failed assessment and lack scores
        valid_hypotheses = [
            h for h in all_hypotheses if 'severity' in h and 'likelihood' in h
        ]
        
        # The key ranking logic: sort by severity, then by likelihood, both descending.
        ranked_list = sorted(
            valid_hypotheses,
            key=lambda x: (x['severity'], x['likelihood']),
            reverse=True
        )
        return ranked_list

    def _format_ranked_list_for_prompt(self, ranked_list: List[Dict]) -> str:
        """Formats the ranked list into a string for the LLM prompt."""
        formatted_string = ""
        for i, hypo in enumerate(ranked_list):
            formatted_string += f"\n--- Rank {i+1} ---\n"
            formatted_string += f"Diagnosis: {hypo['hypothesis']}\n"
            formatted_string += f"Assigned Severity: {hypo['severity']}\n"
            formatted_string += f"Assigned Likelihood: {hypo['likelihood']}\n"
            formatted_string += f"Justification Notes: {hypo.get('risk_justification', 'N/A')}\n"
        return formatted_string

    def run(self, patient_summary: str, specialty_groups: Dict) -> Dict:
        """
        Runs the synthesizer agent to produce the final report.
        """
        print("-> Synthesizing and ranking the final report...")
        
        # 1. Perform deterministic ranking in Python
        ranked_diagnoses_data = self._rank_hypotheses(specialty_groups)
        
        # 2. Format the ranked list for the LLM to summarize
        ranked_diagnoses_prompt_str = self._format_ranked_list_for_prompt(ranked_diagnoses_data)
        
        # 3. Invoke the LLM to generate the human-readable report
        report = self.chain.invoke({
            "patient_summary": patient_summary,
            "ranked_diagnoses": ranked_diagnoses_prompt_str,
            "format_instructions": self.parser.get_format_instructions(),
        })
        
        return report.dict()