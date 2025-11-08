# agents/critic.py
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict, Literal, Optional

# Define the new, more powerful structured output for the critic's decision
class CriticDecision(BaseModel):
    decision: Literal["APPROVE", "CHALLENGE_SCORE", "ADD_HYPOTHESIS", "DISCARD_HYPOTHESIS"] = Field(
        description="The specific action the critic has decided to take."
    )
    target_specialty: Optional[str] = Field(
        description="The specialty group this decision applies to (e.g., 'Cardiology'). Required for all actions except APPROVE."
    )
    feedback: str = Field(
        description="Justification for the decision. If CHALLENGE_SCORE, a counterfactual question. If ADD/DISCARD, the reason why."
    )
    new_hypothesis_name: Optional[str] = Field(
        description="The name of the new hypothesis to add. Only required if decision is ADD_HYPOTHESIS."
    )
    hypothesis_to_discard: Optional[str] = Field(
        description="The exact name of the hypothesis to discard. Only required if decision is DISCARD_HYPOTHESIS."
    )

class CriticAgent:
    """
    An advanced agent that reviews the analysis, identifies specific weaknesses,
    and issues targeted, actionable directives to refine the diagnosis.
    """
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
        self.system_prompt = """You are a meticulous Chief Medical Officer. Your goal is to find the single most impactful weakness in the current diagnostic analysis and issue a precise directive to fix it.

        Review the patient's situation and the specialists' findings. Choose ONE of the following actions:

        1.  **APPROVE**: If the analysis is sound and comprehensive.
        2.  **CHALLENGE_SCORE**: If a likelihood/severity score seems wrong. Ask a targeted counterfactual question to a specific specialty.
            (Example: target='Cardiology', feedback='You rated MI likelihood high, but what if the intermittent nature of the pain makes angina more likely?')
        3.  **ADD_HYPOTHESIS**: If a plausible diagnosis is missing. Specify the new hypothesis and the specialty to add it to.
            (Example: target='Gastroenterology', new_hypothesis='Esophagitis', feedback='The evidence points to esophageal issues, but Esophagitis itself has not been considered.')
        4.  **DISCARD_HYPOTHESIS**: If a diagnosis is clearly unsupported or redundant. Specify the hypothesis to remove.
            (Example: target='Musculoskeletal', hypothesis_to_discard='Costochondritis', feedback='The evidence found has zero relevance to this diagnosis, making it noise.')

        Your directive must be targeted to a single specialty.
        """
        self.parser = PydanticOutputParser(pydantic_object=CriticDecision)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """
            **Patient Scenario:**
            {patient_summary}

            **Combined Analysis from Specialists:**
            {combined_analysis}

            Review the analysis and provide your single, most impactful directive.
            {format_instructions}
            """)
        ])
        self.chain = self.prompt | self.llm | self.parser

    def _format_analysis_for_prompt(self, specialty_groups: Dict) -> str:
        formatted_string = ""
        for specialty, hypotheses in specialty_groups.items():
            formatted_string += f"\n--- Specialty: {specialty} ---\n"
            for hypo in hypotheses:
                formatted_string += (
                    f"  - Hypothesis: {hypo.get('hypothesis', 'N/A')}\n"
                    f"    - Severity: {hypo.get('severity', 'N/A')}\n"
                    f"    - Likelihood: {hypo.get('likelihood', 'N/A')}\n"
                    f"    - Justification: {hypo.get('risk_justification', 'N/A')}\n"
                )
        return formatted_string

    def run(self, patient_summary: str, specialty_groups: Dict) -> Dict:
        """Runs the critic agent and returns its structured decision."""
        print("-> Critic Agent is reviewing the analysis for targeted feedback...")
        combined_analysis = self._format_analysis_for_prompt(specialty_groups)
        
        result = self.chain.invoke({
            "patient_summary": patient_summary,
            "combined_analysis": combined_analysis,
            "format_instructions": self.parser.get_format_instructions(),
        })
        
        return result.dict()