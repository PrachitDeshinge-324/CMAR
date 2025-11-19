# agents/critic.py
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict, Literal, Optional

class CriticDecision(BaseModel):
    decision: Literal["APPROVE", "CHALLENGE_SCORE", "ADD_HYPOTHESIS", "DISCARD_HYPOTHESIS", "ASK_HUMAN"] = Field(
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
    questions_for_human: Optional[List[str]] = Field(
        description="A list of the MOST CRITICAL questions (Max 5) to ask the clinician. Required if decision is ASK_HUMAN."
    )

class CriticAgent:
    """
    An advanced agent that reviews the analysis, identifies specific weaknesses,
    and issues targeted, actionable directives to refine the diagnosis.
    """
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
        self.system_prompt = """You are a meticulous Diagnostic Reviewer. Your goal is to ensure the differential diagnosis is rigorous, logically sound, and clinically safe.

        Review the patient's scenario and the specialists' list of hypotheses. Choose ONE action:

        1.  **APPROVE**: If the diagnosis is solid and supported by evidence.
        2.  **ASK_HUMAN**: **CRITICAL USE ONLY.** Use this if the case is ambiguous.
            - **CHECK FIRST:** Look at the *Patient Scenario* text. If you see the tag **'[Clinician Responses]'**, you have ALREADY asked the human. You are **FORBIDDEN** from using ASK_HUMAN again. You must make a decision with the info you have.
            - **WHEN TO ASK:** - If the *Chief Complaint* is broad (e.g., "Chest Pain", "Abdominal Pain") and lacks **OPQRST** details (Onset, Provocation, Quality, Radiation, Severity, Time), you **MUST** ask.
              - **Example:** "Intermittent Chest Pain" is too vague. You need to know: "Is it exertional? Relieved by rest? Radiating?"
              - **Vitals are NOT enough:** Even if BP/HR are provided, if the *history* is vague, you must ask.
            - **STRICT LIMIT:** Ask **Max 3-5** high-yield questions. Do not ask generic "Any other symptoms?" questions.
        3.  **ADD_HYPOTHESIS**: If a major, high-probability diagnosis is missing.
        4.  **CHALLENGE_SCORE**: If a score is contradicted by key facts.
        5.  **DISCARD_HYPOTHESIS**: If a diagnosis is anatomically impossible or ruled out.

        **PRIORITY SEQUENCE:**
        1. **One-Shot Check:** Does `[Clinician Responses]` exist? -> If YES, **STOP ASKING**.
        2. **Ambiguity Check:** Is the HPI (History of Present Illness) missing OPQRST details for the main symptom? -> **ASK_HUMAN**.
        3. **Safety Check:** Are red flags for life-threatening conditions (e.g., "Tearing pain" for dissection) unchecked? -> **ASK_HUMAN**.
        4. Otherwise -> Proceed with standard review.
        """
        self.parser = PydanticOutputParser(pydantic_object=CriticDecision)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """
            **Patient Scenario:**
            {patient_summary}

            **Previous Feedback:**
            {previous_feedback}

            **Current Analysis from Specialists:**
            {combined_analysis}

            Review the analysis. Remember: If `[Clinician Responses]` is present, do NOT ask again.
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
    
    def _format_previous_feedback(self, critic_history: List[Dict]) -> str:
        """Format previous critic feedback to help avoid repetition."""
        if not critic_history:
            return "None - This is the first iteration."
        
        formatted = ""
        for i, feedback in enumerate(critic_history, 1):
            formatted += f"\nIteration {i}:\n"
            formatted += f"  Decision: {feedback.get('decision')}\n"
            formatted += f"  Target: {feedback.get('target_specialty', 'N/A')}\n"
            formatted += f"  Feedback: {feedback.get('feedback', 'N/A')}\n"
        return formatted
    
    def _get_targeted_specialties(self, critic_history: List[Dict]) -> set:
        """Extract all specialties that have already been targeted by the critic."""
        if not critic_history:
            return set()
        return {h.get('target_specialty') for h in critic_history if h.get('target_specialty')}
    
    def _get_untargeted_specialties(self, specialty_groups: Dict, critic_history: List[Dict]) -> List[str]:
        """Get list of specialties that haven't been reviewed yet."""
        all_specialties = set(specialty_groups.keys())
        targeted = self._get_targeted_specialties(critic_history)
        untargeted = all_specialties - targeted
        return list(untargeted)

    def run(self, patient_summary: str, specialty_groups: Dict, critic_history: List[Dict] = None) -> Dict:
        """Runs the critic agent with structural memory to ensure specialty diversity."""
        print("-> Critic Agent is reviewing the analysis...")
        
        combined_analysis = self._format_analysis_for_prompt(specialty_groups)
        previous_feedback = self._format_previous_feedback(critic_history or [])
        
        result = self.chain.invoke({
            "patient_summary": patient_summary,
            "combined_analysis": combined_analysis,
            "previous_feedback": previous_feedback,
            "format_instructions": self.parser.get_format_instructions(),
        })
        
        return result.dict()