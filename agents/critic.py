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

        **CRITICAL - CHECK FOR MISSING CHRONIC/LIFESTYLE CONDITIONS:**
        Before approving, verify that the differential includes BOTH acute and chronic conditions when relevant:
        - **Patient with alcohol/drug abuse history + respiratory symptoms** → Check for: Chronic Bronchitis, COPD, Aspiration Pneumonia, Chronic Lung Disease
        - **Patient with smoking history** → Check for: COPD, Chronic Bronchitis, Lung Cancer
        - **Patient with testicular swelling + infertility concerns** → Check for: Varicocele, Hydrocele, Spermatocele
        - **Patient with diabetes** → Check for: Diabetic Neuropathy, Nephropathy, Retinopathy
        - **Patient with chronic GI symptoms** → Check for: IBD, Chronic Gastritis, Peptic Ulcer Disease
        
        If a common chronic condition related to patient history is missing, use ADD_HYPOTHESIS to include it.

        IMPORTANT: 
        - Your directive must be targeted to a single specialty.
        - Review your previous feedback to avoid repeating the same critiques.
        - Target DIFFERENT specialties across iterations to ensure comprehensive review.
        - If you've already challenged a specialty, consider other specialties unless critical issues remain.
        - Prioritize adding missing chronic/lifestyle-related conditions over challenging scores.
        """
        self.parser = PydanticOutputParser(pydantic_object=CriticDecision)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """
            **Patient Scenario:**
            {patient_summary}

            **Previous Feedback (avoid repeating):**
            {previous_feedback}

            **Current Analysis from Specialists:**
            {combined_analysis}

            Review the analysis and provide your single, most impactful directive.
            Focus on a DIFFERENT specialty than previously targeted if possible.
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
        print("-> Critic Agent is reviewing the analysis for targeted feedback...")
        
        # Build specialty guidance for the LLM
        targeted_specialties = self._get_targeted_specialties(critic_history or [])
        untargeted_specialties = self._get_untargeted_specialties(specialty_groups, critic_history or [])
        
        if targeted_specialties:
            print(f"   Previously reviewed: {', '.join(sorted(targeted_specialties))}")
        if untargeted_specialties:
            print(f"   Not yet reviewed: {', '.join(sorted(untargeted_specialties))}")
        
        # Build enhanced guidance string
        specialty_guidance = ""
        if untargeted_specialties:
            specialty_guidance = f"\n\n**PRIORITY: Focus on these UNTARGETED specialties:** {', '.join(untargeted_specialties)}"
            specialty_guidance += "\nYou should target one of these unless a critical issue exists in an already-reviewed specialty."
        elif targeted_specialties:
            specialty_guidance = f"\n\n**NOTE: All specialties have been reviewed at least once.**"
            specialty_guidance += "\nOnly provide additional feedback if a critical issue remains unaddressed."
        
        combined_analysis = self._format_analysis_for_prompt(specialty_groups)
        previous_feedback = self._format_previous_feedback(critic_history or [])
        
        result = self.chain.invoke({
            "patient_summary": patient_summary,
            "combined_analysis": combined_analysis,
            "previous_feedback": previous_feedback + specialty_guidance,
            "format_instructions": self.parser.get_format_instructions(),
        })
        
        return result.dict()