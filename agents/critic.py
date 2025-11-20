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
        description="The specialty group this decision applies to. Required for ADD/DISCARD/CHALLENGE."
    )
    feedback: str = Field(
        description="Justification. In Benchmark Mode, focus on differentiating between the top options."
    )
    new_hypothesis_name: Optional[str] = Field(
        description="Name of hypothesis to add (ADD_HYPOTHESIS only)."
    )
    hypothesis_to_discard: Optional[str] = Field(
        description="Name of hypothesis to discard (DISCARD_HYPOTHESIS only)."
    )
    questions_for_human: Optional[List[str]] = Field(
        description="List of CRITICAL questions (Max 5). Required if decision is ASK_HUMAN."
    )

class CriticAgent:
    def __init__(self, llm: ChatGoogleGenerativeAI, benchmark_mode: bool = False):
        self.llm = llm
        self.benchmark_mode = benchmark_mode
        
        # --- STANDARD CLINICAL PROMPT (Safety Focused - Regular Use) ---
        standard_prompt = """You are a meticulous Diagnostic Reviewer. Your goal is clinical safety and accuracy.

        ACTIONS:
        1.  **APPROVE**: If the analysis is solid given available info.
        2.  **ASK_HUMAN**: **CRITICAL USE ONLY.** Use this ONLY if the case is ambiguous AND vital info is missing (e.g., OPQRST, Red Flags).
            - **STOP RULE:** If the patient text contains `[Clinician Responses]`, you are **FORBIDDEN** from using ASK_HUMAN. You MUST proceed with the info you have.
            - **BATCHING:** Ask ALL necessary questions (Max 5) in one go.
        3.  **ADD_HYPOTHESIS**: If a major diagnosis is missing.
        4.  **CHALLENGE_SCORE**: If a score is contradicted by facts.
        5.  **DISCARD_HYPOTHESIS**: If a diagnosis is ruled out.

        **PRIORITY:**
        1. **Check for `[Clinician Responses]`**: If present -> **DO NOT ASK**. APPROVE or REFINE only.
        2. **Ambiguity Check**: Is OPQRST missing for Chief Complaint? -> ASK_HUMAN.
        3. **Safety Check**: Missing red flags? -> ASK_HUMAN.
        """
        
        # --- BENCHMARK / EXAM PROMPT (Differentiation Focused - Eval Mode) ---
        benchmark_prompt = """You are a Medical Board Exam Tutor taking the USMLE.
        
        **CRITICAL RULES FOR BENCHMARK MODE:**
        1. **NO HUMAN INTERACTION:** You are taking a test. You cannot ask the patient questions. **NEVER select ASK_HUMAN.**
        2. **FOCUS ON OPTIONS:** The patient summary contains a list of "Candidate Diagnoses". Your job is to ensure the correct one is ranked #1.
        
        **YOUR STRATEGY (COUNTERFACTUALS):**
        - Look at the top 2-3 hypotheses.
        - Ask: "What specific evidence rules IN the top choice and rules OUT the runner-up?"
        - If the current ranking seems based on weak evidence, use **CHALLENGE_SCORE** to force the Risk Assessor to re-evaluate specific symptoms.
        - If the "Correct Answer" (based on your knowledge) is missing or ranked low, use **ADD_HYPOTHESIS** or **CHALLENGE_SCORE** to boost it.
        
        **DECISION LOGIC:**
        - If the analysis effectively distinguishes the likely answer from distractors -> **APPROVE**.
        - If the analysis misses a key distinction (e.g., "Pain is relieved by rest" favors Angina over MI) -> **CHALLENGE_SCORE**.
        - **ABSOLUTELY NO 'ASK_HUMAN'.**
        """
        
        self.system_prompt = benchmark_prompt if benchmark_mode else standard_prompt
        
        self.parser = PydanticOutputParser(pydantic_object=CriticDecision)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """
            **Patient Scenario:**
            {patient_summary}

            **Previous Feedback:**
            {previous_feedback}

            **Current Analysis:**
            {combined_analysis}

            {dynamic_instruction}

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
        if not critic_history: return "None - This is the first iteration."
        formatted = ""
        for i, feedback in enumerate(critic_history, 1):
            formatted += f"\nIt {i}: {feedback.get('decision')} - {feedback.get('feedback')}\n"
        return formatted

    def run(self, patient_summary: str, specialty_groups: Dict, critic_history: List[Dict] = None) -> Dict:
        print("-> Critic Agent is reviewing...")
        
        # --- DYNAMIC INSTRUCTION INJECTION ---
        dynamic_instruction = ""
        
        if self.benchmark_mode:
            if "Candidate Diagnoses" in patient_summary:
                dynamic_instruction = "\n\n**REMINDER:** This is an exam question with OPTIONS. Use counterfactuals to pick the BEST option. DO NOT ASK HUMAN."
        else:
            # Regular Mode: Check for existing human responses to prevent loops
            has_human_response = "[Clinician Responses]" in patient_summary or "[Additional Information" in patient_summary
            if has_human_response:
                print("   🛡️ Human input detected. Locking 'ASK_HUMAN' option.")
                dynamic_instruction = "\n\n**SYSTEM NOTICE:** Clinician has ALREADY answered. You are **STRICTLY FORBIDDEN** from selecting 'ASK_HUMAN'. Proceed with available info."

        combined_analysis = self._format_analysis_for_prompt(specialty_groups)
        previous_feedback = self._format_previous_feedback(critic_history or [])
        
        # --- INVOKE LLM ---
        result = self.chain.invoke({
            "patient_summary": patient_summary,
            "combined_analysis": combined_analysis,
            "previous_feedback": previous_feedback,
            "dynamic_instruction": dynamic_instruction,
            "format_instructions": self.parser.get_format_instructions(),
        })
        
        # --- FAILSAFE: HARD OVERRIDE ---
        # If LLM hallucinates and tries to ASK_HUMAN when it shouldn't, force fix it.
        if result.decision == "ASK_HUMAN":
            if self.benchmark_mode:
                print("   🚨 Critic tried to ASK_HUMAN in Benchmark Mode. Overriding to APPROVE.")
                result.decision = "APPROVE"
                result.feedback = "Proceeding with best available evidence (Benchmark Mode Override)."
                result.questions_for_human = []
            elif "[Clinician Responses]" in patient_summary:
                print("   🚨 Critic tried to loop (Ask Human again). Overriding to APPROVE.")
                result.decision = "APPROVE"
                result.feedback = "Proceeding with diagnosis based on provided clinician responses."
                result.questions_for_human = []
            
        return result.dict()