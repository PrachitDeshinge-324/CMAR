# agents/risk_assessor.py
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict, Any

# Define the structured output for a single risk assessment
class RiskAssessment(BaseModel):
    hypothesis: str = Field(description="The name of the medical hypothesis being assessed.")
    severity: int = Field(description="A score from 1 (mild) to 10 (life-threatening) indicating the severity of the condition if it is the correct diagnosis.")
    likelihood: int = Field(description="A score from 1 (very unlikely) to 10 (very likely) indicating how well the evidence supports this diagnosis for this specific patient.")
    justification: str = Field(description="A brief, context-based justification for the assigned severity and likelihood scores, referencing the patient scenario and evidence.")

# Define the top-level list that the LLM will return
class RiskAssessmentList(BaseModel):
    assessments: List[RiskAssessment]

class RiskAssessorAgent:
    """
    An agent that performs contextual risk assessment on medical hypotheses.
    It does NOT use statistical data, but reasons based on the provided
    patient scenario and evidence.
    """
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
        self.system_prompt = """You are an expert clinical diagnostician. Your task is to assess the probability that a specific hypothesis is the CORRECT diagnosis for the patient.

        **CRITICAL INSTRUCTIONS:**
        1.  **Integrate Medical Knowledge:** You MUST combine the provided patient evidence with your internal knowledge of epidemiology, prevalence, and clinical presentation.
        2.  **Assess Likelihood (1-10):** How well does this diagnosis explain the *entire* clinical picture? 
            - 10 = "Textbook presentation", highly probable, explains all key symptoms.
            - 1 = Unlikely, contradicted by key findings, or epidemiologically rare without specific risk factors.
        3.  **Assess Severity (1-10):** (Standard medical severity).
        4.  **Differentiation:** If multiple hypotheses are similar, penalize the ones that miss a key "discriminating feature" present in the patient summary.
        """
        
        self.parser = PydanticOutputParser(pydantic_object=RiskAssessmentList)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """
            **Critic's Challenge (You MUST consider this if present else assess the following hypotheses for the patient.):**
            {critic_challenge}

            **Patient Scenario:**
            {patient_summary}

            **Hypotheses and Evidence:**
            {hypotheses_with_evidence}

            Based on the new challenge, re-assess the severity and likelihood scores.
            {format_instructions}
            """)
        ])
        self.chain = self.prompt | self.llm | self.parser

    def _format_hypotheses_for_prompt(self, hypotheses: List[Dict]) -> str:
        """Formats the list of hypotheses and their evidence into a single string."""
        formatted_string = ""
        for i, hypo in enumerate(hypotheses):
            formatted_string += f"\n--- Hypothesis {i+1} ---\n"
            formatted_string += f"Name: {hypo['hypothesis']}\n"
            formatted_string += f"Evidence: {hypo['evidence']}\n"
        return formatted_string

    def assess_risk_for_specialty(
        self, patient_summary: str, hypotheses: List[Dict], critic_challenge: str = "None. This is the initial assessment."
    ) -> List[Dict]:
        """
        Takes a list of hypotheses for one specialty and enriches them with risk scores.
        Makes a single, efficient LLM call for all hypotheses in the specialty.
        """
        print(f"    - Assessing risk for {len(hypotheses)} hypotheses...")
        
        # Format the input for the LLM
        hypotheses_with_evidence = self._format_hypotheses_for_prompt(hypotheses)
        
        try:
            # Invoke the LLM chain
            result = self.chain.invoke({
                "patient_summary": patient_summary,
                "hypotheses_with_evidence": hypotheses_with_evidence,
                "format_instructions": self.parser.get_format_instructions(),
                "critic_challenge": critic_challenge, # <-- Pass to prompt
            })

            # Integrate the new scores back into the original hypothesis objects
            assessment_map = {assessment.hypothesis: assessment for assessment in result.assessments}
            
            updated_hypotheses = []
            for hypo in hypotheses:
                assessment = assessment_map.get(hypo['hypothesis'])
                if assessment:
                    hypo['severity'] = assessment.severity
                    hypo['likelihood'] = assessment.likelihood
                    hypo['risk_justification'] = assessment.justification
                else:
                    # If no assessment found, provide default values
                    hypo['severity'] = 5
                    hypo['likelihood'] = 5
                    hypo['risk_justification'] = "Assessment not available - using default scores."
                updated_hypotheses.append(hypo)
            
            return updated_hypotheses
            
        except Exception as e:
            print(f"    ⚠️ Error during risk assessment: {str(e)[:200]}")
            # Return hypotheses with default scores
            updated_hypotheses = []
            for hypo in hypotheses:
                hypo['severity'] = 5
                hypo['likelihood'] = 5
                hypo['risk_justification'] = f"Risk assessment failed - using default scores. Error: {str(e)[:100]}"
                updated_hypotheses.append(hypo)
            return updated_hypotheses