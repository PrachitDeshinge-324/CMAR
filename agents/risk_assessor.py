# agents/risk_assessor.py
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict

class RiskAssessment(BaseModel):
    hypothesis: str = Field(description="The option being assessed.")
    severity: int = Field(description="For Exams: Set to 0. For Patients: Severity (1-10).")
    likelihood: int = Field(description="For Exams: Score (1-10) representing how likely this is the CORRECT ANSWER. For Patients: Probability.")
    justification: str = Field(description="Reasoning based on evidence.")

class RiskAssessmentList(BaseModel):
    assessments: List[RiskAssessment]

class RiskAssessorAgent:
    def __init__(self, llm: ChatGoogleGenerativeAI, benchmark_mode: bool = False):
        self.llm = llm
        self.benchmark_mode = benchmark_mode
        
        if self.benchmark_mode:
            self.system_prompt = """You are an expert Medical Board Exam Grader. 
            You are evaluating a list of potential answers (options) for a USMLE question.

            **INSTRUCTIONS:**
            1. **Ignore 'Severity':** Since options can be treatments or mechanisms, 'Severity' is irrelevant. **Set severity to 0 for all.**
            2. **Assess Correctness (Likelihood):** Rate from 1-10 how likely this option is the **CORRECT ANSWER** to the question.
               - 10 = Perfect match, supported by evidence, standard of care.
               - 1 = Clearly wrong, contradicted by facts, or a distractor.
            3. **Justification:** Explain why this option is correct or incorrect based on the patient scenario.
            """
        else:
            self.system_prompt = """You are an expert clinical risk analyst.
            Assess Severity (1-10) and Likelihood (1-10) for each diagnosis based on the patient scenario.
            """
            
        self.parser = PydanticOutputParser(pydantic_object=RiskAssessmentList)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """
            **Context/Challenge:**
            {critic_challenge}

            **Patient Scenario:**
            {patient_summary}

            **Options/Hypotheses & Evidence:**
            {hypotheses_with_evidence}

            {format_instructions}
            """)
        ])
        self.chain = self.prompt | self.llm | self.parser

    def _format_hypotheses_for_prompt(self, hypotheses: List[Dict]) -> str:
        formatted = ""
        for i, hypo in enumerate(hypotheses):
            formatted += f"\nOption {i+1}: {hypo['hypothesis']}\nEvidence: {hypo.get('evidence', 'None')}\n"
        return formatted

    def assess_risk_for_specialty(self, patient_summary: str, hypotheses: List[Dict], critic_challenge: str = "None") -> List[Dict]:
        hypotheses_with_evidence = self._format_hypotheses_for_prompt(hypotheses)
        
        try:
            result = self.chain.invoke({
                "patient_summary": patient_summary,
                "hypotheses_with_evidence": hypotheses_with_evidence,
                "format_instructions": self.parser.get_format_instructions(),
                "critic_challenge": critic_challenge,
            })

            assessment_map = {a.hypothesis: a for a in result.assessments}
            updated_hypotheses = []
            for hypo in hypotheses:
                assessment = assessment_map.get(hypo['hypothesis'])
                if assessment:
                    hypo['severity'] = assessment.severity
                    hypo['likelihood'] = assessment.likelihood
                    hypo['risk_justification'] = assessment.justification
                else:
                    hypo['severity'] = 0
                    hypo['likelihood'] = 5
                    hypo['risk_justification'] = "Assessment failed."
                updated_hypotheses.append(hypo)
            return updated_hypotheses
            
        except Exception as e:
            print(f"    ⚠️ Risk assessment error: {e}")
            return hypotheses # Return unchanged on error