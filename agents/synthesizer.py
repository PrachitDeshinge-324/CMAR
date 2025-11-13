# agents/synthesizer.py
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict, Optional

# Define the structured output for the final report
class FinalDiagnosis(BaseModel):
    rank: int = Field(description="The final rank of the diagnosis, ordered by clinical urgency.")
    diagnosis: str = Field(description="The scientific/medical name of the potential condition.")
    general_name: str = Field(description="The general/common/layman term for the condition (e.g., 'heart attack' for MI, 'lung infection' for pneumonia).")
    severity: int = Field(description="The assessed severity score (1-10).")
    likelihood: int = Field(description="The assessed likelihood score (1-10).")
    justification: str = Field(description="A concise summary of the justification for this ranking, combining evidence and risk assessment.")
    matches_ground_truth: Optional[bool] = Field(default=None, description="True if this diagnosis matches the ground truth (when provided).")

class GroundTruthValidation(BaseModel):
    ground_truth: str = Field(description="The ground truth diagnosis provided for validation.")
    is_correct: bool = Field(description="True if any of the top-K diagnoses match the ground truth.")
    best_match_rank: Optional[int] = Field(default=None, description="The rank of the best matching diagnosis.")
    best_match_diagnosis: Optional[str] = Field(default=None, description="The diagnosis that matched.")
    best_match_reasoning: Optional[str] = Field(default=None, description="Why this diagnosis matches the ground truth.")

class FinalReport(BaseModel):
    patient_summary: str = Field(description="A brief summary of the patient's initial presentation.")
    differential_diagnoses: List[FinalDiagnosis]
    overall_assessment: str = Field(description="A concluding summary of the diagnostic process and key findings.")
    ground_truth_validation: Optional[GroundTruthValidation] = Field(default=None, description="Validation results when ground truth is provided for evaluation.")

class SynthesizerAgent:
    """
    An agent that synthesizes the complete analysis into a final, ranked report
    ordered by clinical urgency, with disease name mapping and ground truth validation in a single LLM call.
    """
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
        self.system_prompt = """You are a senior medical scribe and rapporteur. Your task is to synthesize a complex, multi-specialty diagnostic analysis into a single, clear, and actionable final report.

        The final list of differential diagnoses has already been ranked by a deterministic algorithm based on clinical urgency (severity first, then likelihood).

        Your main responsibilities are:
        1. Accurately transcribe the provided ranked list of diagnoses.
        2. For EACH diagnosis, provide BOTH:
           - The scientific/medical name (e.g., "Myocardial Infarction", "Pneumonia", "COPD")
           - The general/common/layman term (e.g., "heart attack", "lung infection", "smoking-related lung disease")
        3. For each diagnosis, write a concise, professional justification that synthesizes the key findings from the risk assessment.
        4. Write a brief, high-level "Overall Assessment" that summarizes the most critical findings and explains why the top-ranked diagnoses are the most urgent.
        
        IMPORTANT - Ground Truth Validation:
        If a ground truth diagnosis is provided (in the context below), you must:
        - Compare each diagnosis (both scientific AND general names) against the ground truth
        - Mark 'matches_ground_truth' as true for any diagnosis that matches or is closely related to the ground truth
        - Consider matches broadly: "heart attack" matches "Myocardial Infarction", "smoking addiction" matches "COPD", etc.
        - Fill out the ground_truth_validation section with the best match found
        - If no match is found in the differential diagnoses, set is_correct to false
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

            {ground_truth_instruction}

            CRITICAL INSTRUCTIONS:
            1. For EVERY diagnosis, you MUST provide both 'diagnosis' (scientific name) and 'general_name' (layman term)
            2. If ground truth is provided, carefully check if any diagnosis matches it (considering both scientific and general names)
            3. Mark 'matches_ground_truth' appropriately for each diagnosis
            4. Fill out the complete ground_truth_validation section

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

    def run(self, patient_summary: str, specialty_groups: Dict, ground_truth: Optional[str] = None, top_k: int = 5) -> Dict:
        """
        Runs the synthesizer agent to produce the final report with integrated ground truth validation.
        
        Args:
            patient_summary: Summary of the patient's symptoms
            specialty_groups: Dictionary of specialty groups with hypotheses
            ground_truth: Optional ground truth diagnosis for evaluation
            top_k: Number of top diagnoses to consider for validation
            
        Returns:
            Dictionary containing the final report with validation results (if ground_truth provided)
        """
        print("-> Synthesizing and ranking the final report...")
        
        # 1. Perform deterministic ranking in Python
        ranked_diagnoses_data = self._rank_hypotheses(specialty_groups)
        
        # 2. Format the ranked list for the LLM to summarize
        ranked_diagnoses_prompt_str = self._format_ranked_list_for_prompt(ranked_diagnoses_data)
        
        # 3. Prepare ground truth instruction
        if ground_truth:
            ground_truth_instruction = f"""
**GROUND TRUTH FOR VALIDATION:**
The correct diagnosis is: "{ground_truth}"

You MUST:
1. Check if ANY of the top {top_k} diagnoses match this ground truth (consider both scientific and general names)
2. For each diagnosis, set 'matches_ground_truth' to true if it matches or is closely related
3. Fill out the 'ground_truth_validation' section completely with the best match

Examples of matches:
- "heart attack" matches "Myocardial Infarction" (MI)
- "smoking addiction" or "tobacco addiction" matches "COPD" or "Chronic Obstructive Pulmonary Disease"
- "lung infection" matches "Pneumonia"
- "stroke" matches "Cerebrovascular Accident" (CVA)

Be generous - if there's a clear clinical relationship or the diagnosis is caused by/related to the ground truth, consider it a match.
"""
            print(f"-> Validating against ground truth: '{ground_truth}'")
        else:
            ground_truth_instruction = "No ground truth provided for this case."
        
        # 4. Invoke the LLM to generate the report WITH validation in one call
        report = self.chain.invoke({
            "patient_summary": patient_summary,
            "ranked_diagnoses": ranked_diagnoses_prompt_str,
            "ground_truth_instruction": ground_truth_instruction,
            "format_instructions": self.parser.get_format_instructions(),
        })
        
        report_dict = report.dict()
        
        # 5. Print validation results if available
        if ground_truth and report_dict.get('ground_truth_validation'):
            validation = report_dict['ground_truth_validation']
            if validation['is_correct']:
                print(f"   ✅ MATCH FOUND at rank {validation['best_match_rank']}: {validation['best_match_diagnosis']}")
                print(f"      Reasoning: {validation['best_match_reasoning']}")
            else:
                print(f"   ❌ NO MATCH in top {top_k}")
        
        return report_dict
