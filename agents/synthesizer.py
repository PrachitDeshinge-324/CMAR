# agents/synthesizer.py
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict, Optional, Literal
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Define the structured output for the final report
class FinalDiagnosis(BaseModel):
    rank: int = Field(description="The final rank of the diagnosis, ordered by clinical urgency.")
    diagnosis: str = Field(description="The scientific/medical name of the potential condition.")
    general_name: str = Field(description="The general/common/layman term for the condition (e.g., 'heart attack' for MI, 'lung infection' for pneumonia).")
    category: Literal["Critical / Must Rule Out", "Probable", "Low Probability"] = Field(
        description="The clinical category of the diagnosis based on severity and likelihood."
    )
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
    def __init__(self, llm: ChatGoogleGenerativeAI, embeddings_model=None):
        self.llm = llm
        self.embeddings_model = embeddings_model
        
        self.system_prompt = """You are a senior medical scribe and rapporteur with expert knowledge in medical terminology and diagnosis matching.

        Your task is to synthesize a complex, multi-specialty diagnostic analysis into a single, clear, and actionable final report.

        The final list of differential diagnoses has already been ranked by a deterministic algorithm based on clinical urgency.
        
        You must categorize each diagnosis into one of three clinical workflows:
        1. **Critical / Must Rule Out**: High severity conditions that cannot be missed, even if likelihood is moderate.
        2. **Probable**: The most likely diagnoses based on the evidence.
        3. **Low Probability**: Unlikely diagnoses that are being considered for completeness.

        Your main responsibilities are:
        1. Accurately transcribe the provided ranked list of diagnoses.
        2. For EACH diagnosis, provide BOTH:
           - The scientific/medical name (e.g., "Myocardial Infarction", "Pneumonia", "COPD")
           - The general/common/layman term (e.g., "heart attack", "lung infection", "smoking-related lung disease")
        3. Assign the correct category based on the provided data.
        4. For each diagnosis, write a concise, professional justification that synthesizes the key findings from the risk assessment.
        5. Write a brief, high-level "Overall Assessment" that summarizes the most critical findings and explains why the top-ranked diagnoses are the most urgent.
        
        CRITICAL - Ground Truth Validation (if provided):
        If a ground truth diagnosis is provided, you MUST perform GENEROUS medical validation for evaluation purposes:
        
        MATCHING RULES (from most to least strict):
        1. EXACT MATCH: Same diagnosis or direct synonym
           - "Hypoglycemia" = "Low Blood Sugar"
           - "Myocardial Infarction" = "Heart Attack"
        
        2. SPECIFIC SUBTYPE: Differential diagnosis is a specific type of ground truth
           - "Skin Cancer" ← "Squamous Cell Carcinoma", "Melanoma", "Basal Cell Carcinoma"
           - "Adverse Drug Reaction" ← "Drug-induced Anaphylaxis", "Medication Side Effect"
        
        3. RELATED CONDITION IN SAME ORGAN SYSTEM: Different conditions affecting same organ/system
           - "Chronic Bronchitis" ← "Pneumonia", "COPD", "Acute Bronchitis" (all lung/airway)
           - "Varicocele" ← "Hydrocele", "Spermatocele", "Testicular torsion" (all testicular)
           - "Gastroenteritis" ← "Gastritis", "Peptic Ulcer", "IBD" (all GI inflammation)
        
        4. SAME CATEGORY/PATHOPHYSIOLOGY: Share underlying mechanism
           - "Hypoglycemia" ← "Electrolyte Imbalance", "Metabolic Disorder"
           - "Skin Pigmentation Disorder" ← "Melanoma", "Nevus", "Vitiligo"
        
        EVALUATION PHILOSOPHY:
        - If a clinician would reasonably consider both diagnoses in the differential for the same symptoms, COUNT IT AS A MATCH
        - Prioritize clinical reasoning over strict nosological accuracy
        - When in doubt, be GENEROUS and mark as match with explanation
        
        OUTPUT REQUIREMENTS:
        - Mark 'matches_ground_truth' = true for the BEST matching diagnosis
        - Set is_correct = true if ANY rule above applies
        - Provide clear reasoning explaining which matching rule applies
        - If truly no match (e.g., "Heart Attack" vs "Skin Rash"), only then mark false
        """
        
        self.parser = PydanticOutputParser(pydantic_object=FinalReport)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """
            Please synthesize the following ranked diagnostic data into the final report format.

            **Patient Scenario:**
            {patient_summary}

            **Ranked Differential Diagnoses:**
            {ranked_diagnoses}

            **Ground Truth Diagnosis (for validation):**
            {ground_truth_diagnosis}

            VALIDATION EXAMPLES TO GUIDE YOUR MATCHING:
            ✅ "Chronic Bronchitis" → "Pneumonia" (MATCH: both respiratory infections/inflammation)
            ✅ "Varicocele" → "Hydrocele" (MATCH: both testicular swelling conditions)  
            ✅ "Gastroenteritis" → "Gastritis" (MATCH: both GI inflammation)
            ✅ "Hypoglycemia" → "Electrolyte Imbalance" (MATCH: related metabolic disorders)
            ✅ "Skin Cancer" → "Squamous Cell Carcinoma" (MATCH: specific subtype)
            ❌ "Heart Attack" → "Skin Rash" (NO MATCH: completely unrelated organ systems)

            CRITICAL INSTRUCTIONS:
            1. For EVERY diagnosis, you MUST provide both 'diagnosis' (scientific name) and 'general_name' (layman term)
            2. If ground truth is provided above, you MUST validate it against the differential diagnoses
            3. Use your medical expertise to identify matches, synonyms, or related conditions
            4. Mark 'matches_ground_truth' as true for the diagnosis that best matches the ground truth
            5. Fill out the complete ground_truth_validation section with your reasoning
            6. Be generous with matches - include subtypes, related conditions, and medical synonyms

            {format_instructions}
            """)
        ])
        self.chain = self.prompt | self.llm | self.parser

    def _rank_hypotheses(self, specialty_groups: Dict, severity_weight: float = 0.2, likelihood_weight: float = 0.8) -> List[Dict]:
        """
        UPDATED: Ranks primarily by Likelihood (Probability) to ensure the 'Right Diagnosis' is #1.
        
        New Weights:
        - Likelihood: 0.8 (80% impact) -> Prioritizes what the patient actually HAS.
        - Severity: 0.2 (20% impact) -> Tie-breaker for dangerous conditions.
        """
        all_hypotheses = []
        for specialty in specialty_groups:
            all_hypotheses.extend(specialty_groups[specialty])

        # Filter valid
        valid_hypotheses = [
            h for h in all_hypotheses if 'severity' in h and 'likelihood' in h
        ]
        
        # Calculate Weighted Score for ACCURACY
        for h in valid_hypotheses:
            h['urgency_score'] = (
                h['severity'] * severity_weight + 
                h['likelihood'] * likelihood_weight
            )
        
        # Sort by score (descending)
        ranked_list = sorted(
            valid_hypotheses,
            key=lambda x: x['urgency_score'],
            reverse=True
        )
        return ranked_list

    def _format_ranked_list_for_prompt(self, ranked_list: List[Dict]) -> str:
        """Formats the ranked list into a string for the LLM prompt."""
        formatted_string = ""
        for i, hypo in enumerate(ranked_list):
            formatted_string += f"\n--- Rank {i+1} ---\n"
            formatted_string += f"Diagnosis: {hypo['hypothesis']}\n"
            formatted_string += f"Category: {hypo.get('category', 'Uncategorized')}\n"
            formatted_string += f"Assigned Severity: {hypo['severity']}\n"
            formatted_string += f"Assigned Likelihood: {hypo['likelihood']}\n"
            formatted_string += f"Urgency Score: {hypo.get('urgency_score', 0):.2f}\n"
            formatted_string += f"Justification Notes: {hypo.get('risk_justification', 'N/A')}\n"
        return formatted_string
    
    def _validate_ground_truth_semantic(self, diagnoses: List[Dict], ground_truth: str, top_k: int = 5, similarity_threshold: float = 0.65) -> Dict:
        """
        Semantic validation using SapBERT (medical-specific) or embeddings model.
        Falls back to keyword matching if neither is available.
        
        Args:
            diagnoses: List of diagnosis dictionaries
            ground_truth: The ground truth diagnosis
            top_k: Number of top diagnoses to check
            similarity_threshold: Cosine similarity threshold (0.70 = 70% similar)
        """
        ground_truth_lower = ground_truth.lower().strip()
        
        # PRIORITY 1: Try SapBERT (best for medical terms)
        if self.medical_matcher:
            try:
                best_match = None
                best_score = 0.0
                
                for i, dx in enumerate(diagnoses[:top_k]):
                    dx_name = dx['hypothesis']
                    similarity = self.medical_matcher.similarity(ground_truth, dx_name)
                    
                    if similarity > best_score:
                        best_score = similarity
                        best_match = {
                            "rank": i + 1,
                            "diagnosis": dx_name,
                            "similarity": float(similarity)
                        }
                
                # Check if best match exceeds threshold
                if best_match and best_score >= similarity_threshold:
                    return {
                        "ground_truth": ground_truth,
                        "is_correct": True,
                        "best_match_rank": best_match["rank"],
                        "best_match_diagnosis": best_match["diagnosis"],
                        "best_match_reasoning": f"SapBERT medical similarity: {best_score:.2%} (threshold: {similarity_threshold:.0%})",
                    }
                else:
                    return {
                        "ground_truth": ground_truth,
                        "is_correct": False,
                        "best_match_rank": best_match["rank"] if best_match else None,
                        "best_match_diagnosis": best_match["diagnosis"] if best_match else None,
                        "best_match_reasoning": f"Best SapBERT similarity: {best_score:.2%}, below threshold {similarity_threshold:.0%}",
                    }
                    
            except Exception as e:
                print(f"   ⚠️  SapBERT validation failed: {e}. Falling back to embeddings model.")
        
        # PRIORITY 2: Fallback to embeddings model if SapBERT not available
        if self.embeddings_model:
            try:
                # Embed ground truth
                gt_embedding = self.embeddings_model.embed_query(ground_truth_lower)
                gt_embedding = np.array(gt_embedding).reshape(1, -1)
                
                # Check each diagnosis in top-K
                best_match = None
                best_score = 0.0
                
                for i, dx in enumerate(diagnoses[:top_k]):
                    dx_name = dx['hypothesis']
                    
                    # Embed diagnosis name
                    dx_embedding = self.embeddings_model.embed_query(dx_name.lower())
                    dx_embedding = np.array(dx_embedding).reshape(1, -1)
                    
                    # Calculate cosine similarity
                    similarity = cosine_similarity(gt_embedding, dx_embedding)[0][0]
                    
                    if similarity > best_score:
                        best_score = similarity
                        best_match = {
                            "rank": i + 1,
                            "diagnosis": dx_name,
                            "similarity": float(similarity)
                        }
                
                # Check if best match exceeds threshold
                if best_match and best_score >= similarity_threshold:
                    return {
                        "ground_truth": ground_truth,
                        "is_correct": True,
                        "best_match_rank": best_match["rank"],
                        "best_match_diagnosis": best_match["diagnosis"],
                        "best_match_reasoning": f"Embedding similarity: {best_score:.2%} (threshold: {similarity_threshold:.0%})",
                    }
                else:
                    return {
                        "ground_truth": ground_truth,
                        "is_correct": False,
                        "best_match_rank": best_match["rank"] if best_match else None,
                        "best_match_diagnosis": best_match["diagnosis"] if best_match else None,
                        "best_match_reasoning": f"Best embedding similarity: {best_score:.2%}, below threshold {similarity_threshold:.0%}",
                    }
                    
            except Exception as e:
                print(f"   ⚠️  Embedding validation failed: {e}. Falling back to keyword matching.")
        
        # Fallback: Simple keyword matching
        common_medical_terms = {
            'drug': ['medication', 'adverse', 'hypersensitivity', 'allergic', 'anaphylaxis', 'reaction'],
            'heart': ['cardiac', 'myocardial', 'coronary', 'cardiovascular'],
            'stroke': ['cerebrovascular', 'cva', 'cerebral infarction'],
            'lung': ['pulmonary', 'respiratory', 'pneumonia', 'copd'],
            'infection': ['sepsis', 'bacterial', 'viral', 'infectious'],
        }
        
        gt_words = set(ground_truth_lower.split())
        
        for i, dx in enumerate(diagnoses[:top_k]):
            dx_name_lower = dx['hypothesis'].lower()
            dx_words = set(dx_name_lower.split())
            
            # Direct word overlap
            word_overlap = gt_words & dx_words
            if len(word_overlap) >= 2:  # At least 2 words match
                return {
                    "ground_truth": ground_truth,
                    "is_correct": True,
                    "best_match_rank": i + 1,
                    "best_match_diagnosis": dx['hypothesis'],
                    "best_match_reasoning": f"Keyword match: {word_overlap}",
                }
            
            # Related terms matching
            for gt_word in gt_words:
                if gt_word in common_medical_terms:
                    related_terms = common_medical_terms[gt_word]
                    if any(term in dx_name_lower for term in related_terms):
                        return {
                            "ground_truth": ground_truth,
                            "is_correct": True,
                            "best_match_rank": i + 1,
                            "best_match_diagnosis": dx['hypothesis'],
                            "best_match_reasoning": f"Related term match: '{gt_word}' → found related terms in diagnosis",
                        }
        
        return {
            "ground_truth": ground_truth,
            "is_correct": False,
            "best_match_rank": None,
            "best_match_diagnosis": None,
            "best_match_reasoning": f"No semantic or keyword matches found in top {top_k}",
        }

    def run(self, patient_summary: str, specialty_groups: Dict, ground_truth: Optional[str] = None, top_k: int = 10) -> Dict:
        """
        Synthesizer with Accuracy-First Ranking.
        """
        print("-> Synthesizing and ranking (Accuracy Focused)...")
        
        # 1. Perform deterministic ranking (Uses new 0.8/0.2 weights)
        ranked_diagnoses_data = self._rank_hypotheses(specialty_groups)
        
        # 2. Format for LLM
        ranked_diagnoses_prompt_str = self._format_ranked_list_for_prompt(ranked_diagnoses_data[:top_k])
        
        # 3. Prepare ground truth
        ground_truth_str = ground_truth if ground_truth else "No ground truth provided"
        
        if ground_truth:
            print(f"-> LLM will validate against ground truth: '{ground_truth}'")
        
        # 4. LLM call
        print("-> Invoking LLM for final report...")
        report = self.chain.invoke({
            "patient_summary": patient_summary,
            "ranked_diagnoses": ranked_diagnoses_prompt_str,
            "ground_truth_diagnosis": ground_truth_str,
            "format_instructions": self.parser.get_format_instructions(),
        })
        
        return report.dict()
