# agents/evidence_evaluator.py
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict
import re

class EvidenceEvaluatorAgent:
    """
    An agent that evaluates a list of hypotheses for a given specialty.
    Optimized with batched retrieval to reduce API calls.
    """
    # Updated __init__ to accept llm
    def __init__(self, llm: ChatGoogleGenerativeAI, retriever_tool: BaseTool, web_search_tool: BaseTool, enable_batching: bool = True):
        self.llm = llm
        self.retriever = retriever_tool
        self.web_search = web_search_tool
        self.enable_batching = enable_batching

    def _format_evidence(self, source: str, content: str) -> str:
        return f"Source: {source}\nContent: {content}\n"

    def _batch_retrieve_evidence(self, patient_summary: str, hypotheses: List[Dict]) -> Dict[str, str]:
        """
        Optimized: Retrieve evidence for ALL hypotheses in a single batched query.
        """
        if len(hypotheses) == 1:
            return self._retrieve_evidence_single(patient_summary, hypotheses[0])
        
        hypothesis_names = [h['hypothesis'] for h in hypotheses]
        combined_query = (
            f"Medical evidence for differential diagnoses in a patient with: {patient_summary}\n"
            f"Consider these conditions: {', '.join(hypothesis_names)}"
        )
        
        print(f"    -> Batched retrieval for {len(hypotheses)} hypotheses...")
        
        try:
            rag_results = self.retriever.invoke(combined_query)
        except Exception as e:
            print(f"      ⚠️ Retrieval error: {e}")
            rag_results = []
        
        if not rag_results:
            return {}
        
        # Distribute evidence based on keyword matching
        evidence_map = {h['hypothesis']: "" for h in hypotheses}
        
        for doc in rag_results:
            content = doc.page_content.lower()
            source = doc.metadata.get('source', 'PubMed Central')
            formatted_evidence = self._format_evidence(source, doc.page_content)
            
            matched_any = False
            for hypo_name in hypothesis_names:
                # Simple keyword check to avoid attaching "Heart Attack" evidence to "Anxiety"
                # Split by words to avoid partial matches like "cat" in "catch"
                keywords = hypo_name.lower().split()
                if any(k in content for k in keywords if len(k) > 3):
                    evidence_map[hypo_name] += formatted_evidence
                    matched_any = True
            
            # Fallback: if it doesn't match any specific keyword but was retrieved, 
            # it might be general context. Add it to all if it's short, otherwise discard.
            if not matched_any:
                pass # Discard unassigned evidence to reduce noise

        return evidence_map

    def _retrieve_evidence_single(self, patient_summary: str, hypothesis: Dict) -> Dict[str, str]:
        hypothesis_name = hypothesis['hypothesis']
        query = f"Evidence for {hypothesis_name} diagnosis patient: {patient_summary}"
        
        try:
            rag_results = self.retriever.invoke(query)
        except Exception:
            rag_results = []
            
        evidence_str = ""
        if rag_results:
            for doc in rag_results:
                evidence_str += self._format_evidence(doc.metadata.get('source', 'PubMed'), doc.page_content)
        
        return {hypothesis_name: evidence_str}

    def evaluate_hypotheses_for_specialty(self, patient_summary: str, hypotheses: List[Dict]) -> List[Dict]:
        """
        Takes a list of hypotheses and enriches them with evidence.
        """
        if self.enable_batching and len(hypotheses) > 1:
            evidence_map = self._batch_retrieve_evidence(patient_summary, hypotheses)
            
            updated_hypotheses = []
            for hypo in hypotheses:
                name = hypo['hypothesis']
                evidence_str = evidence_map.get(name, "")
                
                # Fallback to web search
                if not evidence_str:
                    print(f"    - No batch evidence for '{name}', checking web...")
                    try:
                        web_results = self.web_search.invoke(f"Evidence for {name} given {patient_summary}")
                        if web_results and "No good search result" not in web_results:
                            evidence_str = self._format_evidence("DuckDuckGo", web_results)
                    except Exception:
                        pass
                
                updated_hypo = hypo.copy()
                updated_hypo['evidence'] = evidence_str if evidence_str else "No specific evidence found."
                updated_hypotheses.append(updated_hypo)
            return updated_hypotheses
        
        else:
            # Sequential fallback
            updated_hypotheses = []
            for hypo in hypotheses:
                name = hypo['hypothesis']
                print(f"    - Evaluating: '{name}'")
                res = self._retrieve_evidence_single(patient_summary, hypo)
                evidence_str = res.get(name, "")
                
                if not evidence_str:
                    try:
                        web_results = self.web_search.invoke(f"Evidence for {name} given {patient_summary}")
                        evidence_str = self._format_evidence("DuckDuckGo", web_results)
                    except Exception:
                        pass

                updated_hypo = hypo.copy()
                updated_hypo['evidence'] = evidence_str if evidence_str else "No evidence found."
                updated_hypotheses.append(updated_hypo)
            return updated_hypotheses