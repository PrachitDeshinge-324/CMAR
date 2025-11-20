# agents/evidence_evaluator.py
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict

class EvidenceEvaluatorAgent:
    def __init__(self, llm: ChatGoogleGenerativeAI, retriever_tool: BaseTool, web_search_tool: BaseTool, enable_batching: bool = True):
        self.llm = llm
        self.retriever = retriever_tool
        self.web_search = web_search_tool
        self.enable_batching = enable_batching

    def _format_evidence(self, source: str, content: str) -> str:
        return f"Source: {source}\nContent: {content}\n"

    def _batch_retrieve_evidence(self, patient_summary: str, hypotheses: List[Dict]) -> Dict[str, str]:
        if len(hypotheses) == 1:
            return self._retrieve_evidence_single(patient_summary, hypotheses[0])
        
        hypothesis_names = [h['hypothesis'] for h in hypotheses]
        
        combined_query = (
            f"Clinical evidence regarding the following options for this patient case:\n"
            f"Patient: {patient_summary}\n"
            f"Options: {', '.join(hypothesis_names)}"
        )
        
        try:
            rag_results = self.retriever.invoke(combined_query)
        except Exception:
            rag_results = []
        
        if not rag_results: return {}
        
        evidence_map = {h['hypothesis']: "" for h in hypotheses}
        for doc in rag_results:
            content = doc.page_content.lower()
            source = doc.metadata.get('source', 'PubMed Central')
            formatted_evidence = self._format_evidence(source, doc.page_content)
            
            matched_any = False
            for hypo_name in hypothesis_names:
                keywords = hypo_name.lower().split()
                if any(k in content for k in keywords if len(k) > 4):
                    evidence_map[hypo_name] += formatted_evidence
                    matched_any = True
            
            if not matched_any:
                 for hypo_name in hypothesis_names:
                    evidence_map[hypo_name] += formatted_evidence

        return evidence_map

    def _retrieve_evidence_single(self, patient_summary: str, hypothesis: Dict) -> Dict[str, str]:
        hypothesis_name = hypothesis['hypothesis']
        query = f"Evidence supporting '{hypothesis_name}' for patient: {patient_summary}"
        
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
        Refactored to avoid infinite recursion.
        """
        evidence_map = {}
        
        # 1. DETERMINE RETRIEVAL STRATEGY
        if self.enable_batching and len(hypotheses) > 1:
            # Use Batch Strategy
            try:
                evidence_map = self._batch_retrieve_evidence(patient_summary, hypotheses)
            except Exception as e:
                print(f"    ⚠️ Batch retrieval failed: {e}. Falling back to sequential.")
                # Fallback to sequential if batch fails
                for h in hypotheses:
                    evidence_map.update(self._retrieve_evidence_single(patient_summary, h))
        else:
            # Use Sequential Strategy (No Recursion - Calls worker directly)
            for h in hypotheses:
                evidence_map.update(self._retrieve_evidence_single(patient_summary, h))

        # 2. PROCESS RESULTS & WEB FALLBACK
        updated_hypotheses = []
        for hypo in hypotheses:
            name = hypo['hypothesis']
            evidence_str = evidence_map.get(name, "")
            
            # Web Search Fallback
            if not evidence_str:
                try:
                    web_results = self.web_search.invoke(f"Medical evidence for '{name}' in context of {patient_summary}")
                    if web_results and "No good search result" not in web_results:
                        evidence_str = self._format_evidence("DuckDuckGo", web_results)
                except Exception: pass
            
            updated_hypo = hypo.copy()
            updated_hypo['evidence'] = evidence_str if evidence_str else "No specific evidence found."
            updated_hypotheses.append(updated_hypo)
            
        return updated_hypotheses