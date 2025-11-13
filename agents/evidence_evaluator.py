# agents/evidence_evaluator.py
from langchain_core.tools import BaseTool
from typing import List, Dict

class EvidenceEvaluatorAgent:
    """
    An agent that evaluates a list of hypotheses for a given specialty.
    Optimized with batched retrieval to reduce API calls.
    """
    def __init__(self, retriever_tool: BaseTool, web_search_tool: BaseTool, enable_batching: bool = True):
        self.retriever = retriever_tool
        self.web_search = web_search_tool
        self.enable_batching = enable_batching

    def _format_evidence(self, source: str, content: str) -> str:
        """Formats the evidence into a readable string."""
        return f"Source: {source}\nContent: {content}\n"

    def _batch_retrieve_evidence(self, patient_summary: str, hypotheses: List[Dict]) -> Dict[str, str]:
        """
        Optimized: Retrieve evidence for ALL hypotheses in a single batched query.
        Returns a dictionary mapping hypothesis names to evidence strings.
        """
        if len(hypotheses) == 1:
            # Single hypothesis - use original method
            return self._retrieve_evidence_single(patient_summary, hypotheses[0])
        
        # Build a combined query mentioning all hypotheses
        hypothesis_names = [h['hypothesis'] for h in hypotheses]
        combined_query = (
            f"Medical evidence for differential diagnoses in a patient with: {patient_summary}\n"
            f"Consider these conditions: {', '.join(hypothesis_names)}"
        )
        
        print(f"    -> Batched retrieval for {len(hypotheses)} hypotheses...")
        
        # Single RAG call for all hypotheses
        rag_results = self.retriever.invoke(combined_query)
        
        if not rag_results:
            print(f"      -> No RAG results. Will use individual fallback if needed.")
            return {}
        
        print(f"      -> Retrieved {len(rag_results)} documents for entire batch")
        
        # Distribute evidence to hypotheses based on keyword matching
        evidence_map = {h['hypothesis']: "" for h in hypotheses}
        
        for doc in rag_results:
            content = doc.page_content.lower()
            source = doc.metadata.get('source', 'PubMed Central')
            formatted_evidence = self._format_evidence(source, doc.page_content)
            
            # Smart distribution: assign document to hypotheses mentioned in it
            matched_any = False
            for hypo_name in hypothesis_names:
                # Check if hypothesis keywords appear in the document
                hypo_keywords = hypo_name.lower().split()
                if any(keyword in content for keyword in hypo_keywords if len(keyword) > 3):
                    evidence_map[hypo_name] += formatted_evidence
                    matched_any = True
            
            # If no specific match, add to all (generic evidence)
            if not matched_any:
                for hypo_name in hypothesis_names:
                    evidence_map[hypo_name] += formatted_evidence
        
        return evidence_map

    def _retrieve_evidence_single(self, patient_summary: str, hypothesis: Dict) -> Dict[str, str]:
        """Original single-hypothesis retrieval method."""
        hypothesis_name = hypothesis['hypothesis']
        query = (
            f"Evidence for {hypothesis_name} as a diagnosis for a patient presenting with: "
            f"'{patient_summary}'"
        )
        
        rag_results = self.retriever.invoke(query)
        evidence_str = ""
        
        if rag_results:
            for doc in rag_results:
                evidence_str += self._format_evidence(
                    doc.metadata.get('source', 'PubMed'), 
                    doc.page_content
                )
        
        return {hypothesis_name: evidence_str}

    def evaluate_hypotheses_for_specialty(
        self, patient_summary: str, hypotheses: List[Dict]
    ) -> List[Dict]:
        """
        Takes a list of hypotheses for one specialty and enriches them with evidence.
        Now with intelligent batching to reduce retrieval calls.
        """
        if self.enable_batching and len(hypotheses) > 1:
            # Use optimized batch retrieval
            evidence_map = self._batch_retrieve_evidence(patient_summary, hypotheses)
            
            updated_hypotheses = []
            for hypo in hypotheses:
                hypothesis_name = hypo['hypothesis']
                evidence_str = evidence_map.get(hypothesis_name, "")
                
                # Fallback to web search if no evidence found
                if not evidence_str:
                    print(f"    - No batch evidence for '{hypothesis_name}', using web search fallback...")
                    web_results = self.web_search.invoke(
                        f"Evidence for {hypothesis_name} given {patient_summary}"
                    )
                    if web_results:
                        evidence_str = self._format_evidence("DuckDuckGo Web Search", web_results)
                
                updated_hypo = hypo.copy()
                updated_hypo['evidence'] = evidence_str if evidence_str else "No evidence found."
                updated_hypotheses.append(updated_hypo)
            
            return updated_hypotheses
        
        else:
            # Original implementation for single hypothesis or when batching disabled
            updated_hypotheses = []
            for hypo in hypotheses:
                hypothesis_name = hypo['hypothesis']
                print(f"    - Evaluating hypothesis: '{hypothesis_name}'")
                
                query = (
                    f"Evidence for {hypothesis_name} as a diagnosis for a patient presenting with: "
                    f"'{patient_summary}'"
                )

                rag_results = self.retriever.invoke(query)
                
                evidence_str = ""
                if rag_results:
                    print(f"      -> Found {len(rag_results)} snippets from RAG.")
                    for doc in rag_results:
                        evidence_str += self._format_evidence(
                            doc.metadata.get('source', 'PubMed'), 
                            doc.page_content
                        )
                else:
                    print("      -> No results from RAG. Falling back to web search.")
                    web_results = self.web_search.invoke(query)
                    if web_results:
                        evidence_str = self._format_evidence("DuckDuckGo Web Search", web_results)

                updated_hypo = hypo.copy()
                updated_hypo['evidence'] = evidence_str if evidence_str else "No evidence found."
                updated_hypotheses.append(updated_hypo)
            
            return updated_hypotheses