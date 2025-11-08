# agents/evidence_evaluator.py
from langchain_core.tools import BaseTool
from typing import List, Dict

class EvidenceEvaluatorAgent:
    """
    An agent that evaluates a list of hypotheses for a given specialty.
    It uses a RAG retriever first and falls back to a web search tool if needed.
    """
    def __init__(self, retriever_tool: BaseTool, web_search_tool: BaseTool):
        self.retriever = retriever_tool
        self.web_search = web_search_tool

    def _format_evidence(self, source: str, content: str) -> str:
        """Formats the evidence into a readable string."""
        return f"Source: {source}\nContent: {content}\n"

    def evaluate_hypotheses_for_specialty(
        self, patient_summary: str, hypotheses: List[Dict]
    ) -> List[Dict]:
        """
        Takes a list of hypotheses for one specialty and enriches them with evidence.
        """
        updated_hypotheses = []
        for hypo in hypotheses:
            hypothesis_name = hypo['hypothesis']
            print(f"    - Evaluating hypothesis: '{hypothesis_name}'")
            
            # Create a detailed query for better retrieval
            query = (
                f"Evidence for {hypothesis_name} as a diagnosis for a patient presenting with: "
                f"'{patient_summary}'"
            )

            # 1. Try the RAG retriever first
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
                # 2. Fallback to web search if RAG returns nothing
                print("      -> No results from RAG. Falling back to web search.")
                web_results = self.web_search.invoke(query)
                if web_results:
                    evidence_str = self._format_evidence("DuckDuckGo Web Search", web_results)

            # Create a new dictionary with the added evidence
            updated_hypo = hypo.copy()
            updated_hypo['evidence'] = evidence_str if evidence_str else "No evidence found."
            updated_hypotheses.append(updated_hypo)
        
        return updated_hypotheses