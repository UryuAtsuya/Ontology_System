import json
from typing import List, Dict, Any, Optional

CONTEXT_TEMPLATE = """Reference Information:
{context_items}
"""

class HybridRetriever:
    def __init__(self, ontology_manager: Any):
        self.om = ontology_manager
        # In production, initialize VectorStore (e.g., ChromaDB) here
        # self.vector_store = ...

    def semantic_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Performs a hybrid search:
        1. Vector Search (Mock/Placeholder)
        2. Ontology Search (Keyword/Graph traversal)
        3. Result Fusion
        """
        results: List[Dict[str, Any]] = []
        
        # A. Ontology Search (Simple Keyword Match on Individuals)
        if self.om.ontology:
            for ind in self.om.ontology.individuals():
                # Simple check if query terms appear in individual name or properties
                # This is a basic heuristic; in a real system, use SPARQL or graph traversal
                search_text = f"{ind.name} {' '.join([str(v) for v in ind.get_properties()])}"
                if query.lower() in search_text.lower():
                    # Structure the entity info
                    entity_info = {
                        "uri": ind.iri,
                        "name": ind.name,
                        "type": [c.name for c in ind.is_a if hasattr(c, 'name')],
                        "source": "Ontology"
                    }
                    results.append(entity_info)
        
        # B. Document Search (Vector Search - Mock)
        # results.append({"content": "...", "source": "doc_id", "uri": "..."})
        
        return results

    def format_context_for_llm(self, retrieved_items: List[Dict[str, Any]]) -> str:
        """Formats retrieved items into a context string for the LLM."""
        if not retrieved_items:
            return "No specific reference information found in ontology."

        items_str = ""
        for item in retrieved_items:
            items_str += f"- Entity: {item['name']} (Type: {', '.join(item['type'])})\n"
            items_str += f"  URI: {item['uri']}\n"
        
        return CONTEXT_TEMPLATE.format(context_items=items_str)
