import json
from typing import List, Dict, Any, Optional

CONTEXT_TEMPLATE = """Reference Information:
{context_items}
"""

class HybridRetriever:
    def __init__(self, ontology_manager: Any, llm_client: Any = None):
        self.om = ontology_manager
        self.llm_client = llm_client
        self.vector_store = {} # {uri: embedding}
        self.ids_to_text = {} # {uri: text}

    def _compute_cosine_similarity(self, vec1, vec2):
        if not vec1 or not vec2: return 0.0
        dot_product = sum(a*b for a, b in zip(vec1, vec2))
        norm_a = sum(a*a for a in vec1) ** 0.5
        norm_b = sum(b*b for b in vec2) ** 0.5
        if norm_a == 0 or norm_b == 0: return 0.0
        return dot_product / (norm_a * norm_b)

    def semantic_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Performs a hybrid search:
        1. Vector Search (Real implementation using Gemini Embeddings)
        2. Ontology Search (Keyword/Graph traversal)
        3. Result Fusion
        """
        results: List[Dict[str, Any]] = []
        seen_uris = set()

        # Pre-compute embeddings for ontology individuals if not done
        # Note: In a real app, this should be done asynchronously or pre-loaded
        if self.llm_client and not self.vector_store and self.om.ontology:
            print("Initializing Vector Store (generating embeddings)...")
            for ind in self.om.ontology.individuals():
                # Create a rich textual representation
                props_text = ", ".join([f"{p.name}: {v}" for p in ind.get_properties() for v in p[ind]])
                classes = [c.name for c in ind.is_a if hasattr(c, 'name')]
                text = f"Building Name: {ind.name}. Type: {', '.join(classes)}. Details: {props_text}"
                
                self.ids_to_text[ind.iri] = text
                self.vector_store[ind.iri] = self.llm_client.get_embedding(text)

        # 1. Vector Search
        if self.llm_client and self.vector_store:
            query_embedding = self.llm_client.get_embedding(query)
            scores = []
            for uri, embedding in self.vector_store.items():
                score = self._compute_cosine_similarity(query_embedding, embedding)
                scores.append((score, uri))
            
            # Get Top 3 Semantic Matches
            top_k = sorted(scores, key=lambda x: x[0], reverse=True)[:3]
            for score, uri in top_k:
                if score > 0.4: # Threshold
                    ind = self.om.ontology.search_one(iri=uri)
                    if ind:
                        results.append({
                            "uri": ind.iri,
                            "name": ind.name,
                            "type": [c.name for c in ind.is_a if hasattr(c, 'name')],
                            "source": f"Vector (Score: {score:.2f})",
                            "details": self.ids_to_text.get(uri, "")
                        })
                        seen_uris.add(uri)

        # 2. Ontology Search (Keyword Match)
        if self.om.ontology:
            for ind in self.om.ontology.individuals():
                if ind.iri in seen_uris: continue
                
                search_text = f"{ind.name} {' '.join([str(v) for v in ind.get_properties()])}"
                if query.lower() in search_text.lower():
                    results.append({
                        "uri": ind.iri,
                        "name": ind.name,
                        "type": [c.name for c in ind.is_a if hasattr(c, 'name')],
                        "source": "Keyword"
                    })
                    seen_uris.add(ind.iri)
        
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
