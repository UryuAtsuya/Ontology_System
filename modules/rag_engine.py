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
        Performs a hybrid search with Keyword Boosting:
        1. Access Vector Store
        2. Compute Cosine Similarity
        3. Boost score if query keywords appear in Building Name
        4. Return Top K
        """
        results: List[Dict[str, Any]] = []
        
        # Ensure Vector Store is ready
        if self.llm_client and not self.vector_store and self.om.ontology:
            print("Initializing Vector Store (generating embeddings)...")
            for ind in self.om.ontology.individuals():
                classes = [c.name for c in ind.is_a if hasattr(c, 'name')]
                # Rich text for embedding
                props_text = ", ".join([f"{p.name}: {v}" for p in ind.get_properties() for v in p[ind]])
                text = f"Building Name: {ind.name}. Type: {', '.join(classes)}. Details: {props_text}"
                
                self.ids_to_text[ind.iri] = text
                self.vector_store[ind.iri] = self.llm_client.get_embedding(text)

        if not self.vector_store or not self.llm_client:
            return []

        query_embedding = self.llm_client.get_embedding(query)
        scored_candidates = []

        # Keywords for boosting
        keywords = query.lower().split()

        for uri, embedding in self.vector_store.items():
            base_score = self._compute_cosine_similarity(query_embedding, embedding)
            final_score = base_score
            
            # Retrieve entity info to check name
            ind = self.om.ontology.search_one(iri=uri)
            if ind:
                # KEYWORD BOOSTING
                name_lower = ind.name.lower()
                if any(k in name_lower for k in keywords if len(k) > 1):
                    final_score *= 1.2 # Boost by 20%
                
                scored_candidates.append((final_score, ind))

        # Sort by Final Score and take Top 5
        top_k = sorted(scored_candidates, key=lambda x: x[0], reverse=True)[:5]
        
        for score, ind in top_k:
            if score > 0.4: # Threshold
                results.append({
                    "uri": ind.iri,
                    "name": ind.name,
                    "type": [c.name for c in ind.is_a if hasattr(c, 'name')],
                    "score": score,
                    "details": self.ids_to_text.get(ind.iri, "")
                })
        
        return results

    def format_context_for_llm(self, retrieved_items: List[Dict[str, Any]]) -> str:
        """Formats retrieved items into a context string for the LLM using citation format."""
        if not retrieved_items:
            return "No specific reference information found in ontology."

        items_str = ""
        for item in retrieved_items:
            # Using @entity{URI, label, type} format as requested
            items_str += f"@entity{{URI={item['uri']}, label='{item['name']}', type='{','.join(item['type'])}'}}\n"
            items_str += f"Description: {item.get('details', '')}\n"
        
        return CONTEXT_TEMPLATE.format(context_items=items_str)
