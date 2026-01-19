from __future__ import annotations

import os
import sys
import time
import math
import sqlite3
import re
from typing import List, Tuple, Dict, Any, Callable, Optional
from dataclasses import dataclass, asdict
from collections import Counter


# -------------------------- optional deps --------------------------
# rapidfuzz Levenshtein (fast); fallback to pure python
try:
    from rapidfuzz.distance import Levenshtein  # type: ignore
    def levenshtein(a: str, b: str) -> int:
        return int(Levenshtein.distance(a, b))
except Exception:
    def levenshtein(a: str, b: str) -> int:
        # O(len(a)*len(b)) DP
        la, lb = len(a), len(b)
        if la == 0: return lb
        if lb == 0: return la
        dp = list(range(lb+1))
        for i in range(1, la+1):
            prev = dp[0]
            dp[0] = i
            ai = a[i-1]
            for j in range(1, lb+1):
                temp = dp[j]
                cost = 0 if ai == b[j-1] else 1
                dp[j] = min(dp[j] + 1,     # deletion
                            dp[j-1] + 1,   # insertion
                            prev + cost)   # substitution
                prev = temp
        return dp[-1]

# bert_score (Lazy import in bertscore_f1)
_BERT_OK = False

# transformers (NLI)
_TRANSFORMERS_OK = True
try:
    from transformers import pipeline
except Exception:
    _TRANSFORMERS_OK = False

# -------------------------- utils --------------------------

def normalize_text(s: str) -> str:
    s = s.strip()
    s = s.replace("\u3000", " ")
    s = " ".join(s.split())
    return s

def tokenize_words(s: str) -> List[str]:
    # Use character-level tokenization for Japanese robustness
    return list(normalize_text(s).lower())

def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def lcs(a: List[str], b: List[str]) -> int:
    la, lb = len(a), len(b)
    dp = [0] * (lb + 1)
    for i in range(1, la + 1):
        prev = 0
        for j in range(1, lb + 1):
            temp = dp[j]
            if a[i-1] == b[j-1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j-1])
            prev = temp
    return dp[-1]

def chrF(ref: str, cand: str, n: int = 6, beta: float = 2.0) -> float:
    def char_ngrams(s: str, k: int) -> Counter:
        s = normalize_text(s)
        return Counter([s[i:i+k] for i in range(len(s)-k+1)]) if len(s) >= k else Counter()
    precisions, recalls = [], []
    for k in range(1, n+1):
        Rk = char_ngrams(ref, k)
        Ck = char_ngrams(cand, k)
        if not Rk and not Ck:
            continue
        overlap = sum((Rk & Ck).values())
        p = overlap / max(sum(Ck.values()), 1)
        r = overlap / max(sum(Rk.values()), 1)
        precisions.append(p); recalls.append(r)
    if not precisions:
        return 0.0
    p = sum(precisions) / len(precisions)
    r = sum(recalls) / len(recalls)
    if p == 0 and r == 0:
        return 0.0
    beta2 = beta * beta
    return (1 + beta2) * p * r / (beta2 * p + r)

def safe_split_sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=[。．\.!?])\s+|\n+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def simple_tokenize(text: str) -> List[str]:
    t = re.sub(r'[、，。．・,.\(\)\[\]{}「」『』:：;；!?！？]', ' ', text)
    t = re.sub(r'\s+', ' ', t)
    return t.strip().split()

def ngram_set(tokens: List[str], n: int = 2) -> set:
    if n <= 1:
        return set(tokens)
    return set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

# -------------------------- metrics --------------------------

def edit_sim(reference: str, candidate: str) -> float:
    if not reference or not candidate:
        return 0.0
    d = levenshtein(reference, candidate)
    m = max(len(reference), len(candidate))
    return 1.0 - d / m if m else 1.0

def token_f1(reference: str, candidate: str) -> float:
    ref_tokens = tokenize_words(reference)
    cand_tokens = tokenize_words(candidate)
    if not ref_tokens and not cand_tokens:
        return 1.0
    ref_counts = Counter(ref_tokens)
    cand_counts = Counter(cand_tokens)
    overlap = sum((ref_counts & cand_counts).values())
    precision = overlap / max(len(cand_tokens), 1)
    recall = overlap / max(len(ref_tokens), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def jaccard(reference: str, candidate: str) -> float:
    a, b = set(tokenize_words(reference)), set(tokenize_words(candidate))
    if not a and not b: return 1.0
    return len(a & b) / len(a | b) if (a | b) else 0.0

def rouge_l_f1(reference: str, candidate: str) -> float:
    ref_t = tokenize_words(reference); cand_t = tokenize_words(candidate)
    if not ref_t and not cand_t:
        return 1.0
    L = lcs(ref_t, cand_t)
    prec = L / max(len(cand_t), 1)
    rec = L / max(len(ref_t), 1)
    if prec + rec == 0: return 0.0
    return 2 * prec * rec / (prec + rec)

# ...

def bertscore_f1(reference: str, candidate: str) -> float:
    if not _BERT_OK:
        return 0.0
    try:
        # Default with baseline rescale
        P, R, F1 = bert_score.score([candidate], [reference], lang="ja", rescale_with_baseline=True)
        return float(F1[0])
    except Exception as e:
        # Debug print
        print(f"[BERTScore Error] {e}") 
        try:
            # Fallback without rescale
            P, R, F1 = bert_score.score([candidate], [reference], lang="ja", rescale_with_baseline=False)
            return float(F1[0])
        except Exception as e2:
            print(f"[BERTScore Failed] {e2}")
            return 0.0

# -------------------------- Thesis Evaluation (T1, T2, T3) --------------------------
from modules.ontology_manager import OntologyManager
from modules.rag_engine import HybridRetriever
from modules.llm_client import LLMClient

class ThesisEvaluator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        print("Initializing System components...")
        self.mgr = OntologyManager() # Loads temp_Ontology.owl by default
        self.llm = LLMClient(api_key)
        self.retriever = HybridRetriever(self.mgr, self.llm)
        
    def run_thesis_evaluation(self, dataset_path: str = "evaluation_dataset.json", output_path: str = "thesis_eval_results.json"):
        import json
        import time 
        
        debug_log_path = "debug_eval_log.txt"
        with open(debug_log_path, "w", encoding="utf-8") as dlog:
            dlog.write("--- Debug Log for Thesis Evaluation ---\n") 
        
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
            
        results = []
        print(f"Starting Comparative Evaluation (Zero-shot vs Few-shot/RAG) on {len(dataset)} items...")
        
        for i, data in enumerate(dataset):
            print(f"Testing [{data['id']}]: {data['query']}")
            
            # Rate Limiting: Minimal sleep for OpenAI (optional, just to be safe)
            if i > 0:
                time.sleep(1)

            # --- 1. Zero-shot Generation (Baseline) ---
            try:
                # No context provided
                zs_response = self.llm.generate_response(data['query'], context_str="")
            except Exception as e:
                print(f"Zero-shot Error: {e}")
                zs_response = ""

            # --- 2. Few-shot/RAG Generation (Proposed) ---
            try:
                context_items = self.retriever.semantic_search(data['query'])
                context_str = self.retriever.format_context_for_llm(context_items)
                fs_response = self.llm.generate_response(data['query'], context_str, retrieved_items=context_items)
                retrieved_uris = [item['uri'] for item in context_items]
            except Exception as e:
                print(f"Few-shot Error: {e}")
                fs_response = str(e)
                retrieved_uris = []
            
            # --- Debug Log ---
            with open("debug_eval_log.txt", "a", encoding="utf-8") as dlog:
                dlog.write(f"\n[{data['id']}] {data['query']}\n")
                dlog.write(f"  ZS Response: {zs_response}\n")
                dlog.write(f"  FS Response: {fs_response}\n")
                dlog.write(f"  FS Retrieved: {retrieved_uris}\n")
            
            # --- 3. Metrics Calculation ---
            
            # Helper to calculate accuracy based on keyword labels
            def calc_accuracy(text, labels):
                if not labels: return 0.0
                norm_text = normalize_text(text).lower()
                hits = sum(1 for label in labels if label.lower() in norm_text)
                return hits / len(labels)

            # Ground Truths
            gt_labels = data['ground_truth_labels']
            gt_uris = data['ground_truth_uri']
            if isinstance(gt_uris, str): gt_uris = [gt_uris]
            
            # A. Label Accuracy
            acc_zs = calc_accuracy(zs_response, gt_labels)
            acc_fs = calc_accuracy(fs_response, gt_labels)
            
            # B. Citation Recall (Only relevant for RAG)
            citation_hit = any(u in retrieved_uris for u in gt_uris)
            citation_score = 1.0 if citation_hit else 0.0
            
            # C. NLP Metrics (BERTScore, ROUGE, Token-F1, chrF, EditSim, Jaccard)
            pseudo_ref = " ".join(gt_labels)
            
            # ROUGE-L
            rouge_zs = rouge_l_f1(pseudo_ref, zs_response)
            rouge_fs = rouge_l_f1(pseudo_ref, fs_response)
            
            # BERTScore
            bert_zs = bertscore_f1(pseudo_ref, zs_response)
            bert_fs = bertscore_f1(pseudo_ref, fs_response)
            
            # Token-F1
            tok_f1_zs = token_f1(pseudo_ref, zs_response)
            tok_f1_fs = token_f1(pseudo_ref, fs_response)
            
            # chrF
            chrf_zs = chrF(pseudo_ref, zs_response)
            chrf_fs = chrF(pseudo_ref, fs_response)
            
            # EditSim
            editsim_zs = edit_sim(pseudo_ref, zs_response)
            editsim_fs = edit_sim(pseudo_ref, fs_response)
            
            # Jaccard
            jaccard_zs = jaccard(pseudo_ref, zs_response)
            jaccard_fs = jaccard(pseudo_ref, fs_response)
            
            # D. Retrieval Metrics (Precision, nDCG)
            # Find rank of first correct URI
            rank = -1
            for idx, r_uri in enumerate(retrieved_uris):
                if r_uri in gt_uris:
                    rank = idx + 1
                    break
            
            if rank > 0:
                ndcg_10 = 1.0 / math.log2(rank + 1)
            else:
                ndcg_10 = 0.0
                
            # Citation Precision
            relevant_count = sum(1 for r_uri in retrieved_uris if r_uri in gt_uris)
            citation_precision = relevant_count / len(retrieved_uris) if retrieved_uris else 0.0

            result_entry = {
                "id": data['id'],
                "query": data['query'],
                "zero_shot": {
                    "response": zs_response,
                    "metrics": {
                        "accuracy": acc_zs,
                        "rouge_l": rouge_zs,
                        "bert_score": bert_zs,
                        "token_f1": tok_f1_zs,
                        "chrf": chrf_zs,
                        "edit_sim": editsim_zs,
                        "jaccard": jaccard_zs
                    }
                },
                "few_shot": {
                    "response": fs_response,
                    "retrieved_uris": retrieved_uris,
                    "metrics": {
                        "accuracy": acc_fs,
                        "citation_hit": citation_score, # This effectively acts as Recall@k if k=len(retrieved)
                        "citation_precision": citation_precision,
                        "ndcg_10": ndcg_10,
                        "rouge_l": rouge_fs,
                        "bert_score": bert_fs,
                        "token_f1": tok_f1_fs,
                        "chrf": chrf_fs,
                        "edit_sim": editsim_fs,
                        "jaccard": jaccard_fs
                    }
                }
            }
            results.append(result_entry)
            print(f"  Result -> ZS Acc: {acc_zs:.2f} | FS Acc: {acc_fs:.2f} (Cite Recall: {citation_score:.2f}, Prec: {citation_precision:.2f}, nDCG: {ndcg_10:.2f})")

        # Summary Statistics
        def avg(key, subkey):
            vals = [r[key]["metrics"][subkey] for r in results]
            valid_vals = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
            return sum(valid_vals) / len(valid_vals) if valid_vals else 0.0

        summary = {
            "total_samples": len(dataset),
            "averages": {
                "zero_shot": {
                    "accuracy": avg("zero_shot", "accuracy"),
                    "rouge_l": avg("zero_shot", "rouge_l"),
                    "bert_score": avg("zero_shot", "bert_score"),
                    "token_f1": avg("zero_shot", "token_f1"),
                    "chrf": avg("zero_shot", "chrf"),
                    "edit_sim": avg("zero_shot", "edit_sim"),
                    "jaccard": avg("zero_shot", "jaccard")
                },
                "few_shot": {
                    "accuracy": avg("few_shot", "accuracy"),
                    "citation_recall": avg("few_shot", "citation_hit"),
                    "citation_precision": avg("few_shot", "citation_precision"),
                    "ndcg_10": avg("few_shot", "ndcg_10"),
                    "rouge_l": avg("few_shot", "rouge_l"),
                    "bert_score": avg("few_shot", "bert_score"),
                    "token_f1": avg("few_shot", "token_f1"),
                    "chrf": avg("few_shot", "chrf"),
                    "edit_sim": avg("few_shot", "edit_sim"),
                    "jaccard": avg("few_shot", "jaccard")
                }
            },
            "details": results
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            
        print(f"\nComparative Evaluation Complete!")
        print(f"ZS Accuracy: {summary['averages']['zero_shot']['accuracy']:.2f} | FS Accuracy: {summary['averages']['few_shot']['accuracy']:.2f}")
        print(f"ZS BERTScore: {summary['averages']['zero_shot']['bert_score']:.2f} | FS BERTScore: {summary['averages']['few_shot']['bert_score']:.2f}")
        print(f"FS nDCG@10: {summary['averages']['few_shot']['ndcg_10']:.2f} | FS Citation Precision: {summary['averages']['few_shot']['citation_precision']:.2f}")
        print(f"Results saved to {output_path}")
        
        return summary
