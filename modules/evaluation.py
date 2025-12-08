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
import google.generativeai as genai

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

# bert_score
_BERT_OK = True
try:
    import bert_score  # type: ignore
except Exception:
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
    return normalize_text(s).lower().split()

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

def bertscore_f1(reference: str, candidate: str) -> float:
    if not _BERT_OK:
        return float('nan')
    try:
        P, R, F1 = bert_score.score([candidate], [reference], lang="ja", rescale_with_baseline=True)
        return float(F1[0])
    except Exception:
        try:
            P, R, F1 = bert_score.score([candidate], [reference], lang="ja", rescale_with_baseline=False)
            return float(F1[0])
        except Exception:
            return float('nan')

# diversity
def distinct_n(corpus: List[str], n: int) -> float:
    all_ngrams = set()
    total = 0
    for s in corpus:
        toks = tokenize_words(s)
        grams = ngrams(toks, n)
        total += max(len(grams), 1)
        all_ngrams.update(grams)
    return len(all_ngrams) / max(total, 1)

# -------------------------- Factual Accuracy (NLI) --------------------------

@dataclass
class FactualScores:
    nli_entail_rate: float
    evidence_recall: float

def build_nli_pipeline(model_name: str = "joeddav/xlm-roberta-large-xnli"):
    if not _TRANSFORMERS_OK:
        return None
    try:
        clf = pipeline("text-classification", model=model_name, tokenizer=model_name)
        return clf
    except Exception:
        return None

def nli_entailment_rate(answer: str, gold_sentences: List[str], clf) -> float:
    if not clf or not gold_sentences:
        return 0.0
    ans_sents = safe_split_sentences(answer)
    if not ans_sents:
        return 0.0
    entailed = 0
    for s in ans_sents:
        ok = False
        for g in gold_sentences:
            try:
                res = clf({"text": g, "text_pair": s}, return_all_scores=True)
                scores = res[0] if isinstance(res, list) else res
                best = max(scores, key=lambda x: x.get("score", 0.0))
                label = best.get("label", "").upper()
                if "ENTAIL" in label:
                    ok = True
                    break
            except Exception:
                continue
        if ok:
            entailed += 1
    return entailed / max(len(ans_sents), 1)

def evidence_ngram_recall(answer: str, gold_text: str, n: int = 2) -> float:
    ans_toks = simple_tokenize(answer)
    gold_toks = simple_tokenize(gold_text)
    if not ans_toks or not gold_toks:
        return 0.0
    gold_ngrams = ngram_set(gold_toks, n=n)
    ans_ngrams = ngram_set(ans_toks, n=n)
    inter = len(gold_ngrams & ans_ngrams)
    return inter / max(len(gold_ngrams), 1)

def compute_factual_scores(answer: str, gold_text: str, clf) -> FactualScores:
    gold_sents = safe_split_sentences(gold_text)
    entail_rate = nli_entailment_rate(answer, gold_sents, clf) if clf else 0.0
    ev_recall = evidence_ngram_recall(answer, gold_text, n=2)
    return FactualScores(nli_entail_rate=entail_rate, evidence_recall=ev_recall)

# -------------------------- Database Logging --------------------------

def init_sqlite(db_path: str):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS runs (
          run_id TEXT PRIMARY KEY,
          timestamp REAL,
          query TEXT,
          model TEXT,
          runs INTEGER,
          fewshot_file TEXT,
          gold_file TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS answers (
          run_id TEXT,
          mode TEXT,
          idx INTEGER,
          text TEXT,
          PRIMARY KEY (run_id, mode, idx)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
          run_id TEXT PRIMARY KEY,
          acc_fs_mean REAL,
          acc_zs_mean REAL,
          bert_fs_mean REAL,
          bert_zs_mean REAL,
          diff_vocab_count INTEGER,
          factual_fs_mean REAL,
          factual_zs_mean REAL
        )
    """)
    conn.commit()
    conn.close()

def log_to_sqlite(db_path: str, run_id: str, meta: Dict, results: Dict):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Insert Run
    cur.execute("""
        INSERT OR REPLACE INTO runs(run_id, timestamp, query, model, runs, fewshot_file, gold_file)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        run_id, time.time(), meta["query"], meta["model"], meta["runs"], meta.get("fewshot_file", ""), meta.get("gold_file", "")
    ))
    
    # Insert Answers
    for i, txt in enumerate(results["fewshot_answers"]):
        cur.execute("INSERT OR REPLACE INTO answers VALUES (?, ?, ?, ?)", (run_id, "fewshot", i, txt))
    for i, txt in enumerate(results["zeroshot_answers"]):
        cur.execute("INSERT OR REPLACE INTO answers VALUES (?, ?, ?, ?)", (run_id, "zeroshot", i, txt))
        
    # Insert Metrics
    # Calculate means
    def mean(xs): return sum(xs)/len(xs) if xs else 0.0
    
    acc_fs = mean(results["metrics_fs"]["ROUGE-L"]) # Using ROUGE-L as proxy for general accuracy in summary
    acc_zs = mean(results["metrics_zs"]["ROUGE-L"])
    
    bert_fs = mean(results["metrics_fs"].get("BERTScore", []))
    bert_zs = mean(results["metrics_zs"].get("BERTScore", []))
    
    fact_fs = 0.0
    if results.get("factual_fs"):
        fact_fs = mean([(x.nli_entail_rate + x.evidence_recall)/2 for x in results["factual_fs"]])
        
    fact_zs = 0.0
    if results.get("factual_zs"):
        fact_zs = mean([(x.nli_entail_rate + x.evidence_recall)/2 for x in results["factual_zs"]])

    cur.execute("""
        INSERT OR REPLACE INTO metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        run_id, acc_fs, acc_zs, bert_fs, bert_zs, len(results["diff_tokens"]), fact_fs, fact_zs
    ))
    
    conn.commit()
    conn.close()

# -------------------------- Gemini wrapper --------------------------

def call_gemini(prompt: str, system_instruction: str | None = None, model_name: str = "gemini-1.5-flash", max_retries: int = 3, api_key: str = "") -> str:
    if api_key:
        genai.configure(api_key=api_key)
    
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel(model_name, system_instruction=system_instruction)
            resp = model.generate_content(prompt)
            if hasattr(resp, "text") and resp.text:
                return resp.text.strip()
            time.sleep(1.0)
        except Exception as e:
            if attempt == max_retries - 1:
                return f"Error: {e}"
            time.sleep(1.5 * (attempt + 1))
    return "Error: Empty Response"

# -------------------------- qualitative --------------------------

def qualitative_diff(fewshot_answers: List[str], zeroshot_answers: List[str], top_k: int = 20) -> List[Tuple[str, int]]:
    fs = " ".join(fewshot_answers).lower()
    zs = " ".join(zeroshot_answers).lower()
    fs_tokens = Counter(tokenize_words(fs))
    zs_tokens = Counter(tokenize_words(zs))
    diff = fs_tokens - zs_tokens
    return diff.most_common(top_k)

def worst_examples(reference: str, candidates: List[str], k: int = 3) -> List[Tuple[int, float, str]]:
    scored = []
    for i, c in enumerate(candidates):
        s = rouge_l_f1(reference, c)
        scored.append((i, s, c))
    scored.sort(key=lambda x: x[1])
    return scored[:k]

# -------------------------- Main Evaluator Class --------------------------

class GeminiEvaluator:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def run_evaluation(
        self,
        query: str,
        fewshot_doc: str,
        runs: int = 5,
        model_name: str = "gemini-1.5-flash",
        gold_text: str | None = None,
        enable_factual: bool = False,
        progress_callback: Callable[[float, str], None] | None = None
    ) -> Dict[str, Any]:
        
        system_instruction = "You are a helpful domain expert. Answer concisely and factually."
        prompt_fs = f"{fewshot_doc}\n\n[Query]\n{query}"
        prompt_zs = query

        # 1. Collect Few-shot
        fewshot_answers: List[str] = []
        for i in range(runs):
            if progress_callback:
                progress_callback((i) / (runs * 2), f"Collecting Few-shot response {i+1}/{runs}...")
            fewshot_answers.append(call_gemini(prompt_fs, system_instruction, model_name=model_name, api_key=self.api_key))

        # 2. Collect Zero-shot
        zeroshot_answers: List[str] = []
        for i in range(runs):
            if progress_callback:
                progress_callback((runs + i) / (runs * 2), f"Collecting Zero-shot response {i+1}/{runs}...")
            zeroshot_answers.append(call_gemini(prompt_zs, system_instruction, model_name=model_name, api_key=self.api_key))

        if progress_callback:
            progress_callback(1.0, "Calculating metrics...")

        reference = fewshot_answers[0] if fewshot_answers else ""

        # 3. Quantitative Metrics (Standard)
        def eval_all(cands: List[str]) -> Dict[str, List[float]]:
            mets = {
                "EditSim": [],
                "TokenF1": [],
                "Jaccard": [],
                "ROUGE-L": [],
                "chrF": [],
            }
            if _BERT_OK:
                mets["BERTScore"] = []
            for c in cands:
                mets["EditSim"].append(edit_sim(reference, c))
                mets["TokenF1"].append(token_f1(reference, c))
                mets["Jaccard"].append(jaccard(reference, c))
                mets["ROUGE-L"].append(rouge_l_f1(reference, c))
                mets["chrF"].append(chrF(reference, c))
                if _BERT_OK:
                    mets["BERTScore"].append(bertscore_f1(reference, c))
            return mets

        metrics_fs = eval_all(fewshot_answers)
        metrics_zs = eval_all(zeroshot_answers)

        # 4. Factual Accuracy (Optional)
        factual_fs: List[FactualScores] = []
        factual_zs: List[FactualScores] = []
        
        if enable_factual and gold_text:
            if progress_callback:
                progress_callback(1.0, "Loading NLI model (this may take a while)...")
            
            clf = build_nli_pipeline()
            if clf:
                if progress_callback:
                    progress_callback(1.0, "Computing Factual Scores...")
                
                for a in fewshot_answers:
                    factual_fs.append(compute_factual_scores(a, gold_text, clf))
                for a in zeroshot_answers:
                    factual_zs.append(compute_factual_scores(a, gold_text, clf))
            else:
                # Fallback if NLI fails but gold text exists (only evidence recall)
                for a in fewshot_answers:
                    factual_fs.append(FactualScores(0.0, evidence_ngram_recall(a, gold_text)))
                for a in zeroshot_answers:
                    factual_zs.append(FactualScores(0.0, evidence_ngram_recall(a, gold_text)))

        # 5. Diversity & Qualitative
        diversity_fs = {f"distinct-{n}": distinct_n(fewshot_answers, n) for n in (1, 2, 3)}
        diversity_zs = {f"distinct-{n}": distinct_n(zeroshot_answers, n) for n in (1, 2, 3)}
        diff_tokens = qualitative_diff(fewshot_answers, zeroshot_answers, top_k=20)
        worst_fs = worst_examples(reference, fewshot_answers[1:], k=3)
        worst_zs = worst_examples(reference, zeroshot_answers, k=3)

        results = {
            "fewshot_answers": fewshot_answers,
            "zeroshot_answers": zeroshot_answers,
            "metrics_fs": metrics_fs,
            "metrics_zs": metrics_zs,
            "diversity_fs": diversity_fs,
            "diversity_zs": diversity_zs,
            "diff_tokens": diff_tokens,
            "worst_fs": worst_fs,
            "worst_zs": worst_zs,
            "reference": reference,
            "factual_fs": factual_fs,
            "factual_zs": factual_zs
        }

        # 6. Log to DB
        try:
            db_path = "runs/eval_runs.sqlite"
            init_sqlite(db_path)
            run_id = f"run_{int(time.time())}"
            meta = {
                "query": query,
                "model": model_name,
                "runs": runs,
                "fewshot_file": "Ontology.txt", # simplified
                "gold_file": "User Input" if gold_text else ""
            }
            log_to_sqlite(db_path, run_id, meta, results)
        except Exception as e:
            print(f"DB Log Error: {e}")

        return results

    def run_batch_evaluation(
        self,
        queries: List[str],
        fewshot_doc: str,
        runs: int = 3,
        model_name: str = "gemini-1.5-flash",
        gold_texts: Optional[List[str]] = None,
        enable_factual: bool = False,
        progress_callback: Callable[[float, str], None] | None = None
    ) -> List[Dict[str, Any]]:
        """
        Runs evaluation for a list of queries.
        """
        batch_results = []
        total_queries = len(queries)
        
        for idx, query in enumerate(queries):
            gold = gold_texts[idx] if gold_texts and idx < len(gold_texts) else None
            
            # Update progress for the batch
            if progress_callback:
                progress_callback(idx / total_queries, f"Processing Query {idx+1}/{total_queries}...")
            
            # Run single evaluation
            # Note: We pass None for progress_callback to avoid nested progress updates conflicting
            res = self.run_evaluation(
                query=query,
                fewshot_doc=fewshot_doc,
                runs=runs,
                model_name=model_name,
                gold_text=gold,
                enable_factual=enable_factual,
                progress_callback=None 
            )
            batch_results.append(res)
            
        if progress_callback:
            progress_callback(1.0, "Batch Evaluation Complete!")
            
        return batch_results

def export_for_human_eval(batch_results: List[Dict[str, Any]], queries: List[str], filename: str = "human_eval.csv"):
    """
    Exports batch results to a CSV format suitable for human evaluation.
    Columns: Query, Mode, Answer, Reference (Gold), Human_Score (Empty)
    """
    import pandas as pd
    
    rows = []
    for i, res in enumerate(batch_results):
        query = queries[i]
        
        # Few-shot answers
        for ans in res["fewshot_answers"]:
            rows.append({
                "Query": query,
                "Mode": "Few-shot",
                "Answer": ans,
                "Reference": res.get("reference", ""), # First few-shot answer is used as ref in single eval, but maybe gold is better if available
                "Human_Score": ""
            })
            
        # Zero-shot answers
        for ans in res["zeroshot_answers"]:
            rows.append({
                "Query": query,
                "Mode": "Zero-shot",
                "Answer": ans,
                "Reference": res.get("reference", ""),
                "Human_Score": ""
            })
            
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    return filename
