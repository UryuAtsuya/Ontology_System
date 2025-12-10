"""
openai_fewshot_eval.py
---------------------------------
概要:
  OpenAI 応答を few-shot / zero-shot で収集し、定量 (複数メトリクス)・定性の両面で比較評価する。
  - Robustness Evaluation: 同じクエリをN回実行して安定性を確認するために使用。
  - BERTScore は導入済みなら自動で使用。未導入なら ROUGE-L / token-F1 等を代替指標として出力。

Usage:
  python3 openai_fewshot_eval.py --fewshot_file Updated_Ontology.txt --query "東京タワーとスカイツリーの違いは？" --runs 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import math
from typing import List, Tuple, Dict
from collections import Counter

# -------------------------- optional deps --------------------------
# tqdm
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# rapidfuzz Levenshtein (fast); fallback to pure python
try:
    from rapidfuzz.distance import Levenshtein
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
                dp[j] = min(dp[j] + 1, prev + cost, dp[j-1] + 1)
                prev = temp
        return dp[-1]

# bert_score
_BERT_OK = True
try:
    import bert_score
except Exception:
    _BERT_OK = False

# OpenAI
try:
    from openai import OpenAI
except ImportError as e:
    print("openai ライブラリが見つかりません。`pip install openai` を実行してください。", file=sys.stderr)
    raise e

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
        # Suppress potential warnings by catching
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Use baseline check if available, else simple
            P, R, F1 = bert_score.score([candidate], [reference], lang="ja", rescale_with_baseline=True)
        return float(F1[0])
    except Exception as e:
        # Fallback
        try:
             P, R, F1 = bert_score.score([candidate], [reference], lang="ja", rescale_with_baseline=False)
             return float(F1[0])
        except:
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

# -------------------------- OpenAI wrapper --------------------------

def get_openai_client() -> OpenAI:
    # Try environment variable first
    api_key = os.environ.get("OPENAI_API_KEY")
    
    # Try secrets.toml second (manual parse to avoid toml dependency)
    if not api_key:
        secrets_path = os.path.join(os.path.dirname(__file__), ".streamlit/secrets.toml")
        if os.path.exists(secrets_path):
            try:
                with open(secrets_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if "OPENAI_API_KEY" in line:
                            # format: OPENAI_API_KEY = "sk-..."
                            parts = line.split("=", 1)
                            if len(parts) == 2:
                                val = parts[1].strip().strip('"').strip("'")
                                if val:
                                    api_key = val
                                    break
            except Exception as e:
                print(f"Error loading secrets.toml: {e}", file=sys.stderr)
    
    if not api_key:
        raise RuntimeError("OpenAI API Key not found. Please set OPENAI_API_KEY in environment or .streamlit/secrets.toml")
        
    return OpenAI(api_key=api_key)

def call_openai(client: OpenAI, prompt: str, system_instruction: str | None = None, model_name: str = "gpt-4o-mini", max_retries: int = 3) -> str:
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.7 
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"OpenAI Call Failed: {e}", file=sys.stderr)
                return ""
            time.sleep(1.0 * (attempt + 1))
    return ""

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
        s = rouge_l_f1(reference, c)  # 安定するスコアでソート
        scored.append((i, s, c))
    scored.sort(key=lambda x: x[1])
    return scored[:k]

# -------------------------- I/O --------------------------

def save_results_to_file(
    fewshot_answers: List[str],
    zeroshot_answers: List[str],
    metrics_fs: Dict[str, List[float]],
    metrics_zs: Dict[str, List[float]],
    diff_tokens: List[Tuple[str, int]],
    output_path: str,
) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("[few-shot answers]\n")
        for i, s in enumerate(fewshot_answers):
            f.write(f"[{i}] {s}\n\n")
        f.write("\n[zero-shot answers]\n")
        for i, s in enumerate(zeroshot_answers):
            f.write(f"[{i}] {s}\n\n")

        def write_stats(name: str, xs: List[float]):
            vals = [x for x in xs if not math.isnan(x)]
            avg = sum(vals)/len(vals) if vals else float('nan')
            f.write(f"{name}: {', '.join(f'{x:.3f}' if not math.isnan(x) else 'nan' for x in xs)}\n")
            f.write(f"  avg={avg:.3f}\n")

        f.write("\n[metrics: few-shot]\n")
        for k, v in metrics_fs.items():
            write_stats(k, v)

        f.write("\n[metrics: zero-shot]\n")
        for k, v in metrics_zs.items():
            write_stats(k, v)

        f.write("\n[diff tokens (few-shot unique)]\n")
        for tok, cnt in diff_tokens:
            f.write(f"{tok}\t{cnt}\n")

# -------------------------- main --------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fewshot_file", type=str, required=True, help="few-shot info file")
    p.add_argument("--query", type=str, required=True, help="query to evaluate")
    p.add_argument("--model", type=str, default="gpt-4o-mini")
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--output", type=str, default="openai_robustness_results", help="output prefix")
    args = p.parse_args()

    client = get_openai_client()

    # few-shot prompt construction
    if not os.path.exists(args.fewshot_file):
        print(f"Error: {args.fewshot_file} not found.", file=sys.stderr)
        return

    fewshot_doc = open(args.fewshot_file, "r", encoding="utf-8").read()
    system_instruction = "You are a helpful domain expert. Answer concisely and factually."
    
    # In Few-shot mode, we embed the ontology content into the prompt or system message
    # To mimic the previous behavior, passing it as context in prompt
    prompt_fs = f"Reference Information:\n{fewshot_doc}\n\n[Query]\n{args.query}"
    prompt_zs = args.query

    print(f"Starting Evaluation with {args.model} ({args.runs} runs)...")

    print("[1] Collecting Few-shot responses...")
    fewshot_answers: List[str] = []
    for _ in tqdm(range(args.runs)):
        fewshot_answers.append(call_openai(client, prompt_fs, system_instruction, model_name=args.model))

    print("[2] Collecting Zero-shot responses...")
    zeroshot_answers: List[str] = []
    for _ in tqdm(range(args.runs)):
        zeroshot_answers.append(call_openai(client, prompt_zs, system_instruction, model_name=args.model))

    # reference: First few-shot answer is treated as 'pseudo-gold' for comparison logic in this script context
    # ideally we should have a fixed gold standard, but the original script used this logic.
    reference = fewshot_answers[0] if fewshot_answers else ""

    # ---------------- Quantitative ----------------
    print("\n[3] Quantitative Eval (Ref = few-shot[0])")
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

    def avg(xs: List[float]) -> float:
        vals = [x for x in xs if not math.isnan(x)]
        return sum(vals)/len(vals) if vals else float('nan')

    def print_block(name: str, metrics: Dict[str, List[float]]):
        print(f"{name}:")
        for k, v in metrics.items():
            print(f"  {k} avg = {avg(v):.3f}")
        print()

    print_block("few-shot", metrics_fs)
    print_block("zero-shot", metrics_zs)

    # Diversity
    print("[4] Diversity (distinct-n)")
    for n in (1, 2, 3):
        print(f"  distinct-{n} few-shot = {distinct_n(fewshot_answers, n):.3f} / zero-shot = {distinct_n(zeroshot_answers, n):.3f}")

    # Qualitative Diff
    print("\n[5] Qualitative Diff (Few-shot unique terms Top-20)")
    diff_tokens = qualitative_diff(fewshot_answers, zeroshot_answers, top_k=20)
    for tok, cnt in diff_tokens:
        print(f"    {tok}\t{cnt}")

    # Worst Examples
    if reference:
        print("\n[6] Worst Cases (Lowest ROUGE-L)")
        for title, arr in [("few-shot", fewshot_answers[1:]), ("zero-shot", zeroshot_answers)]:
            w = worst_examples(reference, arr, k=3)
            print(f"  {title}:")
            for idx, s, txt in w:
                snippet = txt[:50].replace("\n", " ") + "..."
                print(f"    #{idx} score={s:.3f} -> {snippet}")

    # Save
    out_file = f"{args.output}_runs{args.runs}.txt"
    print(f"\n[7] Saved to: {out_file}")
    save_results_to_file(
        fewshot_answers, zeroshot_answers,
        metrics_fs, metrics_zs,
        diff_tokens, out_file
    )

if __name__ == "__main__":
    main()
