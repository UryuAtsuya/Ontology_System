"""
batch_eval.py
---------------------------------
概要:
  evaluation_dataset.json から複数のクエリを読み込み、
  OpenAI API (Few-shot/Zero-shot) で応答を収集し、
  定量的・定性的メトリクスを計算してJSON/Markdown形式で出力する。

使用方法:
  python3 batch_eval.py --config config.json

更新日: 2025-12-11
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import math
import json
from datetime import datetime
from typing import List, Tuple, Dict, Any
from collections import Counter
from pathlib import Path

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
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            P, R, F1 = bert_score.score([candidate], [reference], lang="ja", rescale_with_baseline=True)
        return float(F1[0])
    except Exception:
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
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        secrets_path = os.path.join(os.path.dirname(__file__), "../.streamlit/secrets.toml")
        if os.path.exists(secrets_path):
            try:
                with open(secrets_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if "OPENAI_API_KEY" in line:
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
        s = rouge_l_f1(reference, c)
        scored.append((i, s, c))
    scored.sort(key=lambda x: x[1])
    return scored[:k]

# -------------------------- statistics --------------------------

def calculate_stats(values: List[float]) -> Dict[str, float]:
    """平均値と標準偏差を計算"""
    valid_vals = [v for v in values if not math.isnan(v)]
    if not valid_vals:
        return {"avg": float('nan'), "std": float('nan'), "min": float('nan'), "max": float('nan')}
    
    avg = sum(valid_vals) / len(valid_vals)
    if len(valid_vals) > 1:
        variance = sum((x - avg) ** 2 for x in valid_vals) / len(valid_vals)
        std = math.sqrt(variance)
    else:
        std = 0.0
    
    return {
        "avg": round(avg, 4),
        "std": round(std, 4),
        "min": round(min(valid_vals), 4),
        "max": round(max(valid_vals), 4)
    }

# -------------------------- evaluation --------------------------

def evaluate_single_query(
    client: OpenAI,
    query_data: Dict[str, Any],
    fewshot_doc: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """1つのクエリを評価"""
    
    query_id = query_data['id']
    query = query_data['query']
    runs = config['runs_per_query']
    model = config['model']
    system_instruction = config['system_instruction']
    
    print(f"\n[{query_id}] {query}")
    
    # Few-shot prompt
    prompt_fs = f"Reference Information:\n{fewshot_doc}\n\n[Query]\n{query}"
    prompt_zs = query
    
    # Collect responses
    print(f"  Collecting Few-shot responses ({runs} runs)...")
    fewshot_answers: List[str] = []
    for _ in tqdm(range(runs), desc="Few-shot"):
        fewshot_answers.append(call_openai(client, prompt_fs, system_instruction, model_name=model))
        time.sleep(0.5)  # Rate limiting
    
    print(f"  Collecting Zero-shot responses ({runs} runs)...")
    zeroshot_answers: List[str] = []
    for _ in tqdm(range(runs), desc="Zero-shot"):
        zeroshot_answers.append(call_openai(client, prompt_zs, system_instruction, model_name=model))
        time.sleep(0.5)  # Rate limiting
    
    # Reference: First few-shot answer
    reference = fewshot_answers[0] if fewshot_answers else ""
    
    # Calculate quantitative metrics
    print(f"  Calculating quantitative metrics...")
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
    
    # Calculate qualitative metrics
    print(f"  Calculating qualitative metrics...")
    diff_tokens = qualitative_diff(fewshot_answers, zeroshot_answers, top_k=20)
    worst_fs = worst_examples(reference, fewshot_answers[1:], k=3) if len(fewshot_answers) > 1 else []
    worst_zs = worst_examples(reference, zeroshot_answers, k=3)
    
    # Diversity
    diversity_fs = {f"distinct_{n}": distinct_n(fewshot_answers, n) for n in (1, 2, 3)}
    diversity_zs = {f"distinct_{n}": distinct_n(zeroshot_answers, n) for n in (1, 2, 3)}
    
    # Build result
    result = {
        "query_id": query_id,
        "query": query,
        "category": query_data.get("category", "unknown"),
        "few_shot": {
            "responses": fewshot_answers,
            "quantitative": {
                metric: {
                    "values": values,
                    **calculate_stats(values)
                } for metric, values in metrics_fs.items()
            },
            "qualitative": {
                **diversity_fs,
                "unique_tokens": diff_tokens[:10],  # Top 10
                "worst_examples": [
                    {"index": idx, "score": round(score, 4), "snippet": txt[:100]}
                    for idx, score, txt in worst_fs
                ]
            }
        },
        "zero_shot": {
            "responses": zeroshot_answers,
            "quantitative": {
                metric: {
                    "values": values,
                    **calculate_stats(values)
                } for metric, values in metrics_zs.items()
            },
            "qualitative": {
                **diversity_zs,
                "worst_examples": [
                    {"index": idx, "score": round(score, 4), "snippet": txt[:100]}
                    for idx, score, txt in worst_zs
                ]
            }
        }
    }
    
    # Calculate improvement
    result["comparison"] = {}
    for metric in metrics_fs.keys():
        fs_avg = calculate_stats(metrics_fs[metric])["avg"]
        zs_avg = calculate_stats(metrics_zs[metric])["avg"]
        if not math.isnan(fs_avg) and not math.isnan(zs_avg) and zs_avg > 0:
            improvement = ((fs_avg - zs_avg) / zs_avg) * 100
            result["comparison"][metric] = f"{improvement:+.1f}%"
        else:
            result["comparison"][metric] = "N/A"
    
    return result

# -------------------------- output --------------------------

def save_results_json(results: Dict[str, Any], output_path: str):
    """結果をJSON形式で保存"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Results saved to: {output_path}")

def generate_summary_markdown(results: Dict[str, Any], output_path: str):
    """Markdownサマリーを生成"""
    metadata = results["metadata"]
    query_results = results["results"]
    
    lines = []
    lines.append("# Batch Evaluation Summary\n")
    lines.append(f"**Date**: {metadata['timestamp']}\n")
    lines.append(f"**Model**: {metadata['model']}\n")
    lines.append(f"**Runs per Query**: {metadata['runs_per_query']}\n")
    lines.append(f"**Total Queries**: {metadata['total_queries']}\n")
    
    lines.append("\n## Overall Results\n")
    
    # Calculate overall averages
    if query_results:
        metrics = list(query_results[0]["few_shot"]["quantitative"].keys())
        lines.append("| Metric | Few-shot (Avg) | Zero-shot (Avg) | Improvement |\n")
        lines.append("|--------|----------------|-----------------|-------------|\n")
        
        for metric in metrics:
            fs_avgs = [r["few_shot"]["quantitative"][metric]["avg"] for r in query_results]
            zs_avgs = [r["zero_shot"]["quantitative"][metric]["avg"] for r in query_results]
            
            fs_avg = sum(v for v in fs_avgs if not math.isnan(v)) / len([v for v in fs_avgs if not math.isnan(v)]) if any(not math.isnan(v) for v in fs_avgs) else float('nan')
            zs_avg = sum(v for v in zs_avgs if not math.isnan(v)) / len([v for v in zs_avgs if not math.isnan(v)]) if any(not math.isnan(v) for v in zs_avgs) else float('nan')
            
            if not math.isnan(fs_avg) and not math.isnan(zs_avg) and zs_avg > 0:
                improvement = ((fs_avg - zs_avg) / zs_avg) * 100
                lines.append(f"| {metric} | {fs_avg:.3f} | {zs_avg:.3f} | {improvement:+.1f}% |\n")
            else:
                lines.append(f"| {metric} | {fs_avg:.3f} | {zs_avg:.3f} | N/A |\n")
    
    lines.append("\n## Query-by-Query Results\n")
    
    for result in query_results:
        lines.append(f"\n### {result['query_id']}: {result['query']}\n")
        lines.append(f"\n**Category**: {result['category']}\n")
        
        lines.append("\n| Metric | Few-shot | Zero-shot | Improvement |\n")
        lines.append("|--------|----------|-----------|-------------|\n")
        
        for metric in result["few_shot"]["quantitative"].keys():
            fs = result["few_shot"]["quantitative"][metric]["avg"]
            zs = result["zero_shot"]["quantitative"][metric]["avg"]
            imp = result["comparison"].get(metric, "N/A")
            lines.append(f"| {metric} | {fs:.3f} | {zs:.3f} | {imp} |\n")
        
        # Qualitative
        lines.append("\n**Qualitative Analysis**:\n")
        fs_qual = result["few_shot"]["qualitative"]
        zs_qual = result["zero_shot"]["qualitative"]
        
        for n in (1, 2, 3):
            key = f"distinct_{n}"
            lines.append(f"- Distinct-{n}: {fs_qual[key]:.3f} (Few-shot) vs {zs_qual[key]:.3f} (Zero-shot)\n")
        
        if fs_qual.get("unique_tokens"):
            lines.append(f"- Unique Few-shot Terms: {', '.join(f'{tok} ({cnt})' for tok, cnt in fs_qual['unique_tokens'][:5])}\n")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    
    print(f"✅ Summary saved to: {output_path}")

# -------------------------- main --------------------------

def main():
    p = argparse.ArgumentParser(description="Batch evaluation using evaluation_dataset.json")
    p.add_argument("--config", type=str, default="batch_evaluation/config.json", help="Config file path")
    p.add_argument("--dataset", type=str, default="evaluation_dataset.json", help="Dataset file path")
    args = p.parse_args()
    
    # Load config
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        return
    
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # Load dataset
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset file not found: {args.dataset}", file=sys.stderr)
        return
    
    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    # Load few-shot document
    fewshot_file = config['fewshot_file']
    if not os.path.exists(fewshot_file):
        print(f"Error: Few-shot file not found: {fewshot_file}", file=sys.stderr)
        return
    
    with open(fewshot_file, "r", encoding="utf-8") as f:
        fewshot_doc = f.read()
    
    # Initialize OpenAI client
    client = get_openai_client()
    
    print("="*60)
    print("🚀 Batch Evaluation Started")
    print("="*60)
    print(f"Model: {config['model']}")
    print(f"Runs per query: {config['runs_per_query']}")
    print(f"Total queries: {len(dataset)}")
    print("="*60)
    
    # Evaluate each query
    results = []
    for query_data in dataset:
        result = evaluate_single_query(client, query_data, fewshot_doc, config)
        results.append(result)
    
    # Prepare output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_json = output_dir / f"batch_eval_{timestamp}.json"
    output_md = output_dir / f"batch_eval_{timestamp}_summary.md"
    
    # Build final result
    final_result = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": config['model'],
            "runs_per_query": config['runs_per_query'],
            "total_queries": len(dataset),
            "fewshot_file": config['fewshot_file']
        },
        "results": results
    }
    
    # Save
    save_results_json(final_result, str(output_json))
    generate_summary_markdown(final_result, str(output_md))
    
    print("\n" + "="*60)
    print("✅ Batch Evaluation Completed!")
    print("="*60)

if __name__ == "__main__":
    main()
