"""
evaluation/run_eval.py
---------------------------------
概要:
  Gemini 応答を few‑shot / zero‑shot で収集し、Gold Standard (正解データ) と比較評価する。
  - 基準 (Reference) として `evaluation/gold_standard.txt` を使用。
  - 定量: EditSim, Token‑F1, Jaccard, ROUGE‑L(F1), chrF, (optional) BERTScore(F1)
  - 定性: few‑shot 特有語彙 Top‑K、ワースト事例の抜粋、diversity (distinct‑n)

更新日: 2025/12/03
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
    from tqdm import tqdm  # type: ignore
except Exception:
    def tqdm(x, **kwargs):
        return x

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

# google generative AI
try:
    import google.generativeai as genai  # type: ignore
except ImportError as e:
    print("google‑generativeai が見つかりません。`pip install google-generativeai` を実行してください。", file=sys.stderr)
    raise e

# -------------------------- utils --------------------------

def normalize_text(s: str) -> str:
    s = s.strip()
    # 典型的な正規化 (空白統一・全角空白→半角など最小限)
    s = s.replace("\u3000", " ")
    s = " ".join(s.split())
    return s

def tokenize_words(s: str) -> List[str]:
    # 英日混在を想定して単純な分割 (必要なら後でMeCab等に差し替え可能)
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
    # 簡易 chrF: character n-gram precision/recall の Fβ
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
        P, R, F1 = bert_score.score([candidate], [reference], lang="ja", rescale_with_baseline=True)
        return float(F1[0])
    except Exception as e:
        print(f"BERTScore 計算エラー: {e}", file=sys.stderr)
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

# -------------------------- Gemini wrapper --------------------------

def configure_genai(api_key: str | None = None) -> None:
    k = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY_GEMINI")
    if not k:
        raise RuntimeError("Google API Key が未設定です。環境変数 GOOGLE_API_KEY に設定してください。")
    genai.configure(api_key=k)

def call_gemini(prompt: str, system_instruction: str | None = None, model_name: str = "gemini-2.5-flash", max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel(model_name, system_instruction=system_instruction)
            resp = model.generate_content(prompt)
            if hasattr(resp, "text") and resp.text:
                return resp.text.strip()
            return ""
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Gemini 呼び出し失敗: {e}", file=sys.stderr)
                return ""
            time.sleep(1.5 * (attempt + 1))
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
        f.write("[few‑shot answers]\n")
        for i, s in enumerate(fewshot_answers):
            f.write(f"[{i}] {s}\n\n")
        f.write("\n[zero‑shot answers]\n")
        for i, s in enumerate(zeroshot_answers):
            f.write(f"[{i}] {s}\n\n")

        def write_stats(name: str, xs: List[float]):
            vals = [x for x in xs if not math.isnan(x)]
            avg = sum(vals)/len(vals) if vals else float('nan')
            f.write(f"{name}: {', '.join(f'{x:.3f}' if not math.isnan(x) else 'nan' for x in xs)}\n")
            f.write(f"  avg={avg:.3f}\n")

        f.write("\n[metrics: few‑shot]\n")
        for k, v in metrics_fs.items():
            write_stats(k, v)

        f.write("\n[metrics: zero‑shot]\n")
        for k, v in metrics_zs.items():
            write_stats(k, v)

        f.write("\n[diff tokens (few‑shot unique)]\n")
        for tok, cnt in diff_tokens:
            f.write(f"{tok}\t{cnt}\n")

# -------------------------- main --------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fewshot_file", type=str, required=True, help="few‑shot 用ドキュメント (プロンプトに埋め込む)")
    p.add_argument("--query", type=str, required=True, help="評価する質問/クエリ")
    p.add_argument("--model", type=str, default="gemini-2.5-flash")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--output", type=str, default="evaluation/eval_result_gold", help="出力ファイル接頭辞")
    p.add_argument("--gold_standard", type=str, default="evaluation/gold_standard.txt", help="Gold Standard (正解) ファイルパス")
    args = p.parse_args()

    configure_genai()

    # few‑shot プロンプト
    fewshot_doc = open(args.fewshot_file, "r", encoding="utf-8").read()
    system_instruction = "You are a helpful domain expert. Answer concisely and factually."
    prompt_fs = f"{fewshot_doc}\n\n[Query]\n{args.query}"
    prompt_zs = args.query

    # Gold Standard の読み込み
    try:
        reference = open(args.gold_standard, "r", encoding="utf-8").read().strip()
        print(f"[0] Gold Standard loaded from {args.gold_standard}")
    except Exception as e:
        print(f"Error loading gold standard: {e}", file=sys.stderr)
        sys.exit(1)

    print("[1] few‑shot 応答の収集")
    fewshot_answers: List[str] = []
    for _ in tqdm(range(args.runs)):
        fewshot_answers.append(call_gemini(prompt_fs, system_instruction, model_name=args.model))

    print("[2] zero‑shot 応答の収集")
    zeroshot_answers: List[str] = []
    for _ in tqdm(range(args.runs)):
        zeroshot_answers.append(call_gemini(prompt_zs, system_instruction, model_name=args.model))

    # ---------------- 定量: 複数メトリクス ----------------
    print("\n[3] 定量評価 (reference = Gold Standard)")
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

    print_block("few‑shot", metrics_fs)
    print_block("zero‑shot", metrics_zs)

    # 多様性
    print("[4] 多様性 (distinct‑n)")
    for n in (1, 2, 3):
        print(f"  distinct-{n} few‑shot = {distinct_n(fewshot_answers, n):.3f} / zero‑shot = {distinct_n(zeroshot_answers, n):.3f}")

    # 定性的差分
    print("\n[5] 定性的差分 (few‑shot のみで現れやすい語彙 Top‑20)")
    diff_tokens = qualitative_diff(fewshot_answers, zeroshot_answers, top_k=20)
    for tok, cnt in diff_tokens:
        print(f"    {tok}\t{cnt}")

    # ワースト事例の可視化
    if reference:
        print("\n[6] ワースト事例 (ROUGE‑L F1 が低い順, 各3件)")
        for title, arr in [("few‑shot", fewshot_answers), ("zero‑shot", zeroshot_answers)]:
            w = worst_examples(reference, arr, k=3)
            print(f"  {title}:")
            for idx, s, txt in w:
                snippet = txt[:160].replace("\n", " ")
                ellipsis = "..." if len(txt) > 160 else ""
                print(f"    #{idx} score={s:.3f} -> {snippet}{ellipsis}")

    # 保存
    out_file = f"{args.output}_runs{args.runs}.txt"
    print(f"\n[7] 結果を保存: {out_file}")
    save_results_to_file(
        fewshot_answers, zeroshot_answers,
        metrics_fs, metrics_zs,
        diff_tokens, out_file
    )

if __name__ == "__main__":
    main()
