import os
import re
import math
from typing import List, Dict, Any, Optional
from modules.evaluation import ThesisEvaluator

# --- 1. Fact Extraction & Definitions ---

# Reference Data for Hallucination Check (Ground Truth)
# IDs correspond to evaluation_dataset.json
REFERENCE_FACTS_MAP = {
    "T1_001": {"name": "サンシャイン60", "height": "240", "floors": "60", "year": "1978"}, 
    "T1_002": {"name": "東京スカイツリー", "height": "634", "year": "2012"},
    "T1_003": {"name": "新宿三井ビルディング", "year": "1974", "floors": "55"},
    "T2_001": {"name": "あべのハルカス", "height": "300", "floors": "60", "year": "2014"},
    "T2_002": {"name": "姫路城", "year": "1609", "structure": "木造"},
    "T2_003": {"name": "法隆寺五重塔", "floors": "5", "structure": "木造"},
    "T3_001": {"height": ["333", "634"]}, # Allow either for comparison
    "T3_002": {} # List type - skip strict fact check
}

def extract_facts(text: str) -> Dict[str, Optional[str]]:
    """
    Extract key facts from text using regex.
    Returns dictionary with extracted values or None.
    """
    facts = {
        "height": None, "floors": None, "year": None,
        "usage": None, "structure": None
    }
    
    # Simple regex rules
    patterns = {
        "height": r"(高さ|height)[\s:：]*(\d{2,4})", # Matches 333, 634, 200...
        "floors": r"(階数|floors)[\s:：]*(\d{1,3})",
        "year": r"(竣工|建設|完成|built in)[\s:：]*(\d{4})",
        "usage": r"(用途|usage)[\s:：]*([^\n、。]+)",
        "structure": r"(構造|structure)[\s:：]*([^\n、。]+)"
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Group 2 is usually the value
            val = match.group(2).strip()
            # specific cleanup if needed
            facts[key] = val
            
    return facts

# --- 2. Robustness Metrics Logic ---

def compute_hr(data_list: List[Dict[str, str]]) -> float:
    """
    Hallucination Rate (HR):
    Checks if extracted facts contradict the REFERENCE_FACTS_MAP[id].
    Args:
        data_list: List of dicts checks {'id': 'T1_001', 'text': 'response string'}
    """
    hallucinated_count = 0
    total = len(data_list)
    if total == 0: return 0.0
    
    for item in data_list:
        outcome_id = item['id']
        text = item['text']
        reference = REFERENCE_FACTS_MAP.get(outcome_id, {})
        
        extracted = extract_facts(text)
        is_hallucinated = False
        
        # Logic: If mentioned fact differs from reference -> Hallucination
        # Strict Check on ALL defined reference keys
        for key, ref_val in reference.items():
            if key not in extracted or extracted[key] is None:
                continue # Fact not in output, safe
            
            ext_val = extracted[key]
            
            # Helper to check match (Exact or In-List)
            def is_match(ref, ext):
                if isinstance(ref, list):
                    return any(str(r) in str(ext) for r in ref)
                return str(ref) in str(ext)
            
            # If extracted value does NOT contain the reference value -> Hallucination
            # e.g. Extracted "100m", Reference "333". "333" in "100m" -> False -> Error.
            # e.g. Extracted "約333メートル", Reference "333". "333" in "..." -> True -> OK.
            if not is_match(ref_val, ext_val):
                is_hallucinated = True
                break
        
        if is_hallucinated:
            hallucinated_count += 1
            
    return hallucinated_count / total

def compute_csr(data_list: List[Dict[str, str]]) -> float:
    """
    Constraint Satisfaction Rate (CSR):
    1. Output must not be empty.
    2. If numeric facts (year, height) are mentioned, they must be 'valid' (sanity check).
       - Year: 1000-2100
       - Height: 10-2000
    NO keyword dependency.
    """
    satisfied_count = 0
    total = len(data_list)
    if total == 0: return 0.0
    
    for item in data_list:
        text = item['text']
        if not text or not text.strip():
            continue # Empty -> Failed
            
        extracted = extract_facts(text)
        is_satisfied = True
        
        # Sanity Checks on Stats
        if extracted["year"]:
            try:
                # Remove non-digits for range check
                y_str = re.sub(r'\D', '', extracted["year"])
                if y_str:
                    y = int(y_str)
                    if not (1000 <= y <= 2100):
                        is_satisfied = False
            except:
                pass # parsing fail -> assume OK to be robust
                
        if extracted["height"]:
            try:
                h_str = re.sub(r'\D', '', extracted["height"])
                if h_str:
                    h = int(h_str)
                    if not (10 <= h <= 2000): # 10m to 2000m
                        is_satisfied = False
            except:
                pass

        if is_satisfied:
            satisfied_count += 1
            
    return satisfied_count / total

# --- 3. BERTScore Post-Processing ---
from typing import Tuple

def _load_gold_map(dataset_path: str) -> Dict[str, str]:
    """
    evaluation_dataset.json から {id: gold_text} を作る。
    キー名は環境差があるので複数候補を順に探す。
    """
    import json
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gold_map = {}

    # datasetが list でも dict でも対応
    samples = data["samples"] if isinstance(data, dict) and "samples" in data else data

    # gold候補キー: ground_truth_labels is list, join it.
    gold_keys = ["ground_truth_labels", "gold", "reference", "reference_answer", "gold_answer", "answer", "expected", "ground_truth"]

    for s in samples:
        sid = s.get("id")
        if not sid:
            continue
        g = None
        for k in gold_keys:
            if k in s:
                val = s[k]
                if isinstance(val, list):
                    g = " ".join(val) # Join usage/labels
                elif isinstance(val, str) and val.strip():
                    g = val.strip()
                if g: break
        if g is not None:
            gold_map[sid] = g

    return gold_map


def _compute_bertscore_mean(cands: List[str], refs: List[str],
                           model_type: str = "bert-base-multilingual-cased",
                           device: str = "cpu") -> float:
    """
    BERTScore(F1)の平均を返す。
    bert-score が無い場合は例外を投げる（= インストール促す）。
    """
    try:
        from bert_score import score as bert_score_score
    except ImportError:
        print("Warning: bert-score not installed. Returning 0.0")
        return 0.0

    # rescale_with_baseline=True の方が一般に解釈しやすい（0-1に寄せる）ので推奨
    # Note: baseline rescaling might fail if baseline not found. fallback to False if needed.
    try:
        P, R, F1 = bert_score_score(
            cands, refs,
            model_type=model_type,
            device=device,
            lang="ja", # Explicit lang might help
            rescale_with_baseline=True
        )
    except Exception as e:
        print(f"BERTScore (rescaled) failed: {e}. Trying without rescale.")
        P, R, F1 = bert_score_score(
            cands, refs,
            model_type=model_type,
            device=device,
            lang="ja",
            rescale_with_baseline=False
        )
        
    return float(F1.mean().item())

def load_api_key():
    # 1. Try Environment Variable
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key
    
    # 2. Try Streamlit secrets (Manual Parse)
    try:
        if os.path.exists(".streamlit/secrets.toml"):
            with open(".streamlit/secrets.toml", "r") as f:
                for line in f:
                    if "OPENAI_API_KEY" in line:
                        parts = line.split("=", 1)
                        if len(parts) == 2:
                            val = parts[1].strip().strip('"').strip("'")
                            return val
    except Exception as e:
        print(f"Failed to load secrets.toml: {e}")
        
    return None

def print_table(runs, zs_avgs, zs_hr, zs_csr, fs_avgs, fs_hr, fs_csr, output_file="robustness_results.txt"):
    table_str = f"\n### 表1: 手法別安定性評価結果 ({runs} Runs)\n"
    # Order: EditSim, Token-F1, Jaccard, ROUGE-L, chrF, BERTScore, HR, CSR
    header = "| Method | EditSim | Token-F1 | Jaccard | ROUGE-L | chrF | BERTScore | HR (Low is better) | CSR (High is better) |"
    sep = "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    
    table_str += header + "\n" + sep + "\n"
    
    def row(name, m, hr, csr):
        return f"| {name} | {m['edit_sim']:.3f} | {m['token_f1']:.3f} | {m['jaccard']:.3f} | {m['rouge_l']:.3f} | {m['chrf']:.3f} | {m['bert_score']:.3f} | {hr:.2%} | {csr:.2%} |"

    table_str += row("**Proposed (RAG)**", fs_avgs, fs_hr, fs_csr) + "\n"
    table_str += row("**Zero-shot**", zs_avgs, zs_hr, zs_csr) + "\n"
    
    print(table_str)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(table_str)
    print(f"Summary table saved to {output_file}")

def main():
    print("--- 卒論システム評価実行 (Robustness Test / HR & CSR) ---")
    
    api_key = load_api_key()
    if not api_key:
        api_key = input("OpenAI API Keyが見つかりません。入力してください: ").strip()
    if not api_key:
        print("API Keyが必須です。終了します。")
        return

    import sys
    
    try:
        evaluator = ThesisEvaluator(api_key)
        
        # RUN CONFIGURATION
        # Default to 10, allow override via CLI
        if len(sys.argv) > 1:
            try:
                RUNS = int(sys.argv[1])
            except ValueError:
                print("Invalid number of runs. Defaulting to 10.")
                RUNS = 10
        else:
            RUNS = 10
            
        dataset_path = "evaluation_dataset.json"
        
        # Load Gold Map for BERTScore
        gold_map = _load_gold_map(dataset_path)
        zs_cands, zs_refs = [], []
        fs_cands, fs_refs = [], []
        
        # Store all raw responses (ID + Text for robust checking)
        all_zs_responses = []
        all_fs_responses = []
        
        # Accumulators for standard metrics
        zs_metrics_sum = {
            "edit_sim": 0.0, "token_f1": 0.0, "jaccard": 0.0,
            "rouge_l": 0.0, "chrf": 0.0, "bert_score": 0.0
        }
        fs_metrics_sum = {
            "edit_sim": 0.0, "token_f1": 0.0, "jaccard": 0.0,
            "rouge_l": 0.0, "chrf": 0.0, "bert_score": 0.0
        }
        
        total_valid_samples = 0

        print(f"Executing {RUNS} runs for Robustness Evaluation...")
        
        for r in range(1, RUNS + 1):
            print(f"Run {r}/{RUNS}...")
            # Using temp output file
            summary = evaluator.run_thesis_evaluation(dataset_path=dataset_path, output_path=f"temp_res_{r}.json")
            
            current_samples = len(summary["details"])
            total_valid_samples += current_samples
            
            for item in summary["details"]:
                item_id = item["id"]
                gold = gold_map.get(item_id)
                
                # Zero-shot
                zs_resp = item["zero_shot"]["response"]
                all_zs_responses.append({"id": item_id, "text": zs_resp})
                
                zs_metrics = item["zero_shot"]["metrics"]
                zs_metrics_sum["edit_sim"] += zs_metrics.get("edit_sim", 0)
                zs_metrics_sum["token_f1"] += zs_metrics.get("token_f1", 0)
                zs_metrics_sum["jaccard"] += zs_metrics.get("jaccard", 0)
                zs_metrics_sum["rouge_l"] += zs_metrics.get("rouge_l", 0)
                zs_metrics_sum["chrf"] += zs_metrics.get("chrf", 0)
                # zs_metrics_sum["bert_score"] += zs_metrics.get("bert_score", 0)
                
                # Few-shot
                fs_resp = item["few_shot"]["response"]
                all_fs_responses.append({"id": item_id, "text": fs_resp})
                
                fs_metrics = item["few_shot"]["metrics"]
                fs_metrics_sum["edit_sim"] += fs_metrics.get("edit_sim", 0)
                fs_metrics_sum["token_f1"] += fs_metrics.get("token_f1", 0)
                fs_metrics_sum["jaccard"] += fs_metrics.get("jaccard", 0)
                fs_metrics_sum["rouge_l"] += fs_metrics.get("rouge_l", 0)
                fs_metrics_sum["chrf"] += fs_metrics.get("chrf", 0)
                # fs_metrics_sum["bert_score"] += fs_metrics.get("bert_score", 0)
                
                # Collect for BERTScore
                if gold:
                    zs_cands.append(zs_resp)
                    zs_refs.append(gold)
                    fs_cands.append(fs_resp)
                    fs_refs.append(gold)
                
            # Clean up temp file
            os.remove(f"temp_res_{r}.json") 

        if total_valid_samples == 0:
            print("No samples processed.")
            return

        # Compute Averages (Standard Metrics)
        avg_zs = {k: v / total_valid_samples for k, v in zs_metrics_sum.items()}
        avg_fs = {k: v / total_valid_samples for k, v in fs_metrics_sum.items()}

        # Compute BERTScore (Post-Process)
        print("Calculating BERTScore (Post-Processing)...")
        if zs_cands and zs_refs:
            avg_zs["bert_score"] = _compute_bertscore_mean(zs_cands, zs_refs)
        if fs_cands and fs_refs:
            avg_fs["bert_score"] = _compute_bertscore_mean(fs_cands, fs_refs)

        # Compute HR & CSR
        hr_zs = compute_hr(all_zs_responses)
        csr_zs = compute_csr(all_zs_responses)
        
        hr_fs = compute_hr(all_fs_responses)
        csr_fs = compute_csr(all_fs_responses)
        
        # Display Results
        output_file = f"robustness_results_{RUNS}.txt"
        print_table(RUNS, avg_zs, hr_zs, csr_zs, avg_fs, hr_fs, csr_fs, output_file=output_file)
        
        print("\nEvaluation Complete.")
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")

if __name__ == "__main__":
    main()
