import os
from modules.evaluation import ThesisEvaluator

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
                        # Simple parse: key = "value"
                        parts = line.split("=", 1)
                        if len(parts) == 2:
                            val = parts[1].strip().strip('"').strip("'")
                            return val
    except Exception as e:
        print(f"Failed to load secrets.toml: {e}")
        
    return None

def main():
    print("--- 卒論システム評価実行 (Thesis Evaluation) ---")
    
    api_key = load_api_key()
    if not api_key:
        api_key = input("OpenAI API Keyが見つかりません。入力してください: ").strip()
        
    if not api_key:
        print("API Keyが必須です。終了します。")
        return

    try:
        evaluator = ThesisEvaluator(api_key)
        # T1, T2, T3のデータセットを用いて評価を実行
        summary = evaluator.run_thesis_evaluation(
            dataset_path="evaluation_dataset.json",
            output_path="thesis_eval_results.json"
        )
        
        print("\n" + "="*30)
        print(f"Total Samples: {summary['total_samples']}")
        print(f"Zero-shot | Accuracy: {summary['averages']['zero_shot']['accuracy']:.2%} | BERTScore: {summary['averages']['zero_shot']['bert_score']:.2f}")
        print(f"Few-shot  | Accuracy: {summary['averages']['few_shot']['accuracy']:.2%} | BERTScore: {summary['averages']['few_shot']['bert_score']:.2f}")
        print(f"Citation Recall (FS): {summary['averages']['few_shot']['citation_recall']:.2%}")
        print("="*30)
        print("詳細は 'thesis_eval_results.json' を確認してください。")
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")

if __name__ == "__main__":
    main()
