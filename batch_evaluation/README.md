# Batch Evaluation System

`openai_fewshot_eval.py` をベースにした、複数クエリのバッチ評価システムです。

## 概要

`evaluation_dataset.json` から複数のクエリを読み込み、Few-shot/Zero-shotの両方で応答を収集し、定量的・定性的メトリクスを計算して結果をJSON/Markdown形式で出力します。

## 特徴

### 定量的メトリクス
- **EditSim**: 編集距離ベースの類似度
- **TokenF1**: トークンレベルのF1スコア
- **Jaccard**: 集合ベースの類似度
- **ROUGE-L**: 最長共通部分系列
- **chrF**: 文字n-gramベースのF値
- **BERTScore** (optional): 意味レベルの類似度

### 定性的メトリクス
- **Distinct-n** (n=1,2,3): 多様性指標
- **Unique Tokens**: Few-shot特有の語彙 (Top 20)
- **Worst Examples**: 低スコア事例の抽出 (各3件)

### 統計情報
- 平均値 (avg)
- 標準偏差 (std)
- 最小値 (min)
- 最大値 (max)

## インストール

必要なライブラリ:
```bash
pip install openai tqdm rapidfuzz bert-score
```

## 使用方法

### 1. 設定ファイルの確認

`config.json` を確認・編集:
```json
{
  "model": "gpt-4o-mini",
  "runs_per_query": 10,
  "fewshot_file": "../Updated_Ontology.txt",
  "system_instruction": "You are a helpful domain expert. Answer concisely and factually.",
  "output_dir": "results"
}
```

### 2. 実行

```bash
cd batch_evaluation
python3 batch_eval.py --config config.json --dataset ../evaluation_dataset.json
```

### 3. 結果の確認

`results/` ディレクトリに以下のファイルが生成されます:
- `batch_eval_YYYYMMDD_HHMMSS.json` - 詳細な結果 (JSON)
- `batch_eval_YYYYMMDD_HHMMSS_summary.md` - サマリー (Markdown)

## 出力フォーマット

### JSON出力

```json
{
  "metadata": {
    "timestamp": "2025-12-11T14:15:00+09:00",
    "model": "gpt-4o-mini",
    "runs_per_query": 10,
    "total_queries": 3
  },
  "results": [
    {
      "query_id": "T1",
      "query": "クエリ文",
      "category": "basic_info",
      "few_shot": {
        "responses": ["...", "..."],
        "quantitative": {
          "ROUGE-L": {
            "values": [0.88, 0.87, ...],
            "avg": 0.875,
            "std": 0.015,
            "min": 0.85,
            "max": 0.90
          }
        },
        "qualitative": {
          "distinct_1": 0.45,
          "distinct_2": 0.78,
          "distinct_3": 0.92,
          "unique_tokens": [["麻布台", 15], ...],
          "worst_examples": [...]
        }
      },
      "zero_shot": {...},
      "comparison": {
        "ROUGE-L": "+600%",
        "chrF": "+397%"
      }
    }
  ]
}
```

### Markdown サマリー

```markdown
# Batch Evaluation Summary

**Date**: 2025-12-11 14:15:00
**Model**: gpt-4o-mini
**Runs per Query**: 10
**Total Queries**: 3

## Overall Results

| Metric | Few-shot (Avg) | Zero-shot (Avg) | Improvement |
|--------|----------------|-----------------|-------------|
| ROUGE-L | 0.875 | 0.125 | +600% |
| chrF | 0.895 | 0.180 | +397% |

## Query-by-Query Results

### T1: クエリ文

**Category**: basic_info

| Metric | Few-shot | Zero-shot | Improvement |
|--------|----------|-----------|-------------|
| ROUGE-L | 0.88 | 0.10 | +780% |
```

## カスタマイズ

### 試行回数の変更

`config.json` の `runs_per_query` を変更:
```json
{
  "runs_per_query": 20
}
```

### モデルの変更

`config.json` の `model` を変更:
```json
{
  "model": "gpt-4o"
}
```

### Few-shotファイルの変更

`config.json` の `fewshot_file` を変更:
```json
{
  "fewshot_file": "../custom_ontology.txt"
}
```

## トラブルシューティング

### OpenAI API Keyが見つからない

環境変数 `OPENAI_API_KEY` を設定するか、`.streamlit/secrets.toml` に記載してください:

```bash
export OPENAI_API_KEY="sk-..."
```

または

```toml
# .streamlit/secrets.toml
OPENAI_API_KEY = "sk-..."
```

### BERTScoreが使えない

BERTScoreは optional です。インストールしていない場合は自動的にスキップされます:

```bash
pip install bert-score
```

### Rate Limit エラー

`batch_eval.py` 内の `time.sleep(0.5)` を調整してください (行322, 328付近)。

## 既存システムとの違い

| 項目 | openai_fewshot_eval.py | batch_eval.py |
|------|------------------------|---------------|
| 入力 | コマンドライン (1クエリ) | JSON (複数クエリ) |
| 出力 | TXT | JSON + Markdown |
| 統計 | 平均値のみ | 平均・標準偏差・最小・最大 |
| バッチ処理 | ❌ | ✅ |
| 改善率計算 | ❌ | ✅ |

## ライセンス

このプロジェクトと同じライセンスに従います。
