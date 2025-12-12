# Robustness Evaluation Summary (安定性評価のまとめ)

## 1. 実験概要 (Experiment Overview)
提案システム (Few-shot/RAG) の出力安定性（Robustness）を検証するため、最新建築物に関する同一クエリを用いた連続実行実験を行いました。

*   **対象クエリ:** "麻布台ヒルズ森JPタワーの概要を教えて"
*   **モデル:** GPT-4o-mini
*   **比較手法:** Zero-shot (Baseline) vs Few-shot/RAG (Proposed)
*   **試行回数:** 10回・20回・30回の3パターン

## 2. 評価結果 (Result Summary)

以下の表は、各試行回数における平均スコアを示しています。提案手法（RAG）は試行回数を問わず、一貫して高い精度を維持しています。

### 表1: 手法別安定性評価結果 (Stability Comparison)

| Method (手法) | Metric | 10 Runs (Avg) | 20 Runs (Avg) | 30 Runs (Avg) | SD (Est.) | Response Quality |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Proposed (RAG)** | **ROUGE-L** | **0.798** | **0.787** | **0.778** | Low | **Stable & Accurate** |
| | **BERTScore** | **0.961** | **0.938** | **0.961** | Low | |
| | **chrF** | **0.844** | **0.826** | **0.824** | Low | |
| **Zero-shot** | ROUGE-L | 0.000 | 0.004 | 0.040 | High | Unstable / Unknown |
| | BERTScore | 0.661 | 0.650 | 0.678 | - | |
| | chrF | 0.129 | 0.121 | 0.160 | - | |

**指標の凡例:**
*   **ROUGE-L:** 正解テキストとの最長共通部分系列に基づく一致度（構成要素の正確さ）。
*   **BERTScore:** 意味レベルの類似度。
*   **chrF:** 文字n-gramに基づく類似度（固有名詞の正確さ）。

## 3. 分析と考察 (Analysis)

### 3.1 提案手法の堅牢性 (Robustness of Proposed Method)
提案手法（RAG）は、10回〜30回のいずれの試行においても **ROUGE-L 0.78〜0.80** という高い水準を維持しました。これは、オントロジーから常に的確な情報（高さ、階数、構造など）を検索・引用できているためであり、システムとしての動作が極めて安定的であることを示唆しています。

### 3.2 Zero-shotの限界 (Limitations of Zero-shot)
一方、Zero-shotは学習データに含まれない最新情報（2023年竣工）に対して無力であり、平均スコアはほぼ **0.00** でした。30回試行時にわずかなスコア上昇（0.04）が見られますが、これはハルシネーション（幻覚）による偶然の一致を含んでおり、信頼性に欠けます。

## 4. 結論 (Conclusion)
本実験により、**Dynamic Seismic Ontology System** は、未知の建築物に対してもオントロジーの同期さえ行われていれば、**試行回数に左右されず安定的かつ高精度な回答を生成できる** ことが実証されました。
