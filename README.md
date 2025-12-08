# Dynamic Seismic Ontology System

SWRLルールに基づく自動分類とナレッジグラフ構築、およびLLMを用いた評価システム。

## 概要

このプロジェクトは、建築物の耐震性や属性に関するオントロジーを管理し、SWRLルールを用いて自動的にクラス分類（例：高層建築物、耐震基準適合など）を行うシステムです。また、LLM（Gemini）を用いたRAG（Retrieval-Augmented Generation）による質問応答機能や、Few-shot vs Zero-shotの性能評価機能も備えています。

## ディレクトリ構成

```
.
├── main.py                 # アプリケーションのエントリーポイント (Streamlit)
├── modules/                # 主要なロジックを含むモジュール群
│   ├── evaluation.py       # LLM応答の評価ロジック (定量的・定性的評価)
│   ├── llm_client.py       # Gemini APIクライアント
│   ├── ontology_manager.py # オントロジー操作 (ロード, 保存, 推論, 可視化)
│   ├── rag_engine.py       # RAG (検索・コンテキスト生成) エンジン
│   └── ui_components.py    # Streamlit UIコンポーネント
├── Ontology.txt            # Few-shotプロンプト用のオントロジーテキスト
├── evaluation_config.json  # 評価実行時の設定ファイル
├── requirements.txt        # 依存ライブラリ一覧
└── swrl_examples.md        # SWRLルールの記述例
```

## 各ファイルの役割

### 1. `main.py`
Streamlitアプリケーションのメインファイルです。
- アプリケーション全体のレイアウトとタブ構成（建築物登録、推論・検証、可視化）を定義します。
- `OntologyManager` を初期化し、セッション状態で管理します。

### 2. `modules/ontology_manager.py`
オントロジー操作の中核となるクラス `OntologyManager` を提供します。
- **ロード/保存**: OWLファイルの読み込みと保存。
- **建築物追加**: ユーザー入力に基づいて建築物インスタンスをオントロジーに追加します。
- **推論**: `owlready2` と Pellet Reasoner を使用してSWRLルールを適用し、クラス分類を推論します。
- **可視化**: `graphviz` を用いて建築物とその属性、所属クラスをグラフとして可視化します。

### 3. `modules/llm_client.py`
Google Gemini APIを利用するためのクライアントクラス `LLMClient` を提供します。
- システムプロンプトの設定や、コンテキストに基づいた回答生成を行います。

### 4. `modules/rag_engine.py`
RAG（検索拡張生成）のロジックを担当する `HybridRetriever` クラスを提供します。
- **検索**: オントロジー内の個体に対するキーワード検索（将来的にベクトル検索も統合可能）。
- **コンテキスト整形**: 検索結果をLLMへの入力用プロンプト形式に整形します。

### 5. `modules/evaluation.py`
LLMの応答精度を評価するための `GeminiEvaluator` クラスを提供します。
- **比較評価**: Few-shot（オントロジー誘導）とZero-shotの応答を比較します。
- **指標**: ROUGE-L, BERTScore, 編集距離などの定量的指標に加え、NLI（含意関係）を用いた事実性評価もサポートします。
- **ログ**: 評価結果をSQLiteデータベースに保存します。

### 6. `modules/ui_components.py`
StreamlitのUIパーツを関数化したモジュールです。
- **Ontology Manager Tab**: インスタンスの閲覧・追加・編集。
- **Reasoning Engine Tab**: 推論実行とSWRLルールの編集。
- **AI Assistant Tab**: チャット形式でのQ&Aインターフェース。
- **Evaluation Tab**: 評価実験の設定と実行、結果の可視化。

## セットアップと実行

1. **依存ライブラリのインストール**
   ```bash
   pip install -r requirements.txt
   ```

2. **APIキーの設定**
   StreamlitのサイドバーでGemini APIキーを入力するか、`.streamlit/secrets.toml` に設定してください。

3. **アプリケーションの起動**
   ```bash
   streamlit run main.py
   ```

## 評価機能の使い方

「Evaluation」タブから、モデル（Gemini-1.5-flash等）や試行回数を選択し、テストクエリを入力して「Run Evaluation」を実行することで、プロンプトエンジニアリングの効果を定量・定性的に分析できます。
