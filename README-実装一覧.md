# RAG CLI 実装一覧

## 機能概要

PDF・CSVファイルを取り込み、チャンク分割・ベクトル検索で関連情報を取得し、Cross-Encoderでリランキング後、ローカルLLMで回答を生成するCLI型RAGシステム。LangChain/LangGraph ベースのパイプラインで構成。DI Container（`AppContainer`）による依存性注入でインフラ層を統合管理。

| 項目 | 内容 |
|------|------|
| 推論方式 | CPU推論（外部API不要） |
| 対応ファイル | PDF, CSV |
| ベクトルDB | PostgreSQL 16 + pgvector（langchain-postgres経由） |
| 埋め込みモデル | all-MiniLM-L6-v2（384次元、langchain-huggingface経由） |
| リランカー | ms-marco-MiniLM-L-6-v2（CrossEncoderReranker） |
| LLM | Llama-2-7B Q4_K_M（GGUF形式、langchain-community LlamaCpp経由） |
| オーケストレーション | LangGraph（StateGraph） |
| DI Container | AppContainer（lazy property + シングルトンキャッシュ） |
| 検索戦略 | TwoStageRetrieval（ベクトル検索 → Cross-Encoder リランキング） |
| Protocol | interfaces.py で型安全なインターフェース定義 |
| 設定管理 | setting.yaml + 環境変数オーバーライド |
| 実行環境 | Docker（Python 3.11-slim） |

---

## アーキテクチャ

```
data/pdf/  ──┐
data/csv/  ──┤
             ▼
         ingest.py ── chunking.py ──→ container.vectorstore ──→ PGVector (langchain-postgres)
                     (PDF: split_by_structure)                   (documents コレクション)
                     (CSV: 1行=1ドキュメント)                    source, chunk_index 付き
                                                                        │
                                                                 ベクトル検索 (SEARCH_K=20)
                                                                  + スコア閾値フィルタ (SCORE_THRESHOLD=0.5)
                                                                        │
                                                                 reranker (CrossEncoderReranker, RERANK_TOP_K=5)
                                                                        │
         ask.py ◄───────────────────────────────────────────────────────┘
           │
           ├── container.py   (DI Container — 全インフラ依存を一元管理)
           │     ├── embeddings          (HuggingFaceEmbeddings、lazy property)
           │     ├── vectorstore         (PGVector、lazy property)
           │     ├── reranker            (CrossEncoderReranker、lazy property)
           │     ├── llm                 (LlamaCpp、lazy property)
           │     ├── prompt_builder      (プロンプト構築関数)
           │     └── retrieval_strategy  (TwoStageRetrieval、lazy property)
           │
           ├── interfaces.py  (Protocol 定義 — 型安全なインターフェース)
           │     ├── VectorStoreProtocol
           │     ├── RerankerProtocol
           │     ├── LLMProtocol
           │     ├── RetrievalStrategyProtocol
           │     └── PromptBuilder (型エイリアス)
           │
           ├── retrieval.py   (検索戦略)
           │     └── TwoStageRetrieval (vectorstore → reranker の2段階検索)
           │
           └── graph.py       (LangGraph パイプライン — 2ノード構成)
                 ├── retrieve       (TwoStageRetrieval で検索+リランキング)
                 └── generate_node  (回答生成ノード — container.llm 使用)
                       │
                       ▼
                  Llama-2 (CPU)
```

**DI Container パターン:**

- `AppContainer` が全インフラ依存（embeddings, vectorstore, reranker, llm, prompt_builder, retrieval_strategy）を lazy property で管理
- インフラモジュール（`db.py`, `embeddings.py`, `llm.py`, `reranker.py`）はステートレスなファクトリ関数（`create_*`）のみ提供
- `get_container()` がアプリケーション全体で唯一のシングルトンキャッシュ
- テスト時はコンストラクタ引数でモックを注入可能
- `interfaces.py` の Protocol クラスにより型安全な依存注入を実現

**TwoStageRetrieval パターン:**

- `TwoStageRetrieval` が2段階検索（ベクトル検索 → Cross-Encoder リランキング）をカプセル化
- `container.retrieval_strategy` として lazy property で管理
- graph.py の `retrieve` ノードが `container.retrieval_strategy.retrieve()` を呼び出し、検索+リランキングを一括実行

**データフロー:**

1. `ingest.py` が既存コレクションを `delete_collection()` で削除し、vectorstore を再生成（冪等な再取り込み）
2. PDF/CSV を読み込み、テキストとソースメタデータ（`file:p1`, `file:r1`）を抽出
3. PDF は `split_by_structure` で段落分割、CSV は1行=1ドキュメント（カテゴリ列除外）
4. `langchain_core.documents.Document` にメタデータ（source, chunk_index）付きで格納
5. `container.vectorstore.add_documents()` で PostgreSQL に一括格納
6. `ask.py` の `main()` は `get_container()` 経由で container を取得し、`get_graph(container=...)` で LangGraph パイプラインを構築
7. LangGraph の `retrieve` ノードが `container.retrieval_strategy.retrieve()` を呼び出し、`similarity_search_with_score` でベクトル検索（SEARCH_K=20件）→ スコア閾値フィルタ（SCORE_THRESHOLD=0.5）→ Cross-Encoder リランキング（RERANK_TOP_K=5件）を実行
8. `generate` ノードが `container.prompt_builder` で日本語プロンプトを構築し、`container.llm.invoke()` で回答を生成。検索結果が空の場合は LLM を呼び出さず「該当する情報が見つかりませんでした。」を返却
9. 回答と重複排除されたソース情報を出力

---

## フォルダ・ファイル一覧

```
llm-rag-cli/
├── src/                           # アプリケーション本体
│   ├── rag/                       # RAG コアパッケージ
│   │   ├── core/                  # コア層（設定・インターフェース・DI）
│   │   │   ├── __init__.py
│   │   │   ├── config.py          # 環境変数・定数管理・DB接続文字列（setting.yaml ベース）
│   │   │   ├── interfaces.py      # Protocol 定義（VectorStore, Reranker, LLM, RetrievalStrategy）
│   │   │   └── container.py       # DI Container（AppContainer + RagSettings）
│   │   ├── infra/                 # インフラ層（DB接続）
│   │   │   ├── __init__.py
│   │   │   └── db.py              # PGVector vectorstore ファクトリ
│   │   ├── components/            # コンポーネント層（モデル・リランカー・プロンプト）
│   │   │   ├── __init__.py
│   │   │   ├── embeddings.py      # テキスト埋め込みファクトリ（langchain-huggingface）
│   │   │   ├── llm.py             # LLM推論ファクトリ（langchain-community LlamaCpp）
│   │   │   ├── reranker.py        # CrossEncoderReranker ファクトリ
│   │   │   └── prompting.py       # プロンプト構築（日本語テンプレート）
│   │   ├── data/                  # データ層（取り込み・チャンク分割）
│   │   │   ├── __init__.py
│   │   │   ├── chunking.py        # テキストチャンク分割（固定サイズ・構造ベース）
│   │   │   └── ingest.py          # ドキュメント取り込みパイプライン
│   │   ├── pipeline/              # パイプライン層（検索・グラフ）
│   │   │   ├── __init__.py
│   │   │   ├── graph.py           # LangGraph RAGパイプライン（2ノード StateGraph + ファクトリノード）
│   │   │   └── retrieval.py       # 検索戦略（TwoStageRetrieval: ベクトル検索 → リランキング）
│   │   └── evaluation/            # 評価層（メトリクス・評価パイプライン）
│   │       ├── __init__.py
│   │       ├── metrics.py         # 評価メトリクス（retrieval@k, faithfulness, exact_match, latency）
│   │       └── evaluate.py        # 評価パイプライン実行（graph.invoke() ベース）
│   └── cli/                       # CLI エントリポイント
│       ├── __init__.py
│       └── ask.py                 # 質問応答CLI エントリポイント
├── tests/                         # テストスイート（250テスト: 単体240 + DB統合7 + heavy3）
│   ├── __init__.py                # パッケージ初期化
│   ├── conftest.py                # 共有フィクスチャ（reset_container + DB統合用fixture + heavy用real_vectorstore）
│   ├── test_config.py             # config.py のテスト（25件）
│   ├── test_container.py          # container.py のテスト（21件）
│   ├── test_interfaces.py         # interfaces.py のテスト（4件）
│   ├── test_retrieval.py          # retrieval.py のテスト（8件）
│   ├── test_db.py                 # db.py のテスト（6件）
│   ├── test_db_integration.py     # DB統合テスト（7件、@pytest.mark.integration）
│   ├── test_embeddings.py         # embeddings.py のテスト（2件）
│   ├── test_embeddings_integration.py  # 実Embeddings統合テスト（3件、@pytest.mark.heavy）
│   ├── test_llm.py                # llm.py のテスト（2件）
│   ├── test_chunking.py           # chunking.py のテスト（32件）
│   ├── test_reranker.py           # reranker.py のテスト（3件）
│   ├── test_prompting.py          # prompting.py のテスト（6件）
│   ├── test_ingest.py             # ingest.py のテスト（23件）
│   ├── test_ask.py                # ask.py のテスト（8件）
│   ├── test_graph.py              # graph.py のテスト（19件）
│   ├── test_metrics.py            # metrics.py のテスト（38件）
│   └── test_evaluate.py           # evaluate.py のテスト（43件）
├── data/                          # 入力データ配置先
│   ├── pdf/                       # PDF ファイル格納
│   │   ├── company_overview.pdf       # 会社概要
│   │   └── rag_technical_guide.pdf    # RAG技術ガイド
│   ├── csv/                       # CSV ファイル格納
│   │   ├── faq.csv                    # FAQ データ
│   │   └── products.csv               # 製品データ
│   └── eval_questions.json        # 評価用質問データ（13問）
├── models/                        # LLM モデル配置先
│   └── (llama-2-7b.Q4_K_M.gguf)     # 手動配置が必要
├── docs/
│   ├── 設計書.md                   # 設計ドキュメント
│   ├── リファクタリング.md         # リファクタリング記録
│   └── 実Embeddings統合テストの最小.md  # 実Embeddings統合テストガイド
├── env/                           # 環境設定
│   ├── config/
│   │   └── setting.yaml           # アプリケーション設定ファイル（DB接続・モデル・パラメータ）
│   └── secret/                    # シークレット配置先（空）
├── Dockerfile                     # Python 3.11-slim ベースイメージ
├── docker-compose.yml             # app + PostgreSQL 16 (pgvector) 構成（PYTHONPATH: /app/src）
├── requirements.txt               # 本番依存パッケージ
├── requirements-dev.txt           # テスト依存パッケージ（pytest, pytest-mock）
├── pytest.ini                     # pytest 設定（pythonpath = src）
├── Makefile                       # 開発用コマンド集
├── doppler.yaml                   # Doppler シークレット管理設定
├── .gitignore                     # Git 除外設定
├── CLAUDE.md                      # Claude Code 向けガイド
├── README.md                      # プロジェクト README
└── README-実装一覧.md             # 本ファイル（実装一覧）
```

### ファイル詳細

| ファイル | 役割 |
|----------|------|
| `src/rag/core/config.py` | `_load_settings()` で `env/config/setting.yaml` を読み込み。`get_db_config()` で環境変数（優先）またはYAMLからDB接続情報を取得。`get_connection_string()` で SQLAlchemy 形式の接続文字列を生成。`CONNECTION_STRING`, `COLLECTION_NAME`, `EMBED_MODEL`, `LLM_MODEL_PATH`, `LLM_N_CTX`, `LLM_MAX_TOKENS`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `RERANKER_MODEL`, `SEARCH_K`, `RERANK_TOP_K`, `SCORE_THRESHOLD` 定数を定義 |
| `src/rag/core/interfaces.py` | `VectorStoreProtocol`（`similarity_search_with_score()` メソッド）、`RerankerProtocol`（`compress_documents()` メソッド）、`LLMProtocol`（`invoke()` メソッド）、`RetrievalStrategyProtocol`（`retrieve()` メソッド）、`PromptBuilder` 型エイリアスを定義。DI Container の型安全性を保証 |
| `src/rag/core/container.py` | `RagSettings` データクラスで RAG パラメータ管理。`AppContainer` が全インフラ依存を lazy property でキャッシュ（embeddings, vectorstore, reranker, llm, prompt_builder, retrieval_strategy）。`get_container()` でシングルトン返却。テスト時はコンストラクタ引数でモック注入可能 |
| `src/rag/pipeline/retrieval.py` | `TwoStageRetrieval` frozen dataclass。`vectorstore.similarity_search_with_score()` でベクトル検索後、`score_threshold` でフィルタリングし、`reranker.compress_documents()` でリランキングする2段階検索をカプセル化。`retrieve(query)` で検索結果を返却 |
| `src/rag/infra/db.py` | `create_vectorstore(embeddings)` で `langchain_postgres.PGVector` を生成するファクトリ関数。`CONNECTION_STRING`, `COLLECTION_NAME`, `use_jsonb=True` |
| `src/rag/components/embeddings.py` | `create_embeddings()` で `HuggingFaceEmbeddings` を生成するファクトリ関数 |
| `src/rag/components/llm.py` | `create_llm()` で `LlamaCpp` を生成するファクトリ関数（n_ctx, max_tokens, stop tokens は config 参照） |
| `src/rag/data/chunking.py` | `split_text()` で固定サイズ・単語境界チャンク分割（overlap付き）、`split_by_structure()` で段落ベースチャンク分割 |
| `src/rag/components/reranker.py` | `CrossEncoderReranker` クラス（`compress_documents()` メソッドで RerankerProtocol を実装）。`create_reranker()` ファクトリ関数で生成 |
| `src/rag/components/prompting.py` | `build_prompt(query, contexts)` で日本語プロンプトテンプレートを構築 |
| `src/rag/data/ingest.py` | `load_pdfs()` でPDFのページテキスト抽出（ソースメタデータ付き）、`load_csvs()` でCSVの行をカテゴリ列を除外したテキストに変換、`main()` で既存コレクションを削除後、PDFは `split_by_structure` で分割、CSVは1行=1ドキュメントとして `Document` 化し `add_documents()` で格納 |
| `src/cli/ask.py` | `main()` で `get_container()` → `get_graph(container=...)` で LangGraph パイプラインを構築し、`graph.invoke()` で回答生成、回答とソース情報を出力 |
| `src/rag/pipeline/graph.py` | `RAGState` dataclass でパイプライン状態を定義。`create_retrieve` / `create_generate` ファクトリ関数で container 依存のノードを生成。`build_rag_graph(container=)` で 2ノード StateGraph（retrieve → generate → END）をコンパイル。`get_graph(container=)` でグラフ返却 |
| `src/rag/evaluation/metrics.py` | `retrieval_at_k()` で検索ヒット判定、`faithfulness()` でキーワード一致率算出、`exact_match()` で全キーワード一致判定、`measure_latency()` で関数実行時間計測 |
| `src/rag/evaluation/evaluate.py` | `load_questions()` で評価データ読み込み、`evaluate_single(query, expected_source, expected_keywords, graph)` で `graph.invoke()` 経由の個別評価、`run_evaluation(questions, graph)` で全問評価、`print_report()` でレポート出力 |
| `tests/conftest.py` | `reset_container` (autouse: AppContainer シングルトンリセット)、`test_embeddings` (FakeEmbeddings 384次元)、`test_vectorstore` (実PGVector test_documents コレクション)、`real_vectorstore` (実HuggingFaceEmbeddings + 実PGVector、heavy用) |
| `tests/test_db_integration.py` | DB統合テスト（`@pytest.mark.integration`）。PGVector接続確認、ドキュメント追加・検索・削除、メタデータ保持、kパラメータ制御を実DBで検証 |
| `tests/test_embeddings_integration.py` | 実Embeddings統合テスト（`@pytest.mark.integration` + `@pytest.mark.heavy`）。実HuggingFaceEmbeddingsによるベクトル検索の順序妥当性を検証（cosine値は検証しない） |
| `data/eval_questions.json` | 評価用質問13問（query, expected_source, expected_keywords） |

---

## API一覧

### src/rag/core/config.py

| 関数/定数 | シグネチャ | 説明 |
|-----------|-----------|------|
| `_load_settings()` | `() -> dict` | `env/config/setting.yaml` を読み込み、設定辞書を返却 |
| `get_db_config()` | `() -> dict` | 環境変数（優先）またはYAMLから DB 接続設定を辞書で返却 |
| `get_connection_string()` | `() -> str` | `postgresql+psycopg://` 形式の接続文字列を生成（ポートはYAMLから取得） |
| `CONNECTION_STRING` | `str` | `get_connection_string()` の評価結果 |
| `COLLECTION_NAME` | `str` | YAMLの `collection_name`（デフォルト `"documents"`） |
| `EMBED_MODEL` | `str` | YAMLの `models.embed_model`（`"sentence-transformers/all-MiniLM-L6-v2"`） |
| `LLM_MODEL_PATH` | `str` | YAMLの `models.llm_model_path`（`"./models/llama-2-7b.Q4_K_M.gguf"`） |
| `LLM_N_CTX` | `int` | YAMLの `llm.n_ctx`（デフォルト 2048） |
| `LLM_MAX_TOKENS` | `int` | YAMLの `llm.max_tokens`（デフォルト 300） |
| `CHUNK_SIZE` | `int` | 環境変数 `CHUNK_SIZE` またはYAML（デフォルト 350） |
| `CHUNK_OVERLAP` | `int` | 環境変数 `CHUNK_OVERLAP` またはYAML（デフォルト 80） |
| `RERANKER_MODEL` | `str` | YAMLの `models.reranker_model`（`"cross-encoder/ms-marco-MiniLM-L-6-v2"`） |
| `SEARCH_K` | `int` | 環境変数 `SEARCH_K` またはYAML（デフォルト 20） |
| `RERANK_TOP_K` | `int` | 環境変数 `RERANK_TOP_K` またはYAML（デフォルト 5） |
| `SCORE_THRESHOLD` | `float` | 環境変数 `SCORE_THRESHOLD` またはYAML（デフォルト 0.5） |

### src/rag/core/interfaces.py

| クラス/型 | シグネチャ | 説明 |
|-----------|-----------|------|
| `VectorStoreProtocol` | `Protocol` | `similarity_search_with_score(query: str, k: int) -> list` メソッドを定義 |
| `RerankerProtocol` | `Protocol` | `compress_documents(documents: List[Document], query: str) -> List[Document]` メソッドを定義 |
| `LLMProtocol` | `Protocol` | `invoke(prompt: str) -> str` メソッドを定義 |
| `RetrievalStrategyProtocol` | `Protocol` | `retrieve(query: str) -> List[Document]` メソッドを定義 |
| `PromptBuilder` | `Callable[[str, List[str]], str]` | プロンプト構築関数の型エイリアス |

### src/rag/core/container.py

| 関数/クラス | シグネチャ | 説明 |
|-------------|-----------|------|
| `RagSettings` | `@dataclass(frozen=True)` | RAG パラメータ管理（`search_k`, `rerank_top_k`, `score_threshold`） |
| `AppContainer` | `class` | DI Container。コンストラクタで `settings`, `embeddings`, `vectorstore`, `reranker`, `llm`, `prompt_builder`, `retrieval_strategy` を受け取り（すべてオプション）。未指定の依存は lazy property で遅延生成・キャッシュ |
| `AppContainer.embeddings` | `@property -> HuggingFaceEmbeddings` | `create_embeddings()` で遅延生成 |
| `AppContainer.vectorstore` | `@property -> VectorStoreProtocol` | `create_vectorstore(self.embeddings)` で遅延生成 |
| `AppContainer.reranker` | `@property -> RerankerProtocol` | `create_reranker()` で遅延生成 |
| `AppContainer.llm` | `@property -> LLMProtocol` | `create_llm()` で遅延生成 |
| `AppContainer.prompt_builder` | `@property -> PromptBuilder` | `build_prompt` で遅延設定 |
| `AppContainer.retrieval_strategy` | `@property -> RetrievalStrategyProtocol` | `TwoStageRetrieval(vectorstore, reranker, search_k, rerank_top_k, score_threshold)` で遅延生成 |
| `get_container()` | `() -> AppContainer` | シングルトンの `AppContainer` を返却 |

### src/rag/pipeline/retrieval.py

| クラス | シグネチャ | 説明 |
|--------|-----------|------|
| `TwoStageRetrieval` | `@dataclass(frozen=True)` | `vectorstore`, `reranker`, `search_k`, `rerank_top_k`, `score_threshold` を保持。`retrieve(query)` で `similarity_search_with_score` → スコア閾値フィルタ → リランキングの2段階検索を実行し `List[Document]` を返却 |

### src/rag/infra/db.py

| 関数 | シグネチャ | 説明 |
|------|-----------|------|
| `create_vectorstore(embeddings)` | `(Embeddings) -> PGVector` | `langchain_postgres.PGVector` を生成して返却。`CONNECTION_STRING`, `COLLECTION_NAME`, `use_jsonb=True` |

### src/rag/components/embeddings.py

| 関数 | シグネチャ | 説明 |
|------|-----------|------|
| `create_embeddings()` | `() -> HuggingFaceEmbeddings` | `HuggingFaceEmbeddings` を生成して返却（`EMBED_MODEL` 使用） |

### src/rag/components/llm.py

| 関数 | シグネチャ | 説明 |
|------|-----------|------|
| `create_llm()` | `() -> LlamaCpp` | `LlamaCpp` を生成して返却（`LLM_MODEL_PATH`, `LLM_N_CTX`, `LLM_MAX_TOKENS`, `stop=["質問:", "\\n\\n"]`, `verbose=False`） |

### src/rag/data/chunking.py

| 関数 | シグネチャ | 説明 |
|------|-----------|------|
| `split_text(text, chunk_size=500, overlap=100)` | `(str, int, int) -> list[str]` | テキストを固定サイズで分割（単語境界保持、overlap付き）。空文字は空リスト、chunk_size以下はそのまま返却 |
| `split_by_structure(text, chunk_size=None, overlap=100)` | `(str, int\|None, int) -> list[str]` | 段落（`\n\n`）で分割。`chunk_size` 指定時は長い段落を `split_text` でさらに分割 |

### src/rag/components/reranker.py

| 関数 | シグネチャ | 説明 |
|------|-----------|------|
| `CrossEncoderReranker` | `class` | `HuggingFaceCrossEncoder` を使った reranker。`compress_documents(documents, query)` でドキュメントをスコアリングし上位 `top_n` 件を返却。`RerankerProtocol` を満たす |
| `create_reranker(top_n)` | `(int) -> CrossEncoderReranker` | `HuggingFaceCrossEncoder` + `CrossEncoderReranker` を生成して返却（`RERANKER_MODEL`, `RERANK_TOP_K` 使用） |

### src/rag/components/prompting.py

| 関数 | シグネチャ | 説明 |
|------|-----------|------|
| `build_prompt(query, contexts)` | `(str, list[str]) -> str` | 日本語プロンプトテンプレートを構築。コンテキストと質問を埋め込み、回答マーカー付きで返却 |

### src/rag/data/ingest.py

| 関数 | シグネチャ | 説明 |
|------|-----------|------|
| `load_pdfs()` | `() -> list[tuple[str, str]]` | `data/pdf/` 内の全 PDF から各ページのテキストとソース（`filename:pN`）をタプルで返却 |
| `load_csvs()` | `() -> list[tuple[str, str]]` | `data/csv/` 内の全 CSV から各行をカテゴリ列を除外したテキストとソース（`filename:rN`）でタプル返却 |
| `main()` | `() -> None` | `get_container()` で container 取得 → `delete_collection()` で既存データ削除 → vectorstore 再生成 → PDF/CSV読み込み → PDFは `split_by_structure` で分割、CSVは1行=1ドキュメント → `Document` 化（source, chunk_index メタデータ付き） → `container.vectorstore.add_documents()` で格納 |

### src/cli/ask.py

| 関数 | シグネチャ | 説明 |
|------|-----------|------|
| `main()` | `() -> None` | `sys.argv[1]` から質問を取得、`get_container()` → `get_graph(container=...)` で LangGraph パイプラインを構築、`graph.invoke()` で回答生成、回答とソース情報を出力 |

### src/rag/pipeline/graph.py

| 関数/クラス | シグネチャ | 説明 |
|-------------|-----------|------|
| `RAGState` | `@dataclass` | パイプライン状態（query, reranked_documents, contexts, prompt, answer, sources） |
| `create_retrieve(container)` | `(AppContainer) -> Callable` | container の `retrieval_strategy.retrieve()` を使用する retrieve ノード関数を生成。ベクトル検索+リランキングを一括実行し `reranked_documents` を返却 |
| `create_generate(container)` | `(AppContainer) -> Callable` | container の llm, prompt_builder を使用する generate ノード関数を生成 |
| `build_rag_graph(*, container=None)` | `(AppContainer\|None) -> CompiledGraph` | retrieve → generate → END の 2ノード StateGraph を構築・コンパイル。container 未指定時は `get_container()` から取得 |
| `get_graph(*, container=None)` | `(AppContainer\|None) -> CompiledGraph` | container 指定時は毎回新規構築、未指定時はシングルトン返却 |

### src/rag/evaluation/metrics.py

| 関数 | シグネチャ | 説明 |
|------|-----------|------|
| `retrieval_at_k(sources, expected_source)` | `(list[str], str) -> bool` | ソース文字列リスト内に期待ソースが含まれるか判定 |
| `faithfulness(answer, expected_keywords)` | `(str, list[str]) -> float` | 回答中のキーワード出現率を 0.0〜1.0 で返却。キーワード空リストは 1.0 |
| `exact_match(answer, expected_keywords)` | `(str, list[str]) -> bool` | 全キーワードが回答中に含まれるか判定。キーワード空リストは True |
| `measure_latency(func)` | `(Callable) -> tuple[Any, float]` | 関数を実行し、結果と経過時間（秒）のタプルを返却 |

### src/rag/evaluation/evaluate.py

| 関数 | シグネチャ | 説明 |
|------|-----------|------|
| `load_questions(path="data/eval_questions.json")` | `(str) -> list[dict]` | 評価用質問データをJSONから読み込み |
| `evaluate_single(query, expected_source, expected_keywords, graph)` | `(str, str, list[str], CompiledGraph) -> dict` | `graph.invoke()` で1問を評価し、retrieval_hit, faithfulness, exact_match, latency, answer を含む辞書を返却 |
| `run_evaluation(questions, graph)` | `(list[dict], CompiledGraph) -> list[dict]` | 全質問を順次評価し結果リストを返却 |
| `print_report(results, config)` | `(list[dict], dict) -> None` | 評価レポートを出力（Retrieval@k, Faithfulness, Exact Match, Latency, Re-rank状態） |
| `main()` | `() -> None` | 質問読み込み → `get_container()` → `get_graph(container=...)` → 全問評価 → レポート出力 |

---

## コマンド一覧

### Make コマンド

| コマンド | 説明 |
|----------|------|
| `make build` | Docker イメージをビルド |
| `make up` | Docker コンテナをバックグラウンド起動 |
| `make down` | Docker コンテナを停止・削除 |
| `make shell` | app コンテナの bash に接続 |
| `make test` | 全テスト実行 (`pytest tests/ -v`、単体+統合+heavy) |
| `make test-unit` | 単体テストのみ実行（DB不要、`-m "not integration and not heavy"`） |
| `make test-integration` | DB統合テストのみ実行（PostgreSQL必要、heavy除外） |
| `make test-heavy` | 実Embeddingsテスト実行（PostgreSQL+モデルDL必要、`-m heavy`） |
| `make lint` | 全モジュールの構文チェック (`py_compile`、src/rag/ + src/cli/ 配下) |
| `make ingest` | ドキュメント取り込み実行 (`python -m rag.data.ingest`) |
| `make ask Q="質問文"` | RAG に質問して回答を取得 (`python -m cli.ask`) |
| `make evaluate` | 評価パイプライン実行 (`python -m rag.evaluation.evaluate`) |

### Docker 直接実行

```bash
docker compose up -d                        # コンテナ起動
docker compose exec app bash                # コンテナに入る
python -m rag.data.ingest                   # ドキュメント取り込み
python -m cli.ask "制度の目的は？"           # 質問
python -m pytest tests/ -v                  # テスト実行
python -m rag.evaluation.evaluate           # 評価パイプライン
```

### 環境変数

| 変数名 | デフォルト値 | 説明 |
|--------|-------------|------|
| `DB_HOST` | `localhost` | PostgreSQL ホスト（Docker 内は `db`） |
| `DB_USER` | `rag` | データベースユーザー |
| `DB_PASSWORD` | `rag` | データベースパスワード |
| `DB_NAME` | `rag` | データベース名 |
| `CHUNK_SIZE` | `350` | チャンクサイズ（文字数） |
| `CHUNK_OVERLAP` | `80` | チャンク間オーバーラップ（文字数） |
| `SEARCH_K` | `20` | ベクトル検索の取得件数 |
| `RERANK_TOP_K` | `5` | リランキング後の上位件数 |
| `SCORE_THRESHOLD` | `0.5` | ベクトル検索のスコア閾値（コサイン距離、低いほど類似） |
| `PYTHONPATH` | `/app/src` | Python モジュール検索パス（Docker 内、docker-compose.yml で設定） |

※ デフォルト値は `env/config/setting.yaml` で定義。環境変数が設定されている場合は環境変数が優先される。

---

## DBスキーマ

langchain-postgres (`PGVector`) が自動管理するスキーマ:

```
PGVector(
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    collection_name="documents",
    connection="postgresql+psycopg://rag:rag@db:5432/rag",
    use_jsonb=True,
)
```

- `langchain_pg_collection` テーブル: コレクション管理
- `langchain_pg_embedding` テーブル: ドキュメント・埋め込み・メタデータ（JSONB）格納
- メタデータに `source`, `chunk_index` を格納

---

## 依存パッケージ

### 本番 (requirements.txt)

| パッケージ | 用途 |
|-----------|------|
| `psycopg2-binary` | PostgreSQL ドライバ（レガシー互換） |
| `psycopg[binary]>=3.1.0` | PostgreSQL ドライバ（psycopg3、langchain-postgres用） |
| `sentence-transformers` | 埋め込みモデル基盤 |
| `pypdf` | PDF テキスト抽出 |
| `pandas` | CSV データ処理 |
| `numpy` | 数値計算 |
| `tqdm` | プログレスバー |
| `pyyaml` | YAML 設定ファイル読み込み |
| `llama-cpp-python` | LLM 推論エンジン基盤 |
| `langchain>=0.3.0` | LangChain コアフレームワーク |
| `langchain-core>=0.3.0` | LangChain コア抽象 |
| `langchain-community>=0.3.0` | LangChain コミュニティ統合（LlamaCpp, CrossEncoder） |
| `langchain-text-splitters>=0.3.0` | テキスト分割（RecursiveCharacterTextSplitter） |
| `langchain-huggingface>=0.1.0` | HuggingFace 埋め込み統合 |
| `langchain-postgres>=0.0.12` | PGVector vectorstore |
| `langgraph>=0.2.0` | LangGraph パイプライン |

### テスト (requirements-dev.txt)

| パッケージ | 用途 |
|-----------|------|
| `pytest` | テストフレームワーク |
| `pytest-mock` | モック支援 |
