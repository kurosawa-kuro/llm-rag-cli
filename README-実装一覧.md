# RAG CLI 実装一覧

## 機能概要

PDF・CSVファイルを取り込み、チャンク分割・ベクトル検索で関連情報を取得し、Cross-Encoderでリランキング後、ローカルLLMで回答を生成するCLI型RAGシステム。LangChain/LangGraph ベースのパイプラインで構成。

| 項目 | 内容 |
|------|------|
| 推論方式 | CPU推論（外部API不要） |
| 対応ファイル | PDF, CSV |
| ベクトルDB | PostgreSQL 16 + pgvector（langchain-postgres経由） |
| 埋め込みモデル | all-MiniLM-L6-v2（384次元、langchain-huggingface経由） |
| リランカー | ms-marco-MiniLM-L-6-v2（CrossEncoderReranker） |
| LLM | Llama-2-7B Q4_K_M（GGUF形式、langchain-community LlamaCpp経由） |
| オーケストレーション | LangGraph（StateGraph） |
| 実行環境 | Docker（Python 3.11-slim） |

---

## アーキテクチャ

```
data/pdf/  ──┐
data/csv/  ──┤
             ▼
         ingest.py ── chunking.py ── embeddings.py ──→ PGVector (langchain-postgres)
                     (PDF: split_by_structure)          (documents コレクション)
                     (CSV: RecursiveCharacterTextSplitter)  source, chunk_index 付き
                                                             │
                                                      ベクトル検索 (SEARCH_K=10)
                                                             │
                                                      reranker.py (CrossEncoderReranker, RERANK_TOP_K=3)
                                                             │
         ask.py ◄────────────────────────────────────────────┘
           │
           ├── graph.py     (LangGraph パイプライン)
           │     ├── retrieve       (ベクトル検索ノード)
           │     ├── rerank_node    (リランキングノード)
           │     └── generate_node  (回答生成ノード)
           │
           ├── embeddings.py  (クエリ埋め込み)
           ├── db.py          (PGVector vectorstore)
           ├── reranker.py    (ContextualCompressionRetriever)
           └── llm.py         (LlamaCpp 回答生成)
                 │
                 ▼
            Llama-2 (CPU)
```

**データフロー:**

1. `ingest.py` が PDF/CSV を読み込み、テキストとソースメタデータ（`file:p1`, `file:r1`）を抽出
2. PDF は `split_by_structure` で段落分割、CSV は `RecursiveCharacterTextSplitter` で分割
3. `langchain_core.documents.Document` にメタデータ（source, chunk_index）付きで格納
4. `PGVector.add_documents()` で PostgreSQL に一括格納
5. `ask.py` の `main()` は `graph.py` の LangGraph パイプラインを呼び出し
6. LangGraph の `retrieve` ノードが vectorstore retriever で上位 SEARCH_K=10 件を取得
7. `rerank_node` が CrossEncoderReranker で RERANK_TOP_K=3 件に絞り込み
8. `generate_node` が日本語プロンプトを構築し、LlamaCpp で回答を生成
9. 回答とソース情報を出力

---

## フォルダ・ファイル一覧

```
llm-rag-cli/
├── app/                        # アプリケーション本体
│   ├── __init__.py             # パッケージ初期化
│   ├── config.py               # 環境変数・定数管理・DB接続文字列
│   ├── db.py                   # PGVector vectorstore 管理
│   ├── embeddings.py           # テキスト埋め込み（langchain-huggingface）
│   ├── llm.py                  # LLM推論（langchain-community LlamaCpp）
│   ├── chunking.py             # テキストチャンク分割（固定サイズ・構造ベース）
│   ├── reranker.py             # CrossEncoderReranker + ContextualCompressionRetriever
│   ├── ingest.py               # ドキュメント取り込みパイプライン
│   ├── ask.py                  # 質問応答CLI エントリポイント
│   ├── graph.py                # LangGraph RAGパイプライン（StateGraph）
│   ├── metrics.py              # 評価メトリクス（retrieval@k, faithfulness, exact_match, latency）
│   └── evaluate.py             # 評価パイプライン実行
├── tests/                      # テストスイート（250テスト）
│   ├── __init__.py             # パッケージ初期化
│   ├── conftest.py             # 共有フィクスチャ（mock DB, fake embeddings, mock LLM, mock vectorstore, mock documents）
│   ├── test_config.py          # config.py のテスト（22件）
│   ├── test_db.py              # db.py のテスト（9件）
│   ├── test_embeddings.py      # embeddings.py のテスト（11件）
│   ├── test_llm.py             # llm.py のテスト（11件）
│   ├── test_chunking.py        # chunking.py のテスト（32件）
│   ├── test_reranker.py        # reranker.py のテスト（19件）
│   ├── test_ingest.py          # ingest.py のテスト（23件）
│   ├── test_ask.py             # ask.py のテスト（18件）
│   ├── test_graph.py           # graph.py のテスト（23件）
│   ├── test_metrics.py         # metrics.py のテスト（39件）
│   └── test_evaluate.py        # evaluate.py のテスト（43件）
├── data/                       # 入力データ配置先
│   ├── pdf/                    # PDF ファイル格納
│   │   ├── company_overview.pdf    # 会社概要
│   │   └── rag_technical_guide.pdf # RAG技術ガイド
│   ├── csv/                    # CSV ファイル格納
│   │   ├── faq.csv                 # FAQ データ
│   │   └── products.csv            # 製品データ
│   └── eval_questions.json     # 評価用質問データ（13問）
├── models/                     # LLM モデル配置先
│   └── (llama-2-7b.Q4_K_M.gguf)  # 手動配置が必要
├── docs/
│   ├── 設計書.md                # 設計ドキュメント
│   ├── ロードマップ.md          # ロードマップ
│   └── libアップデート.md       # ライブラリアップデート情報
├── env/                        # 環境設定
│   ├── config/
│   │   └── setting.yaml        # アプリケーション設定ファイル
│   └── secret/                 # シークレット配置先（空）
├── Dockerfile                  # Python 3.11-slim ベースイメージ
├── docker-compose.yml          # app + PostgreSQL 16 (pgvector) 構成
├── requirements.txt            # 本番依存パッケージ
├── requirements-dev.txt        # テスト依存パッケージ（pytest, pytest-mock）
├── pytest.ini                  # pytest 設定
├── Makefile                    # 開発用コマンド集
├── doppler.yaml                # Doppler シークレット管理設定
├── .gitignore                  # Git 除外設定
├── CLAUDE.md                   # Claude Code 向けガイド
├── README.md                   # プロジェクト README
└── README-実装一覧.md          # 本ファイル（実装一覧）
```

### ファイル詳細

| ファイル | 役割 |
|----------|------|
| `app/config.py` | `get_db_config()` で環境変数からDB接続情報を取得。`get_connection_string()` で SQLAlchemy 形式の接続文字列を生成。`DB_CONFIG`, `CONNECTION_STRING`, `COLLECTION_NAME`, `EMBED_MODEL`, `LLM_MODEL_PATH`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `RERANKER_MODEL`, `SEARCH_K`, `RERANK_TOP_K` 定数を定義 |
| `app/db.py` | `get_vectorstore()` で `langchain_postgres.PGVector` を遅延初期化して返却（シングルトン）。`init_db()` で vectorstore を初期化 |
| `app/embeddings.py` | `get_embeddings()` で `HuggingFaceEmbeddings` を遅延ロードして返却（シングルトン）。`embed(texts)` でテキストリストを埋め込みベクトルに変換 |
| `app/llm.py` | `get_llm()` で `LlamaCpp` を遅延ロードして返却（n_ctx=2048, max_tokens=300、シングルトン）。`generate(prompt)` で `invoke()` により回答テキストを生成 |
| `app/chunking.py` | `split_text()` で固定サイズ・単語境界チャンク分割（overlap付き）、`split_by_structure()` で段落ベースチャンク分割 |
| `app/reranker.py` | `get_reranker()` で `HuggingFaceCrossEncoder` + `CrossEncoderReranker` を遅延ロード（シングルトン）。`get_compression_retriever()` で `ContextualCompressionRetriever` を構築。`rerank(query, docs, top_k)` で辞書形式の文書をリランキング |
| `app/ingest.py` | `load_pdfs()` でPDFのページテキスト抽出（ソースメタデータ付き）、`load_csvs()` でCSVの行を `key:value` 形式に変換、`main()` でPDFは `split_by_structure`、CSVは `RecursiveCharacterTextSplitter` で分割後、`Document` 化して `PGVector.add_documents()` で格納 |
| `app/ask.py` | `search(query)` で vectorstore retriever → `ContextualCompressionRetriever` でリランキング付き検索。`main()` で `graph.py` の LangGraph パイプラインを使用し回答・ソースを出力 |
| `app/graph.py` | `RAGState` TypedDict でパイプライン状態を定義。`retrieve` / `rerank_node` / `generate_node` の3ノード構成。`build_rag_graph()` で StateGraph をコンパイル。`get_graph()` でシングルトン返却 |
| `app/metrics.py` | `retrieval_at_k()` で検索ヒット判定、`faithfulness()` でキーワード一致率算出、`exact_match()` で全キーワード一致判定、`measure_latency()` で関数実行時間計測 |
| `app/evaluate.py` | `load_questions()` で評価データ読み込み、`evaluate_single()` で個別評価（exact_match 含む）、`run_evaluation()` で全問評価、`print_report()` でレポート出力（Exact Match 含む） |
| `tests/conftest.py` | `mock_db_connection` (conn/cur モック)、`fake_embeddings` (3×384次元ダミー)、`mock_llm_response` (LLM応答モック)、`mock_vectorstore` (PGVector モック)、`mock_documents` (LangChain Document モック) |
| `data/eval_questions.json` | 評価用質問13問（query, expected_source, expected_keywords） |

---

## API一覧

### app/config.py

| 関数/定数 | シグネチャ | 説明 |
|-----------|-----------|------|
| `get_db_config()` | `() -> dict` | 環境変数から DB 接続設定を辞書で返却 |
| `get_connection_string()` | `() -> str` | `postgresql+psycopg://` 形式の接続文字列を生成 |
| `DB_CONFIG` | `dict` | `get_db_config()` の評価結果（モジュール読み込み時に確定） |
| `CONNECTION_STRING` | `str` | `get_connection_string()` の評価結果 |
| `COLLECTION_NAME` | `str` | `"documents"` |
| `EMBED_MODEL` | `str` | `"sentence-transformers/all-MiniLM-L6-v2"` |
| `LLM_MODEL_PATH` | `str` | `"./models/llama-2-7b.Q4_K_M.gguf"` |
| `CHUNK_SIZE` | `int` | チャンクサイズ（環境変数 `CHUNK_SIZE`、デフォルト 500） |
| `CHUNK_OVERLAP` | `int` | チャンク間オーバーラップ（環境変数 `CHUNK_OVERLAP`、デフォルト 100） |
| `RERANKER_MODEL` | `str` | `"cross-encoder/ms-marco-MiniLM-L-6-v2"` |
| `SEARCH_K` | `int` | ベクトル検索の取得件数（環境変数 `SEARCH_K`、デフォルト 10） |
| `RERANK_TOP_K` | `int` | リランキング後の上位件数（環境変数 `RERANK_TOP_K`、デフォルト 3） |

### app/db.py

| 関数 | シグネチャ | 説明 |
|------|-----------|------|
| `get_vectorstore()` | `() -> PGVector` | `langchain_postgres.PGVector` を遅延初期化して返却（シングルトン）。`CONNECTION_STRING`, `COLLECTION_NAME`, `use_jsonb=True` |
| `init_db()` | `() -> None` | `get_vectorstore()` を呼び出して vectorstore を初期化 |

### app/embeddings.py

| 関数 | シグネチャ | 説明 |
|------|-----------|------|
| `get_embeddings()` | `() -> HuggingFaceEmbeddings` | `HuggingFaceEmbeddings` を遅延ロードして返却（シングルトン） |
| `embed(texts)` | `(list[str]) -> list[list[float]]` | テキストリストを `embed_documents()` で埋め込みベクトルに変換 |

### app/llm.py

| 関数 | シグネチャ | 説明 |
|------|-----------|------|
| `get_llm()` | `() -> LlamaCpp` | `LlamaCpp` を遅延ロードして返却（n_ctx=2048, max_tokens=300、シングルトン） |
| `generate(prompt)` | `(str) -> str` | プロンプトを `invoke()` に渡し、回答テキストを返却 |

### app/chunking.py

| 関数 | シグネチャ | 説明 |
|------|-----------|------|
| `split_text(text, chunk_size=500, overlap=100)` | `(str, int, int) -> list[str]` | テキストを固定サイズで分割（単語境界保持、overlap付き）。空文字は空リスト、chunk_size以下はそのまま返却 |
| `split_by_structure(text, chunk_size=None, overlap=100)` | `(str, int\|None, int) -> list[str]` | 段落（`\n\n`）で分割。`chunk_size` 指定時は長い段落を `split_text` でさらに分割 |

### app/reranker.py

| 関数 | シグネチャ | 説明 |
|------|-----------|------|
| `get_reranker()` | `() -> CrossEncoderReranker` | `HuggingFaceCrossEncoder` + `CrossEncoderReranker` を遅延ロードして返却（シングルトン） |
| `get_compression_retriever(base_retriever)` | `(BaseRetriever) -> ContextualCompressionRetriever` | base_retriever に `CrossEncoderReranker` を組み合わせた `ContextualCompressionRetriever` を返却 |
| `rerank(query, docs, top_k=3)` | `(str, list[dict], int) -> list[dict]` | 辞書形式の文書を `Document` に変換し、`CrossEncoderReranker` でスコアリング後、上位 top_k 件を辞書形式で返却。空リストは空リスト返却 |

### app/ingest.py

| 関数 | シグネチャ | 説明 |
|------|-----------|------|
| `load_pdfs()` | `() -> list[tuple[str, str]]` | `data/pdf/` 内の全 PDF から各ページのテキストとソース（`filename:pN`）をタプルで返却 |
| `load_csvs()` | `() -> list[tuple[str, str]]` | `data/csv/` 内の全 CSV から各行を `"key:value"` 形式とソース（`filename:rN`）でタプル返却 |
| `main()` | `() -> None` | DB初期化 → PDF/CSV読み込み → PDFは `split_by_structure`、CSVは `RecursiveCharacterTextSplitter` で分割 → `Document` 化（source, chunk_index メタデータ付き） → `PGVector.add_documents()` で格納 |

### app/ask.py

| 関数 | シグネチャ | 説明 |
|------|-----------|------|
| `search(query)` | `(str) -> list[dict]` | vectorstore の retriever を `ContextualCompressionRetriever` でラップし、リランキング付き検索。各要素は `{"content": str, "source": str}` |
| `main()` | `() -> None` | `sys.argv[1]` から質問を取得、`graph.py` の LangGraph パイプラインで回答生成、回答とソース情報を出力 |

### app/graph.py

| 関数/クラス | シグネチャ | 説明 |
|-------------|-----------|------|
| `RAGState` | `TypedDict` | パイプライン状態（query, documents, reranked_documents, contexts, prompt, answer, sources） |
| `retrieve(state)` | `(RAGState) -> dict` | vectorstore retriever で SEARCH_K 件取得し `{"documents": [...]}` を返却 |
| `rerank_node(state)` | `(RAGState) -> dict` | `CrossEncoderReranker.compress_documents()` で RERANK_TOP_K 件に絞り込み |
| `generate_node(state)` | `(RAGState) -> dict` | コンテキスト・ソース抽出、日本語プロンプト構築、`LlamaCpp.invoke()` で回答生成 |
| `build_rag_graph()` | `() -> CompiledGraph` | retrieve → rerank → generate → END の StateGraph を構築・コンパイル |
| `get_graph()` | `() -> CompiledGraph` | コンパイル済みグラフを遅延ロードして返却（シングルトン） |

### app/metrics.py

| 関数 | シグネチャ | 説明 |
|------|-----------|------|
| `retrieval_at_k(results, expected_source)` | `(list[dict], str) -> bool` | 検索結果リスト内に期待ソースが含まれるか判定 |
| `faithfulness(answer, expected_keywords)` | `(str, list[str]) -> float` | 回答中のキーワード出現率を 0.0〜1.0 で返却。キーワード空リストは 1.0 |
| `exact_match(answer, expected_keywords)` | `(str, list[str]) -> bool` | 全キーワードが回答中に含まれるか判定。キーワード空リストは True |
| `measure_latency(func)` | `(Callable) -> tuple[Any, float]` | 関数を実行し、結果と経過時間（秒）のタプルを返却 |

### app/evaluate.py

| 関数 | シグネチャ | 説明 |
|------|-----------|------|
| `load_questions(path="data/eval_questions.json")` | `(str) -> list[dict]` | 評価用質問データをJSONから読み込み |
| `evaluate_single(query, expected_source, expected_keywords, search_fn, generate_fn)` | `(...) -> dict` | 1問を評価し、retrieval_hit, faithfulness, exact_match, latency, answer を含む辞書を返却 |
| `run_evaluation(questions, search_fn, generate_fn)` | `(list[dict], Callable, Callable) -> list[dict]` | 全質問を順次評価し結果リストを返却 |
| `print_report(results, config)` | `(list[dict], dict) -> None` | 評価レポートを出力（Retrieval@k, Faithfulness, Exact Match, Latency, Re-rank状態） |
| `main()` | `() -> None` | 質問読み込み → 全問評価 → レポート出力のパイプライン |

---

## コマンド一覧

### Make コマンド

| コマンド | 説明 |
|----------|------|
| `make build` | Docker イメージをビルド |
| `make up` | Docker コンテナをバックグラウンド起動 |
| `make down` | Docker コンテナを停止・削除 |
| `make shell` | app コンテナの bash に接続 |
| `make test` | 全テスト実行 (`pytest tests/ -v`) |
| `make lint` | 全モジュールの構文チェック (`py_compile`、graph.py 含む) |
| `make ingest` | ドキュメント取り込み実行 |
| `make ask Q="質問文"` | RAG に質問して回答を取得 |
| `make evaluate` | 評価パイプライン実行 (`python -m app.evaluate`) |

### Docker 直接実行

```bash
docker compose up -d                        # コンテナ起動
docker compose exec app bash                # コンテナに入る
python -m app.ingest                        # ドキュメント取り込み
python -m app.ask "制度の目的は？"           # 質問
python -m pytest tests/ -v                  # テスト実行
python -m app.evaluate                      # 評価パイプライン
```

### 環境変数

| 変数名 | デフォルト値 | 説明 |
|--------|-------------|------|
| `DB_HOST` | `localhost` | PostgreSQL ホスト（Docker 内は `db`） |
| `DB_USER` | `rag` | データベースユーザー |
| `DB_PASSWORD` | `rag` | データベースパスワード |
| `DB_NAME` | `rag` | データベース名 |
| `CHUNK_SIZE` | `500` | チャンクサイズ（文字数） |
| `CHUNK_OVERLAP` | `100` | チャンク間オーバーラップ（文字数） |
| `SEARCH_K` | `10` | ベクトル検索の取得件数 |
| `RERANK_TOP_K` | `3` | リランキング後の上位件数 |

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
