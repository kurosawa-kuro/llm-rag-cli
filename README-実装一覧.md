# RAG CLI 実装一覧

## 機能概要

PDF・CSVファイルを取り込み、チャンク分割・ベクトル検索で関連情報を取得し、Cross-Encoderでリランキング後、ローカルLLMで回答を生成するCLI型RAGシステム。

| 項目 | 内容 |
|------|------|
| 推論方式 | CPU推論（外部API不要） |
| 対応ファイル | PDF, CSV |
| ベクトルDB | PostgreSQL 16 + pgvector |
| 埋め込みモデル | all-MiniLM-L6-v2（384次元） |
| リランカー | ms-marco-MiniLM-L-6-v2（Cross-Encoder） |
| LLM | Llama-2-7B Q4_K_M（GGUF形式） |
| 実行環境 | Docker（Python 3.11-slim） |

---

## アーキテクチャ

```
data/pdf/  ──┐
data/csv/  ──┤
             ▼
         ingest.py ── chunking.py ── embeddings.py ──→ PostgreSQL + pgvector
                     (PDF: 構造分割)                    (documents テーブル)
                     (CSV: 固定分割)                     source, chunk_index 付き
                                                             │
                                                      ベクトル検索 (SEARCH_K=10)
                                                             │
                                                      reranker.py (RERANK_TOP_K=3)
                                                             │
         ask.py ◄────────────────────────────────────────────┘
           │
           ├── embeddings.py  (クエリ埋め込み)
           ├── db.py          (類似検索)
           ├── reranker.py    (リランキング)
           └── llm.py         (回答生成)
                 │
                 ▼
            Llama-2 (CPU)
```

**データフロー:**

1. `ingest.py` が PDF/CSV を読み込み、テキストとソースメタデータ（`file:p1`, `file:r1`）を抽出
2. `chunking.py` がテキストをチャンク分割（PDFは `split_by_structure` で段落分割、CSVは `split_text` で固定サイズ分割）
3. `embeddings.py` がチャンクを384次元ベクトルに変換
4. ベクトル・テキスト・ソース・チャンクインデックスを PostgreSQL の `documents` テーブルに格納
5. `ask.py` がユーザーの質問をベクトル化し、pgvector `<->` 演算子で上位 SEARCH_K=10 件を取得
6. `reranker.py` が Cross-Encoder で候補をスコアリングし、上位 RERANK_TOP_K=3 件に絞り込み
7. リランク結果のコンテキストと質問を日本語プロンプトに組み立て
8. `llm.py` が Llama-2 で回答を生成し、ソース情報とともに出力

---

## フォルダ・ファイル一覧

```
llm-rag-cli/
├── app/                        # アプリケーション本体
│   ├── __init__.py             # パッケージ初期化
│   ├── config.py               # 環境変数・定数管理
│   ├── db.py                   # PostgreSQL接続・スキーマ初期化
│   ├── embeddings.py           # テキスト埋め込み（sentence-transformers）
│   ├── llm.py                  # LLM推論（llama-cpp-python）
│   ├── chunking.py             # テキストチャンク分割（固定サイズ・構造ベース）
│   ├── reranker.py             # Cross-Encoder リランキング
│   ├── ingest.py               # ドキュメント取り込みパイプライン
│   ├── ask.py                  # 質問応答CLI エントリポイント
│   ├── metrics.py              # 評価メトリクス（retrieval@k, faithfulness, latency）
│   └── evaluate.py             # 評価パイプライン実行
├── tests/                      # テストスイート（109テスト）
│   ├── __init__.py             # パッケージ初期化
│   ├── conftest.py             # 共有フィクスチャ（mock DB, fake embeddings, mock LLM）
│   ├── test_config.py          # config.py のテスト（13件）
│   ├── test_db.py              # db.py のテスト（7件）
│   ├── test_embeddings.py      # embeddings.py のテスト（5件）
│   ├── test_llm.py             # llm.py のテスト（4件）
│   ├── test_chunking.py        # chunking.py のテスト（14件）
│   ├── test_reranker.py        # reranker.py のテスト（9件）
│   ├── test_ingest.py          # ingest.py のテスト（9件）
│   ├── test_ask.py             # ask.py のテスト（10件）
│   ├── test_metrics.py         # metrics.py のテスト（14件）
│   └── test_evaluate.py        # evaluate.py のテスト（24件）
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
| `app/config.py` | `get_db_config()` で環境変数からDB接続情報を取得。`DB_CONFIG`, `EMBED_MODEL`, `LLM_MODEL_PATH`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `RERANKER_MODEL`, `SEARCH_K`, `RERANK_TOP_K` 定数を定義 |
| `app/db.py` | `get_conn()` でDB接続を返却。`init_db()` で pgvector 拡張と documents テーブル（source, chunk_index カラム含む）を作成 |
| `app/embeddings.py` | `get_model()` で SentenceTransformer を遅延ロード。`embed(texts)` でテキストリストを384次元ベクトルに変換 |
| `app/llm.py` | `get_llm()` で Llama-2 を遅延ロード。`generate(prompt)` でプロンプトから回答テキストを生成 |
| `app/chunking.py` | `split_text()` で固定サイズ・単語境界チャンク分割（overlap付き）、`split_by_structure()` で段落ベースチャンク分割 |
| `app/reranker.py` | `get_model()` で CrossEncoder を遅延ロード。`rerank(query, docs, top_k)` でクエリ・文書ペアをスコアリングし上位k件を返却 |
| `app/ingest.py` | `load_pdfs()` でPDFのページテキスト抽出（ソースメタデータ付き）、`load_csvs()` でCSVの行を `key:value` 形式に変換、`main()` でチャンク分割→埋め込み→INSERT の一連のパイプライン実行 |
| `app/ask.py` | `search(query)` でベクトル検索→リランキング、`main()` で日本語プロンプト構築・回答出力・ソース表示 |
| `app/metrics.py` | `retrieval_at_k()` で検索ヒット判定、`faithfulness()` でキーワード一致率算出、`measure_latency()` で関数実行時間計測 |
| `app/evaluate.py` | `load_questions()` で評価データ読み込み、`evaluate_single()` で個別評価、`run_evaluation()` で全問評価実行、`print_report()` でレポート出力 |
| `tests/conftest.py` | `mock_db_connection` (conn/cur モック、コンテキストマネージャ対応), `fake_embeddings` (3×384次元ダミー), `mock_llm_response` (LLM応答モック) |
| `data/eval_questions.json` | 評価用質問13問（query, expected_source, expected_keywords） |

---

## API一覧

### app/config.py

| 関数/定数 | シグネチャ | 説明 |
|-----------|-----------|------|
| `get_db_config()` | `() -> dict` | 環境変数から DB 接続設定を辞書で返却 |
| `DB_CONFIG` | `dict` | `get_db_config()` の評価結果（モジュール読み込み時に確定） |
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
| `get_conn()` | `() -> psycopg2.connection` | `DB_CONFIG` を使い PostgreSQL 接続を返却 |
| `init_db()` | `() -> None` | pgvector 拡張の有効化、documents テーブル（id, content, embedding, source, chunk_index）の作成、commit |

### app/embeddings.py

| 関数 | シグネチャ | 説明 |
|------|-----------|------|
| `get_model()` | `() -> SentenceTransformer` | モデルを遅延ロードして返却（シングルトン） |
| `embed(texts)` | `(list[str]) -> np.ndarray` | テキストリストを (N, 384) の埋め込みベクトルに変換 |

### app/llm.py

| 関数 | シグネチャ | 説明 |
|------|-----------|------|
| `get_llm()` | `() -> Llama` | LLM を遅延ロードして返却（n_ctx=2048、シングルトン） |
| `generate(prompt)` | `(str) -> str` | プロンプトを受け取り、max_tokens=300 で生成した回答テキストを返却 |

### app/chunking.py

| 関数 | シグネチャ | 説明 |
|------|-----------|------|
| `split_text(text, chunk_size=500, overlap=100)` | `(str, int, int) -> list[str]` | テキストを固定サイズで分割（単語境界保持、overlap付き）。空文字は空リスト、chunk_size以下はそのまま返却 |
| `split_by_structure(text, chunk_size=None, overlap=100)` | `(str, int\|None, int) -> list[str]` | 段落（`\n\n`）で分割。`chunk_size` 指定時は長い段落を `split_text` でさらに分割 |

### app/reranker.py

| 関数 | シグネチャ | 説明 |
|------|-----------|------|
| `get_model()` | `() -> CrossEncoder` | CrossEncoder モデルを遅延ロードして返却（シングルトン） |
| `rerank(query, docs, top_k=3)` | `(str, list[dict], int) -> list[dict]` | クエリと文書ペアをスコアリングし、スコア降順で上位 top_k 件の文書を返却。空リストは空リスト返却 |

### app/ingest.py

| 関数 | シグネチャ | 説明 |
|------|-----------|------|
| `load_pdfs()` | `() -> list[tuple[str, str]]` | `data/pdf/` 内の全 PDF から各ページのテキストとソース（`filename:pN`）をタプルで返却 |
| `load_csvs()` | `() -> list[tuple[str, str]]` | `data/csv/` 内の全 CSV から各行を `"key:value"` 形式とソース（`filename:rN`）でタプル返却 |
| `main()` | `() -> None` | DB初期化 → PDF/CSV読み込み → チャンク分割（PDF: `split_by_structure`, CSV: `split_text`） → 埋め込み → INSERT（source, chunk_index 付き） |

### app/ask.py

| 関数 | シグネチャ | 説明 |
|------|-----------|------|
| `search(query)` | `(str) -> list[dict]` | クエリを埋め込み、pgvector で上位 SEARCH_K 件取得後、`rerank()` で RERANK_TOP_K 件に絞り込み。各要素は `{"content": str, "source": str}` |
| `main()` | `() -> None` | `sys.argv[1]` から質問を取得、検索、日本語プロンプト構築、LLM 生成、回答とソース情報を出力 |

### app/metrics.py

| 関数 | シグネチャ | 説明 |
|------|-----------|------|
| `retrieval_at_k(results, expected_source)` | `(list[dict], str) -> bool` | 検索結果リスト内に期待ソースが含まれるか判定 |
| `faithfulness(answer, expected_keywords)` | `(str, list[str]) -> float` | 回答中のキーワード出現率を 0.0〜1.0 で返却。キーワード空リストは 1.0 |
| `measure_latency(func)` | `(Callable) -> tuple[Any, float]` | 関数を実行し、結果と経過時間（秒）のタプルを返却 |

### app/evaluate.py

| 関数 | シグネチャ | 説明 |
|------|-----------|------|
| `load_questions(path="data/eval_questions.json")` | `(str) -> list[dict]` | 評価用質問データをJSONから読み込み |
| `evaluate_single(query, expected_source, expected_keywords, search_fn, generate_fn)` | `(...) -> dict` | 1問を評価し、retrieval_hit, faithfulness, latency, answer を含む辞書を返却 |
| `run_evaluation(questions, search_fn, generate_fn)` | `(list[dict], Callable, Callable) -> list[dict]` | 全質問を順次評価し結果リストを返却 |
| `print_report(results, config)` | `(list[dict], dict) -> None` | 評価レポートを出力（Retrieval@k, Faithfulness, Latency, Re-rank状態） |
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
| `make lint` | 全モジュールの構文チェック (`py_compile`) |
| `make ingest` | ドキュメント取り込み実行 |
| `make ask Q="質問文"` | RAG に質問して回答を取得 |
| `make evaluate` | 評価パイプライン実行 (`python -m app.evaluate`) |

### Docker 直接実行

```bash
docker compose up -d                        # コンテナ起動
docker compose exec app bash                # コンテナに入る
python app/ingest.py                        # ドキュメント取り込み
python app/ask.py "制度の目的は？"           # 質問
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

```sql
-- pgvector 拡張
CREATE EXTENSION IF NOT EXISTS vector;

-- ドキュメント格納テーブル
CREATE TABLE IF NOT EXISTS documents (
    id          SERIAL PRIMARY KEY,
    content     TEXT,
    embedding   VECTOR(384),
    source      TEXT,
    chunk_index INT
);
```
