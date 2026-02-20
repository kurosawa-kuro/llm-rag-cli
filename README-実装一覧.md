# RAG CLI 実装一覧

## 機能概要

PDF・CSVファイルを取り込み、ベクトル検索で関連情報を取得し、ローカルLLMで回答を生成するCLI型RAGシステム。

| 項目 | 内容 |
|------|------|
| 推論方式 | CPU推論（外部API不要） |
| 対応ファイル | PDF, CSV |
| ベクトルDB | PostgreSQL 16 + pgvector |
| 埋め込みモデル | all-MiniLM-L6-v2（384次元） |
| LLM | Llama-2-7B Q4_K_M（GGUF形式） |
| 実行環境 | Docker（Python 3.11-slim） |

---

## アーキテクチャ

```
data/pdf/  ──┐
data/csv/  ──┤
             ▼
         ingest.py ── embeddings.py ──→ PostgreSQL + pgvector
                                         (documents テーブル)
                                              │
                                         ベクトル検索 (k=3)
                                              │
         ask.py ◄────────────────────────────┘
           │
           ├── embeddings.py  (クエリ埋め込み)
           ├── db.py          (類似検索)
           └── llm.py         (回答生成)
                 │
                 ▼
            Llama-2 (CPU)
```

**データフロー:**

1. `ingest.py` が PDF/CSV を読み込み、テキストを抽出
2. `embeddings.py` がテキストを384次元ベクトルに変換
3. ベクトルとテキストを PostgreSQL の `documents` テーブルに格納
4. `ask.py` がユーザーの質問をベクトル化し、pgvector `<->` 演算子で類似検索
5. 上位k件のコンテキストと質問を日本語プロンプトに組み立て
6. `llm.py` が Llama-2 で回答を生成

---

## フォルダ・ファイル一覧

```
lll-rag-cli/
├── app/                        # アプリケーション本体
│   ├── __init__.py             # パッケージ初期化
│   ├── config.py               # 環境変数・定数管理
│   ├── db.py                   # PostgreSQL接続・スキーマ初期化
│   ├── embeddings.py           # テキスト埋め込み（sentence-transformers）
│   ├── llm.py                  # LLM推論（llama-cpp-python）
│   ├── ingest.py               # ドキュメント取り込みパイプライン
│   └── ask.py                  # 質問応答CLI エントリポイント
├── tests/                      # テストスイート（28テスト）
│   ├── __init__.py             # パッケージ初期化
│   ├── conftest.py             # 共有フィクスチャ（mock DB, fake embeddings, mock LLM）
│   ├── test_config.py          # config.py のテスト（4件）
│   ├── test_db.py              # db.py のテスト（5件）
│   ├── test_embeddings.py      # embeddings.py のテスト（5件）
│   ├── test_llm.py             # llm.py のテスト（4件）
│   ├── test_ingest.py          # ingest.py のテスト（5件）
│   └── test_ask.py             # ask.py のテスト（5件）
├── data/                       # 入力データ配置先
│   ├── pdf/                    # PDF ファイル格納
│   └── csv/                    # CSV ファイル格納
├── models/                     # LLM モデル配置先
│   └── (llama-2-7b.Q4_K_M.gguf)  # 手動配置が必要
├── docs/
│   └── 設計書.md                # 設計ドキュメント
├── Dockerfile                  # Python 3.11-slim ベースイメージ
├── docker-compose.yml          # app + PostgreSQL 16 (pgvector) 構成
├── requirements.txt            # 本番依存パッケージ
├── requirements-dev.txt        # テスト依存パッケージ（pytest, pytest-mock）
├── pytest.ini                  # pytest 設定
├── Makefile                    # 開発用コマンド集
├── CLAUDE.md                   # Claude Code 向けガイド
└── README.md                   # プロジェクト README
```

### ファイル詳細

| ファイル | 役割 |
|----------|------|
| `app/config.py` | `get_db_config()` で環境変数からDB接続情報を取得。`EMBED_MODEL`, `LLM_MODEL_PATH` 定数を定義 |
| `app/db.py` | `get_conn()` でDB接続を返却。`init_db()` で pgvector 拡張と documents テーブルを作成 |
| `app/embeddings.py` | `get_model()` で SentenceTransformer を遅延ロード。`embed(texts)` でテキストリストを384次元ベクトルに変換 |
| `app/llm.py` | `get_llm()` で Llama-2 を遅延ロード。`generate(prompt)` でプロンプトから回答テキストを生成 |
| `app/ingest.py` | `load_pdfs()` でPDFのページテキスト抽出、`load_csvs()` でCSVの行を `key:value` 形式に変換、`main()` で取り込みパイプライン実行 |
| `app/ask.py` | `search(query, k)` でベクトル類似検索、`main()` で日本語プロンプト構築・回答出力 |
| `tests/conftest.py` | `mock_db_connection` (conn/cur モック), `fake_embeddings` (384次元ダミー), `mock_llm_response` (LLM応答モック) |

---

## API一覧

### app/config.py

| 関数/定数 | シグネチャ | 説明 |
|-----------|-----------|------|
| `get_db_config()` | `() -> dict` | 環境変数から DB 接続設定を辞書で返却 |
| `DB_CONFIG` | `dict` | `get_db_config()` の評価結果（モジュール読み込み時に確定） |
| `EMBED_MODEL` | `str` | `"sentence-transformers/all-MiniLM-L6-v2"` |
| `LLM_MODEL_PATH` | `str` | `"./models/llama-2-7b.Q4_K_M.gguf"` |

### app/db.py

| 関数 | シグネチャ | 説明 |
|------|-----------|------|
| `get_conn()` | `() -> psycopg2.connection` | `DB_CONFIG` を使い PostgreSQL 接続を返却 |
| `init_db()` | `() -> None` | pgvector 拡張の有効化、documents テーブルの作成、commit |

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

### app/ingest.py

| 関数 | シグネチャ | 説明 |
|------|-----------|------|
| `load_pdfs()` | `() -> list[str]` | `data/pdf/` 内の全 PDF から各ページのテキストを抽出 |
| `load_csvs()` | `() -> list[str]` | `data/csv/` 内の全 CSV から各行を `"key:value"` 形式の文字列に変換 |
| `main()` | `() -> None` | DB初期化 → PDF/CSV読み込み → 埋め込み → INSERT の一連のパイプライン |

### app/ask.py

| 関数 | シグネチャ | 説明 |
|------|-----------|------|
| `search(query, k=3)` | `(str, int) -> list[str]` | クエリを埋め込み、pgvector `<->` 演算子で上位 k 件の content を返却 |
| `main()` | `() -> None` | `sys.argv[1]` から質問を取得、検索、日本語プロンプト構築、LLM 生成、結果出力 |

---

## コマンド一覧

### Make コマンド

| コマンド | 説明 |
|----------|------|
| `make up` | Docker コンテナをバックグラウンド起動 |
| `make down` | Docker コンテナを停止・削除 |
| `make build` | Docker イメージをビルド |
| `make shell` | app コンテナの bash に接続 |
| `make test` | 全テスト実行 (`pytest tests/ -v`) |
| `make ingest` | ドキュメント取り込み実行 |
| `make ask Q="質問文"` | RAG に質問して回答を取得 |
| `make lint` | 全モジュールの構文チェック |

### Docker 直接実行

```bash
docker compose up -d                        # コンテナ起動
docker compose exec app bash                # コンテナに入る
python app/ingest.py                        # ドキュメント取り込み
python app/ask.py "制度の目的は？"           # 質問
python -m pytest tests/ -v                  # テスト実行
```

### 環境変数

| 変数名 | デフォルト値 | 説明 |
|--------|-------------|------|
| `DB_HOST` | `localhost` | PostgreSQL ホスト（Docker 内は `db`） |
| `DB_USER` | `rag` | データベースユーザー |
| `DB_PASSWORD` | `rag` | データベースパスワード |
| `DB_NAME` | `rag` | データベース名 |

---

## DBスキーマ

```sql
-- pgvector 拡張
CREATE EXTENSION IF NOT EXISTS vector;

-- ドキュメント格納テーブル
CREATE TABLE IF NOT EXISTS documents (
    id        SERIAL PRIMARY KEY,
    content   TEXT,
    embedding VECTOR(384)
);
```
