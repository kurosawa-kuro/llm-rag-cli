# RAG CLI

PDF・CSVを取り込み、ベクトル検索＋ローカルLLMで質問に回答するCLI型RAGシステム。
CPU上で完結し、外部APIへの依存なし。

## 必要環境

| 項目 | 要件 |
|------|------|
| OS | Linux / macOS / Windows（WSL2） |
| Docker | Docker Engine 24+ / Docker Compose V2 |
| Python | 3.10+（ホストでテスト実行する場合のみ） |
| ディスク | 約 6 GB（Docker イメージ + LLM モデル 4.1 GB） |

## クイックスタート

```bash
# 1. リポジトリ取得
git clone <repository-url>
cd llm-rag-cli

# 2. LLMモデルダウンロード（約 4.1 GB）
pip install huggingface-hub
huggingface-cli download TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_K_M.gguf --local-dir ./models

# 3. Docker ビルド＆起動
make build
make up

# 4. データ取り込み（コンテナ内で実行）
docker compose exec app python -m rag.data.ingest

# 5. 質問
docker compose exec app python -m cli.ask "制度の目的は？"
```

## 環境構築（詳細）

### 1. Docker のインストール

#### Ubuntu / WSL2

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2
sudo usermod -aG docker $USER
newgrp docker
```

#### WSL2 での Docker デーモン起動

WSL2 では systemd が無効な場合があるため、手動でデーモンを起動する。

```bash
sudo dockerd &
```

> Docker Desktop for Windows を使用している場合は WSL2 統合を有効にすれば上記は不要。

#### 動作確認

```bash
docker --version          # Docker Engine 24+ を確認
docker compose version    # Docker Compose V2 を確認
```

### 2. リポジトリ取得

```bash
git clone <repository-url>
cd llm-rag-cli
```

### 3. LLMモデル配置

GGUF形式の Llama-2-7B（Q4_K_M 量子化、約 4.1 GB）をダウンロードして `models/` に配置する。

```bash
pip install huggingface-hub
huggingface-cli download TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_K_M.gguf --local-dir ./models
```

配置後の確認:

```bash
ls -lh models/llama-2-7b.Q4_K_M.gguf
# → 約 4.1 GB のファイルが存在すること
```

### 4. Docker ビルド＆起動

```bash
make build    # Docker イメージのビルド
make up       # コンテナ起動（PostgreSQL + app）
```

起動確認:

```bash
docker compose ps
# → rag_db (healthy) と rag_app (running) が表示されること
```

### 5. データ配置

取り込みたいファイルを `data/` 配下に置く（サンプルデータ同梱済み）。

```
data/
├── pdf/    # PDFファイルを格納
├── csv/    # CSVファイルを格納
└── eval_questions.json  # 評価用質問セット
```

## 使い方

> **重要**: `ingest` / `ask` / `evaluate` はコンテナ内で実行する。`make up` でコンテナが起動していることを確認してから実行すること。

### ドキュメント取り込み

```bash
docker compose exec app python -m rag.data.ingest
```

`data/pdf/` と `data/csv/` 内のファイルを読み込み、チャンク分割・ベクトル化して PostgreSQL に格納する。

### 質問

```bash
docker compose exec app python -m cli.ask "制度の目的は？"
```

質問文をベクトル検索し、関連ドキュメントをコンテキストとしてLLMが回答を生成する。

### コンテナシェルに入って操作

```bash
make shell
# コンテナ内で直接コマンドを実行できる:
python -m rag.data.ingest
python -m cli.ask "質問文"
python -m rag.evaluation.evaluate
```

### 評価パイプライン

```bash
docker compose exec app python -m rag.evaluation.evaluate
```

評価用質問セット（13問）に対してパイプラインを実行し、Retrieval@k / Faithfulness / Exact Match / Latency を出力する。

## テスト

### ホストで実行する場合（Python 3.10+ 必要）

```bash
# 依存パッケージのインストール（初回のみ）
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 単体テスト（DB・モデル不要）
make test-unit
```

### コンテナ内で実行する場合

```bash
# 全テスト
docker compose exec app python -m pytest tests/ -v

# 単体テストのみ
docker compose exec app python -m pytest tests/ -v -m "not integration and not heavy"

# DB統合テストのみ
docker compose exec app python -m pytest tests/ -v -m "integration and not heavy"

# 実 Embeddings テスト（モデルDL発生）
docker compose exec app python -m pytest tests/ -v -m heavy
```

### テスト種別

| 種別 | 件数 | 必要環境 | コマンド |
|------|------|----------|----------|
| 単体テスト | 268 | Python のみ | `make test-unit` |
| DB 統合テスト | 7 | PostgreSQL | `make test-integration` |
| Heavy テスト | 3 | PostgreSQL + モデルDL | `make test-heavy` |
| 全テスト | 278+ | 全環境 | `make test` |

## Make コマンド一覧

| コマンド | 実行環境 | 説明 |
|----------|----------|------|
| `make build` | ホスト | Docker イメージビルド |
| `make up` | ホスト | コンテナ起動（PostgreSQL + app） |
| `make down` | ホスト | コンテナ停止 |
| `make shell` | ホスト | app コンテナに入る |
| `make test` | ホスト/コンテナ | 全テスト実行 |
| `make test-unit` | ホスト/コンテナ | 単体テストのみ（DB不要） |
| `make test-integration` | コンテナ | DB統合テストのみ |
| `make test-heavy` | コンテナ | 実 Embeddings テスト |
| `make lint` | ホスト/コンテナ | 構文チェック（全15モジュール） |
| `make ingest` | コンテナ | データ取り込み |
| `make ask Q="質問文"` | コンテナ | RAG 質問応答 |
| `make evaluate` | コンテナ | 評価パイプライン実行 |

> `make ingest` / `make ask` / `make evaluate` をホストで直接実行すると依存パッケージ不足でエラーになる。`make shell` でコンテナに入ってから実行するか、`docker compose exec app ...` を使うこと。

## 環境変数

設定は `env/config/setting.yaml` で管理。環境変数で上書き可能。

| 変数名 | デフォルト | 説明 |
|--------|-----------|------|
| `DB_HOST` | `localhost` | PostgreSQL ホスト（Docker内は `db`） |
| `DB_USER` | `rag` | DBユーザー |
| `DB_PASSWORD` | `rag` | DBパスワード |
| `DB_NAME` | `rag` | DB名 |
| `CHUNK_SIZE` | `500` | チャンクサイズ（文字数） |
| `CHUNK_OVERLAP` | `100` | チャンク間オーバーラップ（文字数） |
| `SEARCH_K` | `10` | ベクトル検索の取得件数 |
| `RERANK_TOP_K` | `3` | リランキング後の上位件数 |

## トラブルシューティング

### `Cannot connect to the Docker daemon`

Docker デーモンが起動していない。

```bash
# WSL2 の場合
sudo dockerd &

# Linux（systemd）の場合
sudo systemctl start docker
```

### `make ingest` で `ModuleNotFoundError`

ホストで直接実行している。コンテナ内で実行すること。

```bash
make up
docker compose exec app python -m rag.data.ingest
```

### `models/llama-2-7b.Q4_K_M.gguf` が見つからない

モデルファイルが未ダウンロード。

```bash
pip install huggingface-hub
huggingface-cli download TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_K_M.gguf --local-dir ./models
```

### heavy テストで `sentence_transformers` エラー

ホスト実行時は `sentence-transformers` が必要。

```bash
pip install sentence-transformers
```

またはコンテナ内で実行すれば不要:

```bash
docker compose exec app python -m pytest tests/ -v -m heavy
```

## ドキュメント

| ファイル | 内容 |
|----------|------|
| [docs/設計書.md](docs/設計書.md) | システム設計書（アーキテクチャ、各モジュール仕様、DB定義、Docker構成） |
| [README-実装一覧.md](README-実装一覧.md) | 実装詳細（API一覧、ファイル説明、データフロー、DBスキーマ） |
| [docs/リファクタリング.md](docs/リファクタリング.md) | リファクタリング記録 |
| [docs/実Embeddings統合テストの最小.md](docs/実Embeddings統合テストの最小.md) | 実Embeddings統合テストガイド |
| [CLAUDE.md](CLAUDE.md) | Claude Code 向け開発ガイド |

## 技術スタック

| 領域 | 技術 |
|------|------|
| 言語 | Python 3.11 |
| フレームワーク | LangChain + LangGraph |
| LLM | Llama-2-7B (Q4_K_M) / llama-cpp-python |
| 埋め込み | all-MiniLM-L6-v2 / sentence-transformers |
| リランカー | ms-marco-MiniLM-L-6-v2（Cross-Encoder） |
| ベクトルDB | PostgreSQL 16 + pgvector |
| PDF解析 | pypdf |
| CSV解析 | pandas |
| 設定管理 | pyyaml（setting.yaml） |
| コンテナ | Docker Compose |
| テスト | pytest（280テスト: 単体268 + DB統合7 + heavy3、17ファイル） |

  恒久的な対処（dockerグループに追加）:                                                                                                       
  sudo usermod -aG docker $USER
  newgrp docker

● pgAdmin を追加しました。

  アクセス方法:                                                                                                                               
  - URL: http://localhost:8080
  - ログイン: admin@example.com / admin                                                                                                       
  - DBサーバー「RAG DB」が自動登録済み（初回接続時にパスワード rag を入力）  