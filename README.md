# RAG CLI

PDF・CSVを取り込み、ベクトル検索＋ローカルLLMで質問に回答するCLI型RAGシステム。
CPU上で完結し、外部APIへの依存なし。

## 必要環境

- Docker / Docker Compose
- LLMモデルファイル: `models/llama-2-7b.Q4_K_M.gguf`

## 環境構築

### 1. リポジトリ取得

```bash
git clone <repository-url>
cd lll-rag-cli
```

### 2. LLMモデル配置

GGUF形式の Llama-2-7B（Q4_K_M量子化）を `models/` に配置する。

```bash
mkdir -p models
# ダウンロードしたモデルファイルを配置
cp /path/to/llama-2-7b.Q4_K_M.gguf models/
```

### 3. Docker起動

```bash
make build    # イメージビルド
make up       # コンテナ起動（PostgreSQL + app）
```

または直接:

```bash
docker compose build
docker compose up -d
```

### 4. データ配置

取り込みたいファイルを `data/` 配下に置く。

```
data/
├── pdf/    # PDFファイルを格納
└── csv/    # CSVファイルを格納
```

## 使い方

### ドキュメント取り込み

```bash
make ingest
```

`data/pdf/` と `data/csv/` 内のファイルを読み込み、ベクトル化してDBに格納する。

### 質問

```bash
make ask Q="制度の目的は？"
```

質問文をベクトル検索し、関連ドキュメントをコンテキストとしてLLMが回答を生成する。

### テスト実行

```bash
make test
```

### その他のコマンド

| コマンド | 説明 |
|----------|------|
| `make up` | コンテナ起動 |
| `make down` | コンテナ停止 |
| `make build` | イメージビルド |
| `make shell` | app コンテナに入る |
| `make lint` | 構文チェック |

## 環境変数

| 変数名 | デフォルト | 説明 |
|--------|-----------|------|
| `DB_HOST` | `localhost` | PostgreSQL ホスト（Docker内は `db`） |
| `DB_USER` | `rag` | DBユーザー |
| `DB_PASSWORD` | `rag` | DBパスワード |
| `DB_NAME` | `rag` | DB名 |

## ドキュメント

| ファイル | 内容 |
|----------|------|
| [docs/設計書.md](docs/設計書.md) | システム設計書（アーキテクチャ、各モジュール仕様、DB定義、Docker構成） |
| [README-実装一覧.md](README-実装一覧.md) | 実装詳細（API一覧、ファイル説明、データフロー、DBスキーマ） |
| [CLAUDE.md](CLAUDE.md) | Claude Code 向け開発ガイド |

## 技術スタック

| 領域 | 技術 |
|------|------|
| 言語 | Python 3.11 |
| LLM | Llama-2-7B (Q4_K_M) / llama-cpp-python |
| 埋め込み | all-MiniLM-L6-v2 / sentence-transformers |
| ベクトルDB | PostgreSQL 16 + pgvector |
| PDF解析 | pypdf |
| CSV解析 | pandas |
| コンテナ | Docker Compose |
| テスト | pytest (28テスト) |
