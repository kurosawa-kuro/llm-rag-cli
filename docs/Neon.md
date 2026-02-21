# Docker PostgreSQL → Neon 移行手順書

## 概要

本ドキュメントは、現在 Docker Compose で運用している PostgreSQL（pgvector/pgvector:pg16）を、Neon（サーバーレス PostgreSQL）に移行するための作業手順を記載する。

### 現行構成

| 項目 | 現在の値 |
|------|---------|
| PostgreSQL | Docker `pgvector/pgvector:pg16` コンテナ |
| 接続文字列 | `postgresql+psycopg://rag:rag@db:5432/rag` |
| pgvector | Docker イメージに同梱 |
| ドライバ | `psycopg[binary]>=3.1.0` / `psycopg2-binary` |
| スキーマ管理 | LangChain PGVector が自動作成 |

### 移行後の構成

| 項目 | 移行後の値 |
|------|-----------|
| PostgreSQL | Neon サーバーレス PostgreSQL |
| 接続文字列 | `postgresql+psycopg://user:pass@ep-xxx.region.aws.neon.tech/dbname?sslmode=require` |
| pgvector | Neon コンソールで有効化（`CREATE EXTENSION vector`） |
| ドライバ | 変更なし（psycopg3 はそのまま利用可能） |
| スキーマ管理 | 変更なし（LangChain PGVector が自動作成） |

---

## 前提条件

- Neon アカウントを作成済みであること（https://neon.tech）
- Neon の Free Tier または有料プランを利用可能であること
- Neon Free Tier 制限: ストレージ 0.5GB、コンピュート 190h/月（2026年2月時点）

---

## 作業手順

### Step 1: Neon プロジェクトの作成

1. [Neon Console](https://console.neon.tech) にログイン
2. **「New Project」** をクリック
3. 以下を設定:
   - **Project name**: `llm-rag` （任意）
   - **PostgreSQL version**: `16`（現行と同じバージョンを推奨）
   - **Region**: 最寄りのリージョンを選択（例: `Asia Pacific (Tokyo)` / `ap-northeast-1`）
4. **「Create Project」** をクリック
5. 表示される接続情報を控える:
   - **Host**: `ep-xxxx-yyyy-12345678.ap-northeast-1.aws.neon.tech`
   - **Database**: `neondb`（デフォルト。変更可能）
   - **User**: `neondb_owner`（デフォルト）
   - **Password**: 自動生成されたパスワード

### Step 2: pgvector 拡張の有効化

Neon コンソールの **SQL Editor** またはローカルの `psql` から以下を実行:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

> **補足**: Neon は pgvector をネイティブサポートしているため、追加インストールは不要。`CREATE EXTENSION` のみで有効化できる。

確認:

```sql
SELECT * FROM pg_extension WHERE extname = 'vector';
```

### Step 3: 環境変数 / 接続情報の更新

#### 3-1. `env/config/setting.yaml` の変更

```yaml
# 変更前
db:
  host: localhost
  user: rag
  password: rag
  name: rag
  port: 5432

# 変更後
db:
  host: ep-xxxx-yyyy-12345678.ap-northeast-1.aws.neon.tech
  user: neondb_owner
  password: "<Neonで生成されたパスワード>"
  name: neondb
  port: 5432
```

> **注意**: パスワードに特殊文字（`@`, `/`, `%` 等）が含まれる場合は URL エンコードが必要。`config.py` の接続文字列構築時に問題になるため、Step 4 で対応する。

#### 3-2. `.env` ファイルの作成（推奨）

機密情報を `setting.yaml` に直接書かず、環境変数で上書きする方式を推奨する。プロジェクトルートに `.env` ファイルを作成:

```bash
# .env
DB_HOST=ep-xxxx-yyyy-12345678.ap-northeast-1.aws.neon.tech
DB_USER=neondb_owner
DB_PASSWORD=<Neonで生成されたパスワード>
DB_NAME=neondb
```

`.gitignore` に `.env` が含まれていることを確認:

```bash
echo ".env" >> .gitignore
```

### Step 4: `src/rag/core/config.py` の変更

Neon は SSL 接続を必須とするため、接続文字列に `sslmode=require` を追加する。また、パスワードの URL エンコードに対応する。

```python
# 変更前（現行）
def get_connection_string():
    c = get_db_config()
    port = _settings["db"]["port"]
    return f"postgresql+psycopg://{c['user']}:{c['password']}@{c['host']}:{port}/{c['dbname']}"

# 変更後
def get_connection_string():
    from urllib.parse import quote_plus
    c = get_db_config()
    port = _settings["db"]["port"]
    password = quote_plus(c["password"])
    return f"postgresql+psycopg://{c['user']}:{password}@{c['host']}:{port}/{c['dbname']}?sslmode=require"
```

**変更点**:
1. `sslmode=require` クエリパラメータを追加（Neon は TLS 必須）
2. `quote_plus` でパスワードを URL エンコード（特殊文字対策）

> **補足**: Docker PostgreSQL（ローカル）を引き続き使いたい場合は、環境変数 `DB_SSLMODE` で切り替え可能にする方法もある（後述の「ローカル/Neon 切り替え対応」を参照）。

### Step 5: `docker-compose.yml` の変更

Neon に移行後、Docker の PostgreSQL コンテナと pgAdmin は不要になる。

```yaml
# 変更前
services:
  db:
    image: pgvector/pgvector:pg16
    container_name: rag_db
    environment:
      POSTGRES_USER: rag
      POSTGRES_PASSWORD: rag
      POSTGRES_DB: rag
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U rag -d rag"]
      interval: 2s
      timeout: 5s
      retries: 30

  app:
    build: .
    container_name: rag_app
    depends_on:
      db:
        condition: service_healthy
    volumes:
      - .:/app
    environment:
      DB_HOST: db
      DB_USER: rag
      DB_PASSWORD: rag
      DB_NAME: rag
      PYTHONPATH: /app/src
    tty: true

  pgadmin:
    image: dpage/pgadmin4
    container_name: rag_pgadmin
    depends_on:
      db:
        condition: service_healthy
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@example.com
      PGADMIN_DEFAULT_PASSWORD: admin
      PGADMIN_SERVER_JSON_FILE: /pgadmin4/servers.json
    ports:
      - "8080:80"
    volumes:
      - pgadmin_data:/var/lib/pgadmin
      - ./env/pgadmin/servers.json:/pgadmin4/servers.json:ro

volumes:
  pgdata:
  pgadmin_data:
```

```yaml
# 変更後
services:
  app:
    build: .
    container_name: rag_app
    env_file:
      - .env
    volumes:
      - .:/app
    environment:
      PYTHONPATH: /app/src
    tty: true
```

**変更点**:
1. `db` サービス（PostgreSQL コンテナ）を削除
2. `pgadmin` サービスを削除
3. `app` サービスの `depends_on` を削除
4. DB 接続情報を `.env` ファイルから読み込むように変更（`env_file`）
5. `volumes` セクションから `pgdata`, `pgadmin_data` を削除

### Step 6: `requirements.txt` の確認

現行の依存パッケージはそのまま Neon でも動作する。変更不要。

```
psycopg[binary]>=3.1.0      # Neon と互換性あり
langchain-postgres>=0.0.12   # PGVector 利用
```

> **補足**: `psycopg2-binary` は `psycopg[binary]` (v3) と共存しているが、本プロジェクトでは psycopg3 経由（`postgresql+psycopg://`）で接続しているため、`psycopg2-binary` は削除しても問題ない。将来的なクリーンアップとして検討。

### Step 7: テストの確認・修正

#### 7-1. ユニットテスト（変更不要）

ユニットテストは DB をモックしているため、影響なし。

```bash
make test-unit
```

#### 7-2. 統合テスト

`tests/conftest.py` の `_check_db_connection()` は `CONNECTION_STRING` を使って接続を試みる。Neon の接続情報が正しく設定されていれば、変更不要で動作する。

ただし、統合テストを Neon に対して実行する場合は以下に注意:

- **コールドスタート遅延**: Neon のサーバーレスコンピュートは一定時間アイドル後にサスペンドされる。初回接続時に数秒のコールドスタートが発生する可能性がある
- **テスト用ブランチの利用（推奨）**: Neon の「Branch」機能で本番データベースとは別のブランチを作成し、テスト用途に使用する

```bash
# Neon CLI でテスト用ブランチ作成
neonctl branches create --name test-branch --project-id <project-id>
```

#### 7-3. テスト実行時の環境変数設定

Docker 外でテストを実行する場合:

```bash
export DB_HOST=ep-xxxx-yyyy-12345678.ap-northeast-1.aws.neon.tech
export DB_USER=neondb_owner
export DB_PASSWORD=<password>
export DB_NAME=neondb
make test-integration
```

### Step 8: データの再インジェスト

Neon 上の新しいデータベースにデータを投入する:

```bash
# Docker コンテナ内で実行する場合
make ingest

# ローカルで実行する場合（環境変数を設定済みであること）
python -m rag.data.ingest
```

> **補足**: `ingest.py` は `delete_collection()` → vectorstore 再生成 → `add_documents()` の流れで動作するため、既存データの移行（ダンプ/リストア）は不要。再インジェストで同じ結果が得られる。

### Step 9: 動作確認

```bash
# クエリの実行テスト
make ask Q="テスト質問"

# ユニットテスト
make test-unit

# 統合テスト（Neon 接続）
make test-integration
```

---

## 既存データの移行（任意）

再インジェスト（Step 8）ではなく既存データを保持したい場合は、`pg_dump` / `pg_restore` で移行可能。

```bash
# 1. Docker PostgreSQL からダンプ
docker compose exec db pg_dump -U rag -d rag --format=custom -f /tmp/rag_dump.dump
docker cp rag_db:/tmp/rag_dump.dump ./rag_dump.dump

# 2. Neon にリストア
pg_restore --no-owner --no-acl \
  -d "postgresql://neondb_owner:<password>@ep-xxxx.ap-northeast-1.aws.neon.tech/neondb?sslmode=require" \
  ./rag_dump.dump
```

> **注意**: pgvector 拡張は事前に有効化しておくこと（Step 2）。

---

## ローカル/Neon 切り替え対応（任意）

開発時はローカル Docker PostgreSQL、本番は Neon という構成にしたい場合は、`DATABASE_URL` 環境変数による直接指定をサポートする。

### `src/rag/core/config.py` の拡張案

```python
def get_connection_string():
    # DATABASE_URL が設定されていればそのまま使用
    direct_url = os.getenv("DATABASE_URL")
    if direct_url:
        return direct_url

    from urllib.parse import quote_plus
    c = get_db_config()
    port = _settings["db"]["port"]
    password = quote_plus(c["password"])
    sslmode = os.getenv("DB_SSLMODE", "require")
    return f"postgresql+psycopg://{c['user']}:{password}@{c['host']}:{port}/{c['dbname']}?sslmode={sslmode}"
```

使い分け:
```bash
# ローカル Docker 使用時
export DB_SSLMODE=disable

# Neon 使用時（デフォルト）
# DB_SSLMODE は設定不要（デフォルトで require）

# 接続文字列を直接指定する場合
export DATABASE_URL="postgresql+psycopg://user:pass@ep-xxx.neon.tech/dbname?sslmode=require"
```

---

## Neon 固有の考慮事項

### コネクションプーリング

Neon はビルトインのコネクションプーラーを提供している。接続数が多い場合はプーリングエンドポイントを使用する:

- **Direct接続**: `ep-xxxx-yyyy-12345678.ap-northeast-1.aws.neon.tech`（通常利用）
- **Pooled接続**: `ep-xxxx-yyyy-12345678-pooler.ap-northeast-1.aws.neon.tech`（`-pooler` サフィックス付き）

本プロジェクトは CLI ツールで同時接続数が少ないため、Direct 接続で十分。

### オートサスペンド

Neon の Free Tier ではコンピュートが 5 分間アイドル後にサスペンドされる。次回接続時にコールドスタート（約 0.5〜2 秒）が発生する。

対策:
- 有料プランで `auto_suspend_timeout` を調整
- アプリケーション側でリトライロジックを追加（必要に応じて）

### ストレージ制限

| プラン | ストレージ | コンピュート |
|--------|-----------|-------------|
| Free | 0.5 GB | 190 時間/月 |
| Launch | 10 GB | 300 時間/月 |
| Scale | 50 GB | 750 時間/月 |

本プロジェクトのベクトルデータ（384 次元, JSONB）のサイズに応じてプランを選択する。

---

## 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `env/config/setting.yaml` | DB 接続情報を Neon のものに変更 |
| `src/rag/core/config.py` | `sslmode=require` 追加、パスワード URL エンコード対応 |
| `docker-compose.yml` | `db`, `pgadmin` サービス削除、`env_file` 追加 |
| `.env`（新規） | Neon 接続情報を記載 |
| `.gitignore` | `.env` を追加 |

**変更不要なファイル**:
- `src/rag/infra/db.py` — `CONNECTION_STRING` を使うのみで変更不要
- `src/rag/core/container.py` — DI Container は変更不要
- `src/rag/data/ingest.py` — 変更不要
- `src/rag/pipeline/retrieval.py` — 変更不要
- `tests/conftest.py` — 変更不要（接続文字列は `config.py` から取得）
- `requirements.txt` — 変更不要（psycopg3 は Neon 互換）

---

## チェックリスト

- [ ] Neon プロジェクト作成
- [ ] pgvector 拡張有効化（`CREATE EXTENSION vector`）
- [ ] `setting.yaml` の DB 接続情報更新
- [ ] `.env` ファイル作成・`.gitignore` 追加
- [ ] `config.py` に `sslmode=require` とパスワードエンコード追加
- [ ] `docker-compose.yml` から `db` / `pgadmin` サービス削除
- [ ] ユニットテスト通過確認（`make test-unit`）
- [ ] Neon への接続確認（`psql` で手動接続テスト）
- [ ] データインジェスト実行（`make ingest`）
- [ ] クエリ動作確認（`make ask Q="テスト質問"`）
- [ ] 統合テスト通過確認（`make test-integration`）
- [ ] Docker PostgreSQL ボリュームの削除（`docker volume rm llm-rag-cli_pgdata`）
