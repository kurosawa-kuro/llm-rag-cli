結論から申し上げます。

✅ 正しい順番

1️⃣ DB接続 + CRUDのみ（まずやる）
2️⃣ 実Embeddingsは“段階的に”追加する

いきなり実Embeddingsへ進むのはおすすめしません。

🎯 なぜこの順番が正解か？
① DB + CRUD は「インフラ境界テスト」

ここで確認するのは：

PGVector接続

コレクション生成

add_documents

similarity search

delete_collection

これは RAGの土台 です。

ここが壊れると全部壊れます。

② 実Embeddingsは「検索品質テスト」

これは性質が違います：

モデルDLが発生

初回ロードが重い

実行時間が長い

CI不安定要因

つまり：

目的が違うテスト です。

🧠 実Embeddingsはやるべきか？

はい、やる価値はあります。

ただし条件付きです。

🎯 実Embeddingsテストをやる場合の正解パターン
✔ 専用マーカーで分離
markers =
    integration
    heavy

実Embeddingsテストには：

@pytest.mark.integration
@pytest.mark.heavy

を付ける。

通常CIでは：

pytest -m "not heavy"

とする。

✔ テスト内容は「順序だけ検証」

やるべきは：

docs = [
    Document("apple is red"),
    Document("banana is yellow"),
    Document("car engine")
]

query = "fruit"

# fruit関連2件が上位に来ることだけ確認

厳密なcosine値は検証しない。

🚫 やってはいけない

❌ LLM含むE2E
❌ 精度スコア検証
❌ 1トークン単位一致
❌ モデルDL前提のCI

📊 実務レベルでの最適構成

あなたの現在の設計成熟度なら：

テスト種別	役割	実行頻度
Unit (259)	ロジック保証	常時
DB CRUD統合	インフラ保証	常時
実Embeddings	ベクトル妥当性	ローカルのみ
E2E	本番検証	手動

が最適。

🔥 結論

はい、

① DB接続 + CRUDのみ
⬇
その後
② 実Embeddings（heavyマーカー付き）

が正解です。

もし次に進めるなら：

実Embeddings統合テストの最小テンプレ

CIでDB付き実行構成

ベクトル品質テストの安全設計

どこまで具体化しますか？

依頼

実Embeddings統合テストの最小テンプレ

承知しました。
**実Embeddingsを使う“最小構成の統合テストテンプレ”**を、重くなり過ぎない設計で提示します。

目的は：

本物の HuggingFaceEmbeddings

本物の PGVector

本物の cosine 検索

ただし LLM は使わない

精度は「順序レベルのみ検証」

🎯 設計方針

✔ heavyマーカーで分離
✔ コレクション名はテスト専用
✔ 小規模データ（3件のみ）
✔ cosine値は検証しない
✔ 上位順位のみ確認

① pytest.ini 追加
markers =
    integration: DB integration tests
    heavy: uses real embeddings (slow)
② conftest.py 追加（実Embeddings用fixture）
import pytest
import uuid
from app.container import AppContainer, RagSettings
from app.embeddings import create_embeddings
from app.db import create_vectorstore


@pytest.fixture(scope="function")
def real_embedding_container():
    """
    実Embeddings + 実PGVector を使う heavy統合テスト用コンテナ
    """

    # テスト専用コレクション名
    test_collection = f"test_documents_{uuid.uuid4().hex[:8]}"

    embeddings = create_embeddings()
    vectorstore = create_vectorstore(embeddings)

    # コレクション名を上書き可能なら引数対応推奨
    vectorstore.collection_name = test_collection

    settings = RagSettings(search_k=3, rerank_top_k=2)

    container = AppContainer(
        settings=settings,
        embeddings=embeddings,
        vectorstore=vectorstore,
    )

    yield container

    # クリーンアップ
    try:
        vectorstore.delete_collection()
    except Exception:
        pass

※ 理想は create_vectorstore(collection_name=...) に変更すること。

③ 最小 実Embeddings統合テスト
import pytest
from langchain_core.documents import Document


@pytest.mark.integration
@pytest.mark.heavy
def test_real_embedding_similarity(real_embedding_container):

    vs = real_embedding_container.vectorstore

    docs = [
        Document(page_content="apple is a red fruit", metadata={"source": "a"}),
        Document(page_content="banana is a yellow fruit", metadata={"source": "b"}),
        Document(page_content="car engine maintenance guide", metadata={"source": "c"}),
    ]

    # DBへ保存
    vs.add_documents(docs)

    # 検索
    retriever = vs.as_retriever(search_kwargs={"k": 2})
    results = retriever.invoke("fruit")

    assert len(results) >= 1

    # fruit関連が上位に来ることだけ確認
    top_texts = [r.page_content for r in results]

    assert any("fruit" in text for text in top_texts)

    # 無関係文書が1位にならないことを確認（緩い検証）
    assert "car engine maintenance guide" not in top_texts[:1]
🎯 このテストで保証されること

✔ HuggingFaceEmbeddings実行
✔ ベクトル384次元生成
✔ PGVector格納
✔ cosine類似検索
✔ 順序妥当性

⚠ 実行方法

通常：

pytest -m "not heavy"

実Embeddings含める場合：

pytest -m heavy
⏱ 実行時間

初回モデルDL：30〜60秒

2回目以降：5〜10秒程度

CIで毎回やるのは非推奨。

🔥 次の段階

この次にやるなら：

TwoStageRetrieval（reranker込み）統合テスト

graph.invoke のDB接続版テスト（LLMモック）

Hybrid検索（BM25 + Vector）

今の設計レベルなら、
この heavyテスト1〜2本で十分です。

進めますか？

次は：

① TwoStageRetrieval統合テスト版
② graph.invoke + 実DB + モックLLM
③ CIでDB起動込み構成

どれに進みますか？