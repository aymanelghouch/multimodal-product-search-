from qdrant_client import QdrantClient, models
import numpy as np
import pandas as pd
from pathlib import Path
from src.similarity import l2_normalize
from src.embeddings import embed_text





ROOT = Path(__file__).resolve().parents[1]
COLLECTION = "products"

df = pd.read_csv(ROOT / "data" / "processed" / "products.csv")
X = np.load(ROOT / "data" / "processed" / "clip_image_embeddings.npy")

Xn = l2_normalize(X)




PAYLOAD_FIELDS = [
    "articleType", "baseColour", "gender", "masterCategory",
    "subCategory", "usage", "season", "productDisplayName", "image_path",
]

client = QdrantClient(path=str(ROOT / "data" / "qdrant_db"))


if not client.collection_exists(COLLECTION):
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(
        size= X.shape[1],
        distance=models.Distance.COSINE
    ),
)

points = []

for pid, vector, row in zip(df["id"], Xn, df.to_dict("records")):
    payload = {k: row[k] for k in PAYLOAD_FIELDS}
    points.append(
        models.PointStruct(id=pid, vector=vector.tolist(), payload=payload)
)


client.upsert(collection_name=COLLECTION, points=points)




def search_qdrant(query, k=5, article_type=None):
    q = l2_normalize(embed_text([query]))[0]

    query_filter = None
    if article_type is not None:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="articleType",
                    match=models.MatchValue(value=article_type),
                )
            ]
        )

    res = client.query_points(
        collection_name=COLLECTION,
        query=q.tolist(),
        query_filter=query_filter,
        limit=k,
    )
    return res.points


for p in search_qdrant("black running shoes", article_type="Sports Shoes"):
    print(f"{p.score:.3f}  {p.payload['articleType']:15s} {p.payload['productDisplayName']}")

client.close()

