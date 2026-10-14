from qdrant_client import QdrantClient, models
import numpy as np
import pandas as pd
from pathlib import Path
from src.similarity import l2_normalize
from src.embeddings import embed_text, embed_images





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



def search_by_vector(vector, k=5, article_type=None):
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
        query=vector,
        query_filter=query_filter,
        limit=k
        
    )
    return res.points



def search_text(query, k=5, article_type=None):
    q = l2_normalize(embed_text([query]))[0]
    return search_by_vector(q, k=k, article_type=article_type)

def search_image(image_path, k=5, article_type=None):
    
    q = l2_normalize(embed_images([image_path]))[0]
    return search_by_vector(q, k=k, article_type=article_type)



def search_combined(image_path,text,alpha=0.5,k=5,article_type=None):
    q1 = l2_normalize(embed_text([text]))[0]

    q2 = l2_normalize(embed_images([image_path]))[0]

    q = alpha*q2 + (1-alpha)*q1

    return search_by_vector(q, k=k, article_type=article_type)


#ruta = ROOT / df.loc[0, "image_path"]
#print(df.loc[0, "articleType"], "|", df.loc[0, "productDisplayName"])

#for p in search_image(ruta, k=6):
#    print(f"{p.score:.3f}  {p.payload['articleType']:15s} {p.payload['productDisplayName']}")


ruta = ROOT / df.loc[0, "image_path"]          # el reloj Skagen
ids = lambda res: [p.id for p in res]

# --- Test 1 y 2: los casos límite de alpha ---
print("alpha=1 == search_image:",
      ids(search_combined(ruta, "black", alpha=1.0)) == ids(search_image(ruta)))
print("alpha=0 == search_text: ",
      ids(search_combined(ruta, "black", alpha=0.0)) == ids(search_text("black")))

# --- Consulta contradictoria: imagen de reloj + "running shoes" ---
for alpha in [0.0, 0.2, 0.5, 0.8, 1.0]:
    print(f"\n--- alpha={alpha} ---")
    for p in search_combined(ruta, "running shoes", alpha=alpha):
        print(f"{p.score:.3f}  {p.payload['articleType']:15s} {p.payload['productDisplayName']}")

# --- ¿Qué lado pesa más? ---
v_img = l2_normalize(embed_images([ruta]))[0]
v_txt = l2_normalize(embed_text(["running shoes"]))[0]
s_img = Xn @ v_img
s_txt = Xn @ v_txt
print("\nstd imagen:", round(s_img.std(), 3), "| rango:", round(s_img.min(), 3), round(s_img.max(), 3))
print("std texto: ", round(s_txt.std(), 3), "| rango:", round(s_txt.min(), 3), round(s_txt.max(), 3))

# --- ¿Cambia el ranking si renormalizo la mezcla? ---
q = 0.5 * v_img + 0.5 * v_txt
print("\nnorma de la mezcla:", round(float(np.linalg.norm(q)), 3))
q_norm = l2_normalize(q[None, :])[0]
print("mismos ids con y sin renormalizar:", ids(search_by_vector(q)) == ids(search_by_vector(q_norm)))



client.close()

