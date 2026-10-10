from pathlib import Path
from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "data" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)


ds = load_dataset("ashraq/fashion-product-images-small", split="train")

df = ds.remove_columns(["image"]).to_pandas()

df["ds_idx"] = df.index


top5 = df['articleType'].value_counts().head(5).index
sel = (
    df[df['articleType'].isin(top5)]
    .groupby('articleType', group_keys=False)
    .sample(n=3, random_state=42)
)

for _, row in sel.iterrows():
    product_id = row["id"]
    img = ds[row["ds_idx"]]["image"]

    ruta = IMG_DIR / f"{product_id}.jpg"
    img.convert("RGB").save(ruta)
    print(f"Guardada {ruta.name}  ({row['articleType']})")

print("\nTotal archivos en carpeta:", len(list(IMG_DIR.glob("*.jpg"))))


