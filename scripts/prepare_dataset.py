from pathlib import Path

import pandas as pd
from datasets import load_dataset

# ---------- Configuración ----------
DATASET_NAME = "ashraq/fashion-product-images-small"   # el mismo de download_dataset.py
N_TYPES = 25
PER_TYPE = 40
SEED = 42

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "data" / "images"
OUT_CSV = ROOT / "data" / "processed" / "products.csv"


# ---------- 1. Cargar ----------
def load_metadata():
    ds = load_dataset(DATASET_NAME, split="train")
    df = ds.remove_columns(["image"]).to_pandas()
    df["ds_idx"] = df.index          # clave para volver a la imagen en ds
    return ds, df


# ---------- 2. Limpiar y comprobar calidad ----------
def clean(df):
    df = df.copy()
    df["productDisplayName"] = df["productDisplayName"].str.strip()

    vacios = (df["productDisplayName"] == "").sum()
    duplicados = df["productDisplayName"].duplicated().sum()
    print(f"Nombres vacíos: {vacios} | nombres duplicados: {duplicados}")

    df = df[df["productDisplayName"] != ""]
    df["year"] = df["year"].astype("Int64")
    return df


# ---------- 3. Selección estratificada ----------
def select_products(df):
    counts = df["articleType"].value_counts()
    top_types = counts.head(N_TYPES).index

    # sample(n=40) falla si algún grupo tiene menos de 40
    assert counts.loc[top_types].min() >= PER_TYPE, (
        f"Algún articleType tiene menos de {PER_TYPE} productos"
    )

    sel = (
        df[df["articleType"].isin(top_types)]
        .groupby("articleType", group_keys=False)
        .sample(n=PER_TYPE, random_state=SEED)
        .sort_values("ds_idx")
    )
    return sel


# ---------- 4. Guardar imágenes y metadatos ----------
def save_images(ds, sel):
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for _, row in sel.iterrows():
        ruta = IMG_DIR / f"{row['id']}.jpg"
        ds[int(row["ds_idx"])]["image"].convert("RGB").save(ruta)
        paths.append(ruta.relative_to(ROOT).as_posix())
    sel = sel.copy()
    sel["image_path"] = paths
    return sel


def save_csv(sel):
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    sel.to_csv(OUT_CSV, index=False)


# ---------- 5. Validar ----------
def validate():
    d = pd.read_csv(OUT_CSV)
    esperado = N_TYPES * PER_TYPE

    assert len(d) == esperado, f"Filas: {len(d)}, esperadas: {esperado}"
    assert d["id"].is_unique, "Hay ids duplicados"
    assert d["articleType"].nunique() == N_TYPES, "Número de tipos incorrecto"
    assert (d["articleType"].value_counts() == PER_TYPE).all(), "Grupos desiguales"
    assert d["productDisplayName"].notna().all(), "Hay nombres nulos"

    faltan = [p for p in d["image_path"] if not (ROOT / p).exists()]
    assert not faltan, f"Imágenes que faltan: {faltan[:5]}"

    n_jpg = len(list(IMG_DIR.glob("*.jpg")))
    assert n_jpg == esperado, f"Archivos en disco: {n_jpg}, esperados: {esperado}"

    print(f"OK: {len(d)} productos, {d['articleType'].nunique()} tipos, {n_jpg} imágenes")


# ---------- Main ----------
def main():
    ds, df = load_metadata()
    df = clean(df)
    sel = select_products(df)
    sel = save_images(ds, sel)
    save_csv(sel)
    validate()


if __name__ == "__main__":
    main()