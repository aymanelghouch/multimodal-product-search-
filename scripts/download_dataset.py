from datasets import load_dataset
import pandas as pd

ds = load_dataset("ashraq/fashion-product-images-small", split="train")

print(ds)
print(ds.features)

df = ds.remove_columns(["image"]).to_pandas()


print(df.info())

print(df.isna().sum())


print(df["id"].is_unique, df["id"].nunique(), len(df))


print("\n--- categorías ---")
print(df["masterCategory"].value_counts())
print(df["subCategory"].value_counts().head(15))
print(df["articleType"].value_counts().head(15))



print("\n--- ejemplo de fila ---")
print(df.iloc[0])


# 3. Comprobar que la imagen se lee bien
img = ds[0]["image"]
print("\n--- imagen 0 ---")
print(img.size, img.mode)



print('Cuantos productos tiene Thisrts')
print(len(df[df['articleType'] == 'Tshirts']))

print(df['articleType'].value_counts().head(5).index) 

df_top_5 = df[df['articleType'].isin(df['articleType'].value_counts().head(5).index)]
print(df_top_5)



print('3 productos por categoria')
products_3 = df_top_5.groupby('articleType').sample(n=3, random_state=42)
print (products_3)