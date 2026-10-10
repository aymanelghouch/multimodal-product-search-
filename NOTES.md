# Notas de aprendizaje

## Fase 1: datos

- Dataset: 44.072 productos de moda, 0 nulos, `id` único.
- Imágenes de 60×80 px: CLIP las escalará a 224×224, así que perderán detalle.
- Categorías muy desbalanceadas (Apparel 21k vs Home 1) → muestra estratificada por `articleType`.
- 30% de nombres duplicados (13.271): [lo que viste en los duplicados].
- `ds_idx` conecta el DataFrame con las imágenes en el Dataset de Hugging Face.
- Script idempotente: misma semilla, misma selección.
