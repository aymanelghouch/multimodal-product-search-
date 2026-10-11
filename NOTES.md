# Notas de aprendizaje

Diario del proyecto: qué he hecho, qué he medido y qué he aprendido en cada fase.

---

## Fase 1: preparación de los datos

**Resultado:** 1.000 productos (25 `articleType` × 40) con imagen y metadatos, generados por `scripts/prepare_dataset.py`.

### Qué descubrí del dataset

- 44.072 productos de moda, 0 nulos y `id`# Notas de aprendizaje

Diario del proyecto: qué he hecho, qué he medido y qué he aprendido en cada fase.

---

## Fase 1: preparación de los datos

**Resultado:** 1.000 productos (25 `articleType` × 40) con imagen y metadatos, generados por `scripts/prepare_dataset.py`.

### Qué descubrí del dataset

- 44.072 productos de moda, 0 nulos y `id` único.
- Las imágenes miden 60×80 px. CLIP las escala a 224×224, así que llegan borrosas. Posible mejora futura: probar la versión de alta resolución y medir cuánto cambia.
- Categorías muy desbalanceadas: `Apparel` 21.361 frente a `Home` 1 y `Sporting Goods` 25.
- 13.271 nombres duplicados (~30 %). Por eso no puedo definir "resultado correcto" como "mismo nombre". [completar: qué tipo de duplicados vi, mismo producto en otros colores o ids distintos]

### Decisiones

- Muestra **estratificada por `articleType`** (25 tipos × 40) y no aleatoria pura: con azar saldrían ~480 de `Apparel` y casi nada del resto, y la evaluación no tendría sentido.
- `random_state=42` para que la selección sea aleatoria pero repetible.
- Columna `ds_idx` para volver de una fila del DataFrame a su imagen en el `Dataset` de Hugging Face.
- `to_pandas()` solo sobre las columnas sin imagen: de lo contrario carga las 44.072 imágenes en RAM.
- Validación con `assert` al final del script. Detectó 16 imágenes sobrantes de mis pruebas, que ya estaban en la carpeta.

### Errores que me encontré

- `pyarrow requires NumPy 2.0 or newer, found 1.26.0`: conflicto de versiones en el Python global. Solución: entorno virtual.
- `'DatasetDict' object has no attribute 'features'`: `load_dataset` sin `split` devuelve un diccionario de splits. Solución: `split="train"`.

---

## Fase 2: buscador visual básico con NumPy

**Resultado:** búsqueda por similitud coseno sobre histogramas de color (8 bins por canal, vector de 512 valores).

### Conceptos

- `PIL` da `size = (ancho, alto)` = (60, 80). NumPy da `shape = (filas, columnas, canales)` = (80, 60, 3). Las filas son el alto, por eso el orden se invierte.
- Con vectores de norma 1, la similitud coseno es un producto escalar, y todas las similitudes salen de una sola operación: `X @ X[i]`.
- Para excluir la propia consulta le pongo `-inf` a su score antes de ordenar.

### Qué medí

- El fondo blanco ocupa el 58,6 % de los píxeles de la imagen 0, y el cajón `(7, 7, 7)` domina el histograma.
- Normas de los vectores antes de normalizar: 0.62 a 0.80. Después: 1.
- Consulta 0 (_Skagen Men Black Watch_): scores entre 0.993 y 0.997. Resultados: 2 relojes, 2 mochilas y 1 bolso, todos negros sobre blanco.

### Qué aprendí

- Los scores casi iguales (0.993 a 0.997) significan que casi todos los productos se parecen casi igual a la consulta. El ranking depende de diferencias mínimas, porque el fondo blanco domina el vector.
- El histograma encuentra "negro sobre blanco", no el tipo de objeto.
- Quitar el fondo **empeoró** el resultado (0.226 → 0.165), al contrario de lo que esperaba. Hipótesis sin verificar: al anular el cajón blanco se pierde la información de los productos que son blancos y, al renormalizar, el ruido se amplifica.

### Pendiente (no hecho)

- Precision@5 desglosada por tipo de producto.
- Representación alternativa (imagen 16×16 aplanada).

---

## Fase 3: embeddings con CLIP

**Resultado:** embeddings de imagen y de texto con `openai/clip-vit-base-patch32`. Las 1.000 imágenes se guardan en `data/processed/clip_image_embeddings.npy` (sin normalizar, para poder cambiar de decisión sin recalcular). Búsqueda por texto funcionando.

### Conceptos y problemas

- Torch para este proyecto: un tensor se parece mucho a un array de NumPy. Solo hago inferencia, así que uso `model.eval()` y `torch.no_grad()`, y paso a NumPy con `.numpy()`.
- En mi versión de `transformers`, `get_text_features` y `get_image_features` **no devuelven un tensor**, sino un objeto `BaseModelOutputWithPooling`. Tiene dos campos:
  - `last_hidden_state`: un vector por token, forma `(n, tokens, 512)`. No sirve como embedding del texto entero.
  - `pooler_output`: un vector por texto o imagen, forma `(n, 512)`. Es el que uso.
- Para descubrir qué campo era cada uno miré `.keys()` y `.shape`, no el contenido.
- Errores típicos que me salieron: función sin `return` (devolvía `None`), `.shape()` con paréntesis (`shape` es un atributo), llamar a una función de lotes con un `str` en vez de una lista de rutas, `ModuleNotFoundError: No module named 'src'` al lanzar el script desde `scripts/` (se resuelve con `python -m scripts.nombre` desde la raíz).
- El modelo y el `processor` se cargan **una vez**, fuera de las funciones.
- Las imágenes se procesan por lotes de 32. El último lote puede ser más pequeño y el slice no da error. Comprobé que el vector de una imagen no cambia si va sola o dentro de un lote (`np.allclose`).

### Qué medí

- Normas de los vectores de CLIP antes de normalizar: unos 9.7 a 10.5. Después de `l2_normalize`: 1 (con `allclose`, no con `==`).
- Tiempo de calcular las 1.000 imágenes en CPU: [completar].

### Resultados

| Método                         | Precision@5 |
| ------------------------------ | ----------- |
| Azar (39/999)                  | 0.039       |
| Histograma de color (8 bins)   | 0.226       |
| Histograma sin el cajón blanco | 0.165       |
| **CLIP (imagen → imagen)**     | **0.781**   |

CLIP es 3,5 veces mejor que el histograma y unas 20 veces mejor que el azar, a pesar de las imágenes de 60×80.

### Búsqueda por texto (comprobación a ojo)

- `silver wrist watch`: los 5 resultados son relojes.
- `black running shoes`: los 5 son zapatos, casi todos negros, pero 2 son `Formal Shoes`. Hipótesis sin verificar: CLIP capta "negro" y "zapatos" mejor que "running".
- `blue denim shirt`: salen camisas y vaqueros (CLIP asocia "denim" con jeans).
- Scores texto-imagen entre 0.26 y 0.31, mucho más bajos que imagen-imagen. Los cinco primeros de una consulta quedan muy juntos (0.288 a 0.292), así que el orden entre ellos es poco fiable.
- No usé `top_k` de la fase 2 porque la consulta ya no está dentro de la matriz: no hay `query_idx` ni propio producto que excluir.
- Esto valida que `embed_text` está alineado con el espacio de las imágenes. La forma `(n, 512)` sola no lo probaba.

### Limitaciones de la medida (importantes)

- Precision@5 por `articleType` es **estricta**: un `Casual Shoes` devuelto para un `Sports Shoes` cuenta como error.
- Puede estar **inflada por casi-duplicados** (el mismo producto en varios colores, ver los 13.271 nombres duplicados). No he medido cuánto pesan en mi muestra.
- Las consultas de texto las he mirado a ojo, no tengo todavía métricas para texto.

---

## Siguiente: Fase 4, pipeline de embeddings

- Mover `embed_text` y `embed_images` a `src/embeddings.py`.
- Pipeline reanudable: guardar por lotes, no recalcular lo que ya existe, sobrevivir a un fallo a mitad. único.
- Las imágenes miden 60×80 px. CLIP las escalará a 224×224, así que llegan borrosas. Posible mejora futura: probar la versión de alta resolución y medir cuánto cambia.
- Categorías muy desbalanceadas: `Apparel` 21.361 frente a `Home` 1 y `Sporting Goods` 25.
- 13.271 nombres duplicados (~30 %). Por eso no puedo definir "resultado correcto" como "mismo nombre". [completar: qué tipo de duplicados vi, mismo producto en otros colores o ids distintos]

### Decisiones

- Muestra **estratificada por `articleType`** (25 tipos × 40) y no aleatoria pura: con azar saldrían ~480 de `Apparel` y casi nada del resto, y la evaluación no tendría sentido.
- `random_state=42` para que la selección sea aleatoria pero repetible.
- Columna `ds_idx` para volver de una fila del DataFrame a su imagen en el `Dataset` de Hugging Face.
- `to_pandas()` solo sobre las columnas sin imagen: de lo contrario carga las 44.072 imágenes en RAM.
- Validación con `assert` al final del script. Detectó 16 imágenes sobrantes de mis pruebas, que ya estaban en la carpeta.

### Errores que me encontré

- `pyarrow requires NumPy 2.0 or newer, found 1.26.0`: conflicto de versiones en el Python global. Solución: entorno virtual.
- `'DatasetDict' object has no attribute 'features'`: `load_dataset` sin `split` devuelve un diccionario de splits. Solución: `split="train"`.

---

## Fase 2: buscador visual básico con NumPy

**Resultado:** búsqueda por similitud coseno sobre histogramas de color (8 bins por canal, vector de 512 valores).

### Conceptos

- `PIL` da `size = (ancho, alto)` = (60, 80). NumPy da `shape = (filas, columnas, canales)` = (80, 60, 3). Las filas son el alto, por eso el orden se invierte.
- Con vectores de norma 1, la similitud coseno es un producto escalar, y todas las similitudes salen de una sola operación: `X @ X[i]`.
- Para excluir la propia consulta le pongo `-inf` a su score antes de ordenar.

### Qué medí

- El fondo blanco ocupa el 58,6 % de los píxeles de la imagen 0, y el cajón `(7, 7, 7)` domina el histograma.
- Normas de los vectores antes de normalizar: 0.62 a 0.80. Después: 1.
- Consulta 0 (_Skagen Men Black Watch_): scores entre 0.993 y 0.997. Resultados: 2 relojes, 2 mochilas y 1 bolso, todos negros sobre blanco.
- Construir la matriz de 1.000 histogramas tarda unos 0.5 a 0.8 s.

### Resultados (línea base)

| Método                         | Precision@5 |
| ------------------------------ | ----------- |
| Azar (39/999)                  | 0.039       |
| Histograma de color (8 bins)   | **0.226**   |
| Histograma sin el cajón blanco | 0.165       |

### Qué aprendí

- Los scores casi iguales (0.993 a 0.997) significan que casi todos los productos se parecen casi igual a la consulta. El ranking depende de diferencias mínimas, porque el fondo blanco domina el vector.
- El histograma encuentra "negro sobre blanco", no el tipo de objeto.
- Es 5,8 veces mejor que el azar, pero casi 4 de cada 5 resultados no son del mismo `articleType`.
- Quitar el fondo **empeoró** el resultado (0.226 → 0.165), al contrario de lo que esperaba. Hipótesis sin verificar: al anular el cajón blanco se pierde la información de los productos que son blancos y, al renormalizar, el ruido se amplifica.
- Precision@5 por `articleType` es una medida estricta: un `Casual Shoes` devuelto para un `Sports Shoes` cuenta como error. 0.226 es una cota inferior de la utilidad real.

### Pendiente (no hecho)

- Precision@5 desglosada por tipo de producto.
- Representación alternativa (imagen 16×16 aplanada).

---

## Fase 3: CLIP (en curso)

Objetivo: sustituir el histograma por embeddings de CLIP y superar la línea base de **0.226**. `l2_normalize`, `top_k` y `precision_at_k` se reutilizan sin cambios.
