# Notas de aprendizaje

Diario del proyecto: qué he hecho, qué he medido y qué he aprendido en cada fase.

---

## Fase 1: preparación de los datos

**Resultado:** 1.000 productos (25 `articleType` × 40) con imagen y metadatos, generados por `scripts/prepare_dataset.py`.

### Qué descubrí del dataset

- 44.072 productos de moda, 0 nulos y `id` único.
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
