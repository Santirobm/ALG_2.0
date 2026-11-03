# Informe Técnico Hito 3.3: Primera Validación con Embeddings Reales (D=64)

**Fecha:** 3 de Noviembre de 2026  
**Rama Git:** `hito3.3-realworld-validation`  
**Estado:** Circuito de datos y evaluación cerrado exitosamente. 100.00% IDSetRecall verificado.  
**Núcleos Plan 5A:** Congelados e intactos (`src/execution_plans.cpp`, `include/ieee754_bounds.h`).

---

## 1. Procedencia y Preparación del Corpus Real

Para cumplir con el requerimiento de validar sobre datos reales y no simulaciones sintéticas:
- **Dataset Fuente:** [`lsb/simplewiki2023-all-minilm-l6-v2-embedding`](https://huggingface.co/datasets/lsb/simplewiki2023-all-minilm-l6-v2-embedding).
  - Texto: Artículos de Simple English Wikipedia.
  - Modelo de Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensiones densas nativas).
  - Licencia: Creative Commons Attribution-ShareAlike (CC-BY-SA).
- **Protocolo de Proyección Controlada:**
  - Tamaño total extraído: 100.000 filas.
  - Partición estricta: 90.000 filas de corpus y 10.000 filas de consultas independientes (10% held-out).
  - Ajuste de PCA: `IncrementalPCA` entrenado **exclusivamente sobre las 90.000 filas de corpus**; las 10.000 consultas fueron transformadas con los componentes aprendidos sin contaminar la matriz de covarianza.
  - Varianza explicada ($D=64$): **58.28%**.
  - Normalización: vectores unitarios $L_2$ (`||x|| = 1.0`).
  - Manifiesto criptográfico: SHA-256 generado en [`data/hito3_3/D64/manifest.json`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/data/hito3_3/D64/manifest.json).

---

## 2. Adaptador C++ y Protocolo de Evaluación

- **Arnés de Ejecución:** [`benchmarks/realworld/run_realworld_harness.cpp`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/benchmarks/realworld/run_realworld_harness.cpp).
- **Lectura Directa:** Carga binaria directa de `corpus.f32` ($90.000 \times 64 \times 4$ bytes) y `queries.f32` ($10.000 \times 64 \times 4$ bytes).
- **Entorno de Medición:** Pinning estricto a P-Core 0, prioridad alta de proceso (`HIGH_PRIORITY_CLASS`), 1 hilo OpenMP forzado (`omp_set_num_threads(1)`), y evicción de 36 MB LLC en modo COLD.
- **Oráculo de Verificación:** Ground Truth independiente calculado en `double` ($binary64$) sobre exactamente el mismo conjunto de candidatos elegibles para cada consulta.

---

## 3. Resultados Experimentales Consolidados

Los datos corresponden a la ejecución canónica registrada en [`results/hito3_3/realworld_d64_summary.csv`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/results/hito3_3/realworld_d64_summary.csv):

| Selectividad | Top-$K$ | Modo de Caché | Latencia ALG (Media) | Latencia FAISS (Media) | Speedup ALG vs Flat | Recall IDSet | StrictRankMatch |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1.0%** | 10 | WARM_CACHE | **15.48 µs** | 143.65 µs | **9.28×** | **100.00%** | 100.00% |
| **1.0%** | 10 | COLD_LLC_FLUSH | **86.28 µs** | 189.97 µs | **2.20×** | **100.00%** | 100.00% |
| **1.0%** | 100 | WARM_CACHE | **25.88 µs** | 165.27 µs | **6.39×** | **100.00%** | 100.00% |
| **1.0%** | 100 | COLD_LLC_FLUSH | **94.90 µs** | 199.43 µs | **2.10×** | **100.00%** | 100.00% |
| **10.0%** | 10 | WARM_CACHE | **120.40 µs** | 206.91 µs | **1.72×** | **100.00%** | 100.00% |
| **10.0%** | 10 | COLD_LLC_FLUSH | **397.07 µs** | 393.42 µs | **0.99×** | **100.00%** | 100.00% |
| **10.0%** | 100 | WARM_CACHE | **131.16 µs** | 227.13 µs | **1.73×** | **100.00%** | 100.00% |
| **10.0%** | 100 | COLD_LLC_FLUSH | **411.49 µs** | 409.82 µs | **1.00×** | **100.00%** | 99.00% |
| **100.0%** (Ctrl) | 10 | WARM_CACHE | **1419.05 µs** | 1579.14 µs | **1.11×** | **100.00%** | 100.00% |
| **100.0%** (Ctrl) | 10 | COLD_LLC_FLUSH | **1598.48 µs** | 1776.68 µs | **1.11×** | **100.00%** | 100.00% |
| **100.0%** (Ctrl) | 100 | WARM_CACHE | **1497.77 µs** | 1684.67 µs | **1.12×** | **100.00%** | 100.00% |
| **100.0%** (Ctrl) | 100 | COLD_LLC_FLUSH | **1634.53 µs** | 1871.00 µs | **1.14×** | **100.00%** | 100.00% |

---

## 4. Análisis Arquitectónico y Hallazgos Clave

1. **Exactitud Rigurosa Certificada:**
   - En las 1.200 evaluaciones de consulta sobre datos reales, `IDSetRecall` fue del **100.000%** frente al oráculo `double`. No hubo un solo vecino más cercano falso positivo o falso negativo.
   - La coincidencia de rango (`StrictRankMatch`) fue del 100% en todas las condiciones salvo un empate numérico en Top-100 a selectividad del 10% (99.00%), debido a distancias idénticas dentro de la precisión de mantisa de $float32$.

2. **Rendimiento en Búsqueda Altamente Selectiva (1%):**
   - En consultas con alta selectividad (típico de filtrado por usuario, tenant o categoría específica con 900 candidatos elegibles de 90.000), ALG Plan 5A supera contundentemente a FAISS Flat: **$9.28×$ en WARM** ($15.48\text{ µs}$ vs. $143.65\text{ µs}$) y **$2.20×$ en COLD** ($86.28\text{ µs}$ vs. $189.97\text{ µs}$).

3. **Comportamiento en Selectividad Media (10%):**
   - En el régimen COLD con selectividad del 10% (9.000 candidatos elegibles dispersos entre 90.000 vectores), la distribución de embeddings de texto real presenta una dispersión multidireccional que reduce la poda geométrica esférica.
   - En este punto, ALG Plan 5A y FAISS Flat quedan en **paridad prácticamente exacta** (**$0.99×–1.00×$**, $397\text{ µs}$ vs. $393\text{ µs}$, diferencia inferior a $4\text{ µs}$).
   - Sin embargo, en cuanto el corpus reside en memoria caché (WARM), la organización contigua por bloques de ALG recupera una ventaja clara de **$1.72×–1.73×$** sobre FAISS Flat.

4. **Escaneo Completo (100% de Selectividad sin Filtro):**
   - Evaluando los 90.000 vectores completos sin ningún filtro, ALG Plan 5A mantiene una ventaja consistente de **$1.11×$ a $1.14×$** sobre FAISS Flat en ambos regímenes (COLD y WARM), confirmando la eficiencia superior de su kernel de barrido contiguo AVX2 frente al escaneo de FAISS.

---

## 5. Próximos Pasos en Hito 3.3

Habiendo validado y cerrado exitosamente el circuito de datos sobre embeddings reales en $D=64$:
1. **Extensión a la Batería Dimensional:** Adaptar el indexador y el ejecutor para evaluar las matrices ya preparadas en $D \in \{32, 128, 256\}$.
2. **Filtros Multicolumna:** Evaluar combinaciones complejas de metadatos (`tenant_id == X AND category == Y AND timestamp >= T`).
3. **Auditoría Repetida de Tiempos de Construcción:** Ejecutar las 5 repeticiones independientes con asignación fresca de memoria.
