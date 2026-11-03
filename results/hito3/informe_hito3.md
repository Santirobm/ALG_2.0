# Informe de Resultados Hito 3: ALG-Flat Plan 5A vs Meta FAISS 1.15.1

## 1. Protocolo y Certificación Matemática
- **Configuración:** $D = 64$, $N = 100{,}000$, $float32$ (25.6 MB de corpus crudo).
- **Hardware:** Intel Core i5-13500 (24 MB Intel Smart Cache L3), CPU Pinning P-Core 0, Priority `HIGH_PRIORITY_CLASS`.
- **FAISS:** Versión 1.15.1 con `omp_set_num_threads(1)` estricto (OpenMP de 1 hilo), invocación directa vía CPython C API en memoria sin overhead de I/O.
- **Filtros:** `IDSelectorBitmap` preconstruido para FAISS; listas contiguas pre-indexadas para ALG Plan 5A.
- **Cache:** 5,000 consultas totales divididas en `WARM_CACHE` y `COLD_LLC_FLUSH` (evicción de 36 MB de memoria de purga entre cada motor).
- **Orden Intercalado Balanceado:** `Round 1: ALG -> Flat -> IVF`, `Round 2: Flat -> IVF -> ALG`, `Round 3: IVF -> ALG -> Flat`.
- **Exactitud (RankMatch@K):** **100.000%** exacto frente a FAISS IndexFlatL2 y Ground Truth independiente en `double` (64-bit).

## 2. Resumen de Speedup (ALG Plan 5A vs FAISS IndexFlatL2)

| CacheMode | Selectivity | 10 | 100 |
| --- | --- | --- | --- |
| COLD_LLC_FLUSH | 1 | 5.50 | 3.31 |
| COLD_LLC_FLUSH | 5 | 4.84 | 2.82 |
| COLD_LLC_FLUSH | 10 | 4.27 | 3.80 |
| COLD_LLC_FLUSH | 15 | 3.74 | 3.59 |
| COLD_LLC_FLUSH | 25 | 3.26 | 3.07 |
| WARM_CACHE | 1 | 9.03 | 5.69 |
| WARM_CACHE | 5 | 4.77 | 3.63 |
| WARM_CACHE | 10 | 5.17 | 4.49 |
| WARM_CACHE | 15 | 4.94 | 4.38 |
| WARM_CACHE | 25 | 4.01 | 3.70 |

## 3. Desglose de Rendimiento por Escenario

| Scenario | COLD_LLC_FLUSH | WARM_CACHE |
| --- | --- | --- |
| Adversarial_OOD | 2.51 | 3.48 |
| Bimodal_5050 | 3.60 | 4.13 |
| Concentrated_s003 | 8.07 | 10.32 |
| Diffuse_s015 | 2.36 | 3.37 |
| Medium_s009 | 2.55 | 3.61 |

## 4. Frontera de Calidad: Pérdida Catastrófica de Recall en FAISS IndexIVFFlat (nprobe=32)

Aunque FAISS IVFFlat es más rápido en latencia bruta, su recall colapsa bajo filtrado arbitrario predice:

| Selectivity | Adversarial_OOD | Bimodal_5050 | Concentrated_s003 | Diffuse_s015 | Medium_s009 |
| --- | --- | --- | --- | --- | --- |
| 1.00 | 11.90 | 28.00 | 38.90 | 27.90 | 36.20 |
| 5.00 | 18.50 | 44.00 | 61.00 | 45.10 | 58.80 |
| 10.00 | 22.20 | 51.10 | 68.60 | 53.70 | 66.70 |
| 15.00 | 23.80 | 53.00 | 68.60 | 57.40 | 66.60 |
| 25.00 | 24.60 | 55.20 | 69.20 | 58.60 | 66.60 |
