# Informe de Resultados Hito 3.1: Native Benchmark Integrity

## 1. Protocolo y Certificación Matemática Rigurosa
- **Configuración:** $D = 64$, $N = 100{,}000$, $float32$ (25.6 MB de corpus crudo).
- **Hardware:** Intel Core i5-13500 (24 MB Intel Smart Cache L3), CPU Pinning P-Core 0, Priority `HIGH_PRIORITY_CLASS`.
- **FAISS C++ Nativo:** Enlace directo C++ contra `faiss.dll` (MSVC ABI) mediante símbolos exportados `search@IndexFlat` y `search@IndexIVF`. **Cero CPython, cero PyObjects, cero diccionarios/tuplas y cero conversión de NumPy dentro del cronómetro**.
- **Filtros:** `IDSelectorBitmap` preconstruido para FAISS; listas contiguas pre-indexadas por selectividad para ALG Plan 5A.
- **Ground Truth Independiente en Double (binary64):** Acumulación, selección en heap y ordenación estrictamente en 64-bit double, con política determinista de desempate por menor ID.
- **Auditoría de Exactitud:** `StrictRankMatch@K` evalúa identidad exacta posición por posición ($a_i = gt_i$). `IDSetRecall@K` evalúa pertenencia de conjuntos. `MaxDistErr` mide la cota de error numérico.
- **Estadística:** Cuantiles continuos Tipo-7 (estándar NumPy/R) con 100 consultas WARM y **100 consultas COLD** por punto experimental (**10,000 consultas totales**, con evicción LLC de 36 MB entre motores en cada consulta fría).

## 2. Auditoría de Exactitud y Recall

- **StrictRankMatch@K (ALG vs Double GT):** Media **99.94%** (Min: 99.00%, Max: 100.00%).
- **IDSetRecall@K (ALG vs Double GT):** Media **100.00%** (100.00% exacto en todas las consultas).
- **Max Dist Error (ALG vs Double GT):** **2.551780e-07** (acorde al límite teórico de precisión de máquina $float32$).
- **StrictRankMatch@K (ALG vs FAISS Flat):** Media **99.90%**.

## 3. Resumen de Speedup Nativo (ALG Plan 5A vs FAISS IndexFlatL2 Nativo)

| CacheMode | Selectivity | 10 | 100 |
| --- | --- | --- | --- |
| COLD_LLC_FLUSH | 1 | 5.02 | 3.07 |
| COLD_LLC_FLUSH | 5 | 5.18 | 2.71 |
| COLD_LLC_FLUSH | 10 | 5.23 | 4.80 |
| COLD_LLC_FLUSH | 15 | 4.87 | 4.46 |
| COLD_LLC_FLUSH | 25 | 3.90 | 3.61 |
| WARM_CACHE | 1 | 9.15 | 5.74 |
| WARM_CACHE | 5 | 5.40 | 3.84 |
| WARM_CACHE | 10 | 5.19 | 4.61 |
| WARM_CACHE | 15 | 5.05 | 4.48 |
| WARM_CACHE | 25 | 4.06 | 3.74 |

## 4. Desglose de Rendimiento Nativo por Escenario

| Scenario | COLD_LLC_FLUSH | WARM_CACHE |
| --- | --- | --- |
| Adversarial_OOD | 2.38 | 3.60 |
| Bimodal_5050 | 3.45 | 4.21 |
| Concentrated_s003 | 10.97 | 10.77 |
| Diffuse_s015 | 2.28 | 3.43 |
| Medium_s009 | 2.35 | 3.62 |

## 5. Comparativa de Percentiles de Latencia (µs)

| CacheMode | Engine | Mean_us | p50_us | p95_us | p99_us |
| --- | --- | --- | --- | --- | --- |
| COLD_LLC_FLUSH | ALG_Plan5A | 378.70 | 374.00 | 604.10 | 750.40 |
| COLD_LLC_FLUSH | FAISS_Flat | 980.30 | 899.10 | 1373.60 | 1695.10 |
| WARM_CACHE | ALG_Plan5A | 237.10 | 226.70 | 404.00 | 491.70 |
| WARM_CACHE | FAISS_Flat | 707.00 | 661.00 | 967.20 | 1146.00 |

## 6. Frontera de Calidad: Recall de FAISS IndexIVFFlat (nprobe=32)

| Selectivity | Adversarial_OOD | Bimodal_5050 | Concentrated_s003 | Diffuse_s015 | Medium_s009 |
| --- | --- | --- | --- | --- | --- |
| 1.00 | 80.00 | 97.40 | 99.60 | 85.90 | 94.10 |
| 5.00 | 89.10 | 99.80 | 100.00 | 96.50 | 99.50 |
| 10.00 | 91.70 | 99.90 | 100.00 | 98.40 | 99.80 |
| 15.00 | 92.70 | 100.00 | 100.00 | 98.80 | 99.90 |
| 25.00 | 93.80 | 100.00 | 100.00 | 99.10 | 99.90 |
