# Informe de Investigación Hito 3.2: Curva de Break-Even Selectivity

## 1. Metodología Experimental
- **Corpus:** $N = 100{,}000$ vectores, $D = 64$ ($float32$), semilla fija 8, distribución $\sigma=0.09$.
- **Grilla de Selectividad:** 16 puntos desde 0% (control vacío) hasta 100% (control scan completo):
  `[0.0%, 0.05%, 0.1%, 0.25%, 0.5%, 1.0%, 2.0%, 5.0%, 10.0%, 15.0%, 20.0%, 30.0%, 40.0%, 50.0%, 75.0%, 100.0%]`.
- **Motores Comparados:** ALG Plan 5A (poda por partición en espacio original) vs. Meta FAISS 1.15.1 IndexFlatL2 (+ `IDSelectorBitmap` nativo vía C Shim).
- **Protocolo:** Pinning a P-Core 0, prioridad `HIGH_PRIORITY_CLASS`, 1 hilo OpenMP, evicción estricta de 36 MB LLC en modo COLD.
- **Exactitud:** 100.00% de `IDSetRecall` frente al oráculo Ground Truth en `double` ($binary64$) en todas las selectividades.

## 2. Curva de Speedup por Selectividad (ALG vs FAISS Flat)

| CacheMode | Selectivity | 10 | 100 |
| --- | --- | --- | --- |
| COLD_LLC_FLUSH | 0.00 | 11.15 | 10.81 |
| COLD_LLC_FLUSH | 0.05 | 7.65 | 7.45 |
| COLD_LLC_FLUSH | 0.10 | 7.21 | 5.56 |
| COLD_LLC_FLUSH | 0.25 | 5.65 | 4.58 |
| COLD_LLC_FLUSH | 0.50 | 4.72 | 3.72 |
| COLD_LLC_FLUSH | 1.00 | 4.23 | 3.49 |
| COLD_LLC_FLUSH | 2.00 | 3.06 | 2.96 |
| COLD_LLC_FLUSH | 5.00 | 2.62 | 2.56 |
| COLD_LLC_FLUSH | 10.00 | 2.35 | 2.49 |
| COLD_LLC_FLUSH | 15.00 | 2.08 | 2.06 |
| COLD_LLC_FLUSH | 20.00 | 1.69 | 1.68 |
| COLD_LLC_FLUSH | 30.00 | 1.46 | 1.45 |
| COLD_LLC_FLUSH | 40.00 | 1.34 | 1.34 |
| COLD_LLC_FLUSH | 50.00 | 1.27 | 1.29 |
| COLD_LLC_FLUSH | 75.00 | 1.15 | 1.16 |
| COLD_LLC_FLUSH | 100.00 | 1.07 | 1.07 |
| WARM_CACHE | 0.00 | 19.71 | 18.65 |
| WARM_CACHE | 0.05 | 15.04 | 14.53 |
| WARM_CACHE | 0.10 | 14.77 | 10.70 |
| WARM_CACHE | 0.25 | 13.41 | 8.96 |
| WARM_CACHE | 0.50 | 11.43 | 7.59 |
| WARM_CACHE | 1.00 | 8.46 | 6.61 |
| WARM_CACHE | 2.00 | 5.81 | 5.25 |
| WARM_CACHE | 5.00 | 4.14 | 3.83 |
| WARM_CACHE | 10.00 | 2.93 | 3.01 |
| WARM_CACHE | 15.00 | 2.47 | 2.66 |
| WARM_CACHE | 20.00 | 2.22 | 2.25 |
| WARM_CACHE | 30.00 | 1.71 | 1.83 |
| WARM_CACHE | 40.00 | 1.61 | 1.67 |
| WARM_CACHE | 50.00 | 1.55 | 1.49 |
| WARM_CACHE | 75.00 | 1.31 | 1.33 |
| WARM_CACHE | 100.00 | 1.05 | 1.15 |

## 3. Dinámica Interna de la Poda Geométrica

| Selectivity /  | Avg_Dists_Eval / COLD_LLC_FLUSH | Avg_Dists_Eval / WARM_CACHE | Eligible_Total / COLD_LLC_FLUSH | Eligible_Total / WARM_CACHE | Parts_Visited_Avg / COLD_LLC_FLUSH | Parts_Visited_Avg / WARM_CACHE | Pruning_Ratio / COLD_LLC_FLUSH | Pruning_Ratio / WARM_CACHE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 100.00 | 100.00 |
| 0.05 | 37.00 | 37.00 | 37.00 | 37.00 | 31.00 | 31.00 | 0.00 | 0.00 |
| 0.10 | 101.00 | 101.00 | 101.00 | 101.00 | 61.00 | 61.00 | 0.00 | 0.00 |
| 0.25 | 276.00 | 276.00 | 276.00 | 276.00 | 94.00 | 94.00 | 0.00 | 0.00 |
| 0.50 | 510.00 | 510.00 | 510.00 | 510.00 | 107.00 | 107.00 | 0.00 | 0.00 |
| 1.00 | 996.00 | 996.00 | 996.00 | 996.00 | 112.00 | 112.00 | 0.00 | 0.00 |
| 2.00 | 2037.00 | 2036.70 | 2037.00 | 2037.00 | 117.00 | 117.00 | 0.00 | 0.00 |
| 5.00 | 5038.80 | 5038.50 | 5041.00 | 5041.00 | 120.80 | 120.80 | 0.00 | 0.10 |
| 10.00 | 10109.50 | 10102.60 | 10120.00 | 10120.00 | 120.60 | 120.60 | 0.10 | 0.20 |
| 15.00 | 15045.40 | 15035.10 | 15077.00 | 15077.00 | 121.70 | 121.70 | 0.20 | 0.30 |
| 20.00 | 20030.00 | 20012.10 | 20090.00 | 20090.00 | 123.40 | 123.30 | 0.30 | 0.40 |
| 30.00 | 29897.10 | 29846.60 | 30020.00 | 30020.00 | 123.10 | 123.00 | 0.40 | 0.60 |
| 40.00 | 39757.00 | 39738.90 | 40003.00 | 40003.00 | 122.70 | 122.80 | 0.60 | 0.70 |
| 50.00 | 49514.30 | 49479.40 | 49903.00 | 49903.00 | 122.50 | 122.50 | 0.80 | 0.80 |
| 75.00 | 74185.40 | 74215.80 | 75087.00 | 75087.00 | 121.60 | 121.80 | 1.20 | 1.20 |
| 100.00 | 98411.50 | 98657.10 | 100000.00 | 100000.00 | 121.10 | 121.40 | 1.60 | 1.30 |

## 4. Análisis del Break-Even Físico y Arquitectónico
A continuación se detallan los regímenes donde la poda geométrica supera al escaneo plano continuo y el umbral de cruce:

