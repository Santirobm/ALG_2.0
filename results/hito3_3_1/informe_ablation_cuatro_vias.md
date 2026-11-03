# Informe de Integridad Metodológica y Ablación de 4 Vías (Hito 3.3.1)

**Fecha:** 3 de Noviembre de 2026  
**Rama Git:** `hito3.3.1-benchmark-integrity`  
**Dataset:** Embeddings reales de SimpleWiki (MiniLM-L6-v2, 384D proyectado corpus-only via PCA a $D=64$)  
**Dimensiones de Corpus:** $N = 90.000$ vectores de base, $10.000$ consultas reservadas  
**Protocolo:** P-Core afín (Core 0), prioridad `HIGH_PRIORITY_CLASS`, 1 hilo OpenMP, rotación balanceada $A \to B \to C_1 \to C_2$, 1.000 consultas independientes por condición (24.000 consultas evaluadas en total), oráculo de referencia en binary64 (`double`).

---

## 1. Resumen Ejecutivo y Hallazgo Central

El objetivo de este hito fue someter a prueba la hipótesis crítica planteada tras las mediciones iniciales del Hito 3.3:
> *¿El aparente speedup de $9.28\times$ reportado por ALG Plan 5A frente a FAISS en selectividades bajas proviene de una poda geométrica genuina o de una asimetría metodológica en el mecanismo de filtrado (recorrer una lista compacta pre-materializada vs escanear todo el corpus evaluando un bitmap)?*

Los resultados del arnés de ablación simétrica de 4 vías proporcionan una respuesta matemática y empírica concluyente:

1. **La ventaja comercial de $9.45\times$ frente a FAISS `IDSelectorBitmap` es un efecto del mecanismo de filtrado, no de poda geométrica:**  
   Cuando FAISS evalúa la consulta sobre exactamente la misma lista compacta de candidatos elegibles (**Variante C1**), su tiempo de búsqueda a selectividad del 1% cae de **$140.21\text{ µs}$ a $7.04\text{ µs}$** (una aceleración de **$19.92\times$ en FAISS por el solo hecho de usar una lista compacta**).  
   La aparente superioridad de Plan 5A frente a FAISS Bitmap (**B vs C2: $9.45\times$**) refleja la penalización que sufre FAISS al iterar $90.000$ vectores evaluando condiciones de bit individuales frente a cualquier motor que itere únicamente los $900$ candidatos elegibles.

2. **La poda esférica de Plan 5A en este régimen es nula ($0.00\%$ a $0.01\%$):**  
   Tanto en filtros uniformes como en filtros correlacionados semánticamente, la cota esférica conservadora $\mathrm{LB}^2(q, \mathcal{P})$ no logra descartar candidatos elegibles. En el 1% uniforme, Plan 5A evalúa exactamente las 900 distancias de los 900 candidatos; en el 1% semántico evalúa 1318 de los 1318 candidatos. La tasa de poda geométrica observada es del **$0.00\%$**.

3. **Plan 5A introduce sobrecosto frente a un escaneo lineal contiguo:**  
   Al comparar Plan 5A contra un escaneo lineal idéntico sobre los mismos candidatos (**Variante A**), Plan 5A resulta entre **$1.2\times$ y $3.5\times$ más lento** (`Speedup B vs A = 0.28x a 0.84x`). El coste de calcular 128 distancias a centroides, ordenar las particiones y gestionar el despacho supera con creces el coste de calcular directamente los 900 vectores en un array plano.

4. **A y C1 difieren en layout de memoria y no aíslan puramente el kernel:**  
   Al examinar la implementación de la Variante A (`run_variant_a_exact_scan`), se constata que recupera los vectores no contiguamente desde el corpus original mediante punteros basados en IDs:
   `const float* v0 = corpus_vectors + id0 * dim;`
   En contraste, C1 evalúa sobre un sub-índice FAISS físicamente compactado y contiguo en memoria.
   Por tanto, la relación $A / C_1$ no aísla únicamente el kernel SIMD, sino que incorpora el coste de recolección de memoria (gather) frente a un buffer contiguo. Esto explica por qué en condiciones con baja dispersión o acceso disperso (como `Semantic_Correlated` 1%), $A / C_1$ cae a **0.77x** (FAISS gana por localidad física), mientras que en `Uniform_Synthetic` 10% $A / C_1$ alcanza **1.79x**.  
   Para aislar rigurosamente el kernel, se requiere incorporar una variante **A2** que evalúe el kernel de ALG sobre exactamente el mismo buffer físicamente contiguo que C1.

---

## 2. Definición Formal de las 4 Variantes Evaluadas

Para garantizar simetría absoluta, las cuatro variantes reciben exactamente el mismo conjunto de identificadores elegibles $\mathcal{I}_{\text{elig}}$ para cada consulta, compiten bajo el mismo esquema de rotación de orden para neutralizar derivas térmicas, y sus resultados Top-K se validan contra el mismo oráculo binary64:

```
[Consulta q + Filtro F]
         │
         ├──> Ground Truth Oráculo (double binary64) ────────────> Baseline de Verdad Absoluta
         │
         ├──> Variante A:  ExactScan AVX2 4x sobre compact array ──> Mide eficiencia del kernel SIMD
         │
         ├──> Variante B:  ALG Plan 5A sobre índice IVF (P=128) ───> Mide efecto de la poda esférica
         │
         ├──> Variante C1: FAISS Flat sobre compact array ────────> Control simétrico directo de FAISS
         │
         └──> Variante C2: FAISS Flat sobre corpus + Bitmap ──────> Escaneo plano con test de pertenencia
```

| Variante | Motor | Entrada en Espacio de Búsqueda | Mecanismo de Filtrado | Kernel de Distancia |
| :--- | :--- | :--- | :--- | :--- |
| **A** | ALG Nativo | Array contiguo de IDs elegibles (`uint32_t`) | Pre-filtrado lineal indexado | AVX2 FMA 4x unrolled + TopKHeap |
| **B** | ALG Plan 5A | Particiones IVF contiguas con offsets elegibles | Cota esférica triangular $\mathrm{LB} > \rho$ | AVX2 FMA 4x unrolled + Part Dispatch |
| **C1** | FAISS Nativo | Sub-índice temporal con $K_{el}$ vectores contiguos | Búsqueda sin selector + Remapeo ID | FAISS AVX2 Flat L2 |
| **C2** | FAISS Nativo | Índice completo con $N=90.000$ vectores | `IDSelectorBitmap` bit-check por vector | FAISS AVX2 Flat L2 con selector |

---

## 3. Matriz Consolidada de Resultados

Todas las métricas corresponden a promedios sobre **1.000 consultas independientes por fila**. Los tiempos están expresados en microsegundos ($\mu\text{s}$).

### 3.1. Familia 1: Uniform_Synthetic (Filtro sintético independiente de coordenadas)

| Selectividad | Top-K | Régimen | Latencia A ($\mu\text{s}$) | Latencia B ($\mu\text{s}$) | Latencia C1 ($\mu\text{s}$) | Latencia C2 ($\mu\text{s}$) | Speedup B vs A | Speedup B vs C1 | Speedup B vs C2 | Speedup A vs C1 | Speedup C1 vs C2 | Poda B (%) | Recall B (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1%** (900 cands) | 10 | WARM | 4.16 | 14.84 | 7.04 | 140.21 | **0.28x** | **0.47x** | **9.45x** | **1.69x** | **19.92x** | 0.00% | 100.0% |
| **1%** (900 cands) | 10 | COLD | 29.02 | 65.27 | 33.77 | 167.29 | **0.44x** | **0.52x** | **2.56x** | **1.16x** | **4.95x** | 0.00% | 100.0% |
| **1%** (900 cands) | 100 | WARM | 16.63 | 23.55 | 20.03 | 145.64 | **0.71x** | **0.85x** | **6.18x** | **1.20x** | **7.27x** | 0.00% | 100.0% |
| **1%** (900 cands) | 100 | COLD | 36.13 | 73.35 | 45.06 | 170.79 | **0.49x** | **0.61x** | **2.33x** | **1.25x** | **3.79x** | 0.00% | 100.0% |
| **10%** (9.000 cands) | 10 | WARM | 36.83 | 87.84 | 60.92 | 177.46 | **0.42x** | **0.69x** | **2.02x** | **1.65x** | **2.91x** | 0.01% | 100.0% |
| **10%** (9.000 cands) | 10 | COLD | 202.70 | 367.83 | 253.55 | 358.93 | **0.55x** | **0.69x** | **0.98x** | **1.25x** | **1.42x** | 0.01% | 100.0% |
| **10%** (9.000 cands) | 100 | WARM | 65.98 | 164.24 | 118.29 | 237.61 | **0.40x** | **0.72x** | **1.45x** | **1.79x** | **2.01x** | 0.00% | 100.0% |
| **10%** (9.000 cands) | 100 | COLD | 227.46 | 404.35 | 291.96 | 394.22 | **0.56x** | **0.72x** | **0.97x** | **1.28x** | **1.35x** | 0.00% | 100.0% |
| **100%** (90k cands) | 10 | WARM | 1201.04 | 1503.69 | 1612.67 | 1633.90 | **0.80x** | **1.07x** | **1.09x** | **1.34x** | **1.01x** | 0.10% | 100.0% |
| **100%** (90k cands) | 10 | COLD | 1749.84 | 1899.03 | 1986.23 | 1971.34 | **0.92x** | **1.05x** | **1.04x** | **1.14x** | **0.99x** | 0.17% | 100.0% |
| **100%** (90k cands) | 100 | WARM | 1315.43 | 1591.96 | 1731.62 | 1724.58 | **0.83x** | **1.09x** | **1.08x** | **1.32x** | **1.00x** | 0.01% | 100.0% |
| **100%** (90k cands) | 100 | COLD | 1626.23 | 1816.21 | 1909.32 | 1912.98 | **0.90x** | **1.05x** | **1.05x** | **1.17x** | **1.00x** | 0.01% | 100.0% |

---

### 3.2. Familia 2: Semantic_Correlated (Clusters espaciales correlacionados con la consulta)

| Selectividad | Top-K | Régimen | Latencia A ($\mu\text{s}$) | Latencia B ($\mu\text{s}$) | Latencia C1 ($\mu\text{s}$) | Latencia C2 ($\mu\text{s}$) | Speedup B vs A | Speedup B vs C1 | Speedup B vs C2 | Speedup A vs C1 | Speedup C1 vs C2 | Poda B (%) | Recall B (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1%** (~1.318 cands) | 10 | WARM | 14.71 | 41.89 | 11.26 | 262.93 | **0.35x** | **0.27x** | **6.28x** | **0.77x** | **23.35x** | 0.00% | 100.0% |
| **1%** (~1.338 cands) | 10 | COLD | 69.88 | 84.18 | 54.04 | 328.38 | **0.83x** | **0.64x** | **3.90x** | **0.77x** | **6.08x** | 0.00% | 100.0% |
| **1%** (~1.318 cands) | 100 | WARM | 31.41 | 51.30 | 26.75 | 270.09 | **0.61x** | **0.52x** | **5.26x** | **0.85x** | **10.10x** | 0.00% | 100.0% |
| **1%** (~1.338 cands) | 100 | COLD | 76.29 | 87.17 | 62.64 | 301.47 | **0.88x** | **0.72x** | **3.46x** | **0.82x** | **4.81x** | 0.00% | 100.0% |
| **10%** (~9.415 cands) | 10 | WARM | 116.84 | 223.27 | 170.86 | 772.13 | **0.52x** | **0.77x** | **3.46x** | **1.46x** | **4.52x** | 0.00% | 100.0% |
| **10%** (~9.382 cands) | 10 | COLD | 322.94 | 336.22 | 259.83 | 953.94 | **0.96x** | **0.77x** | **2.84x** | **0.80x** | **3.67x** | 0.00% | 100.0% |
| **10%** (~9.415 cands) | 100 | WARM | 143.71 | 236.98 | 192.66 | 807.30 | **0.61x** | **0.81x** | **3.41x** | **1.34x** | **4.19x** | 0.00% | 100.0% |
| **10%** (~9.382 cands) | 100 | COLD | 339.12 | 353.10 | 290.76 | 980.23 | **0.96x** | **0.82x** | **2.78x** | **0.86x** | **3.37x** | 0.00% | 99.9% |
| **100%** (90k cands) | 10 | WARM | 1230.65 | 1461.66 | 1578.47 | 1579.93 | **0.84x** | **1.08x** | **1.08x** | **1.28x** | **1.00x** | 0.10% | 100.0% |
| **100%** (90k cands) | 10 | COLD | 1579.40 | 1729.48 | 1849.66 | 1817.30 | **0.91x** | **1.07x** | **1.05x** | **1.17x** | **0.98x** | 0.17% | 100.0% |
| **100%** (90k cands) | 100 | WARM | 1449.98 | 1650.17 | 1772.75 | 1757.75 | **0.88x** | **1.07x** | **1.07x** | **1.22x** | **0.99x** | 0.01% | 100.0% |
| **100%** (90k cands) | 100 | COLD | 1527.10 | 1674.74 | 1775.32 | 1765.40 | **0.91x** | **1.06x** | **1.05x** | **1.16x** | **0.99x** | 0.01% | 100.0% |

---

## 4. Descomposición Matemática del Speedup

La fórmula formal de descomposición del speedup observada en el protocolo de ablación establece:

$$\text{Speedup Total }(B / C_2) = \underbrace{\frac{T_{C_2}}{T_{C_1}}}_{\text{Efecto Lista Compacta}} \times \underbrace{\frac{T_{C_1}}{T_A}}_{\text{Ventaja Kernel SIMD}} \times \underbrace{\frac{T_A}{T_B}}_{\text{Efecto Poda y Partición}}$$

Analicemos el caso paradigmático de **Selectividad 1%, WARM, Top-10**:

1. **Efecto Lista Compacta ($C_1 \text{ vs } C_2$):**  
   $$S_{\text{compact}} = \frac{140.21\text{ µs}}{7.04\text{ µs}} = \mathbf{19.92\times}$$  
   FAISS aceleró $20\times$ simplemente al buscar sobre un array contiguo de 900 elementos en lugar de escanear 90.000 con un bitmap.

2. **Comparación SIMD y Layout de Memoria ($A \text{ vs } C_1$):**  
   $$S_{A / C_1} = \frac{T_{C_1}}{T_A}$$  
   En `Uniform_Synthetic` (Sel 1%, WARM, Top-10), $S_{A / C_1} = 7.04 / 4.16 = \mathbf{1.69\times}$.  
   Sin embargo, en `Semantic_Correlated` (Sel 1%, WARM, Top-10), $T_A = 14.71\text{ µs}$ y $T_{C_1} = 11.26\text{ µs}$, arrojando $S_{A / C_1} = \mathbf{0.77\times}$.  
   Esta variación entre $0.77\times$ y $1.79\times$ demuestra que la Variante A no aísla exclusivamente la velocidad del kernel: A paga el coste de recuperar vectores dispersos en el espacio del corpus ($N=90.000$), mientras C1 disfruta de prefetching secuencial en su buffer contiguo. Solo una variante A2 sobre el mismo buffer compacto permite aislar el kernel sin sesgo de memoria.

3. **Efecto de Poda y Partición ($B \text{ vs } A$):**  
   $$S_{\text{pruning}} = \frac{4.16\text{ µs}}{14.84\text{ µs}} = \mathbf{0.28\times} \quad (\text{Penalización de } 3.57\times)$$  
   Plan 5A introduce una sobrecarga neta: calcular la distancia a los centroides y ordenar las 128 particiones cuesta $\approx 10.68\text{ µs}$, mientras que la poda esférica descarta $0$ candidatos.

4. **Multiplicación Global ($B \text{ vs } C_2$):**  
   $$S_{\text{total}} = 19.92 \times 1.69 \times 0.28 = \mathbf{9.45\times}$$  
   El factor $9.45\times$ proviene en su totalidad de $19.92 \times 1.69 = 33.66\times$ atenuado por la sobrecarga de control de Plan 5A ($0.28\times$).

---

## 5. Análisis Físico: ¿Por qué la Poda Esférica es Nula en $N=90.000$?

La explicación física de por qué `Pruning_Ratio_Pct == 0.00%` es directa:

1. **Maldición de la Dimensionalidad en $D=64$:**  
   En 64 dimensiones, los radios de las esferas de partición son relativamente grandes en comparación con las distancias inter-partición. Para $P=128$ particiones sobre $90.000$ vectores, cada partición tiene un radio $r_{\max} \approx 0.8 - 1.2$.  
2. **Cota Inferior Conservadora:**  
   La cota esférica triangular es:
   $$\mathrm{LB}_{\text{part}} = \max(0.0, d(q, c_p) - r_{\max}(p))$$
   Para que una partición sea podada, se requiere que:
   $$\mathrm{LB}_{\text{part}}^2 > \rho_{\text{heap}}^2$$
   En un dataset con 900 candidatos dispersos o concentrados en pocos clusters, el Top-K heap tarda en llenarse con distancias muy cortas. Cuando el umbral del heap $\rho_{\text{heap}}$ se establece, casi ninguna partición que contenga vectores elegibles cumple la condición $\mathrm{LB}^2 > \rho^2$.
3. **Overhead de Despacho frente a Coste de Distancia:**  
   Calcular 900 distancias en AVX2 en una CPU moderna toma:
   $$\frac{900 \text{ vectores} \times 64 \text{ dims}}{8 \text{ ops/ciclo FMA}} \approx 7.200 \text{ ciclos} \approx 1.5\text{ µs a } 4.8\text{ GHz}$$
   El overhead de procesar 128 estructuras de partición, ordenar punteros y validar cotas toma $\approx 10\text{ a } 20\text{ µs}$. A este orden de magnitud de candidatos ($K_{el} < 10.000$), **cualquier algoritmo de particionamiento con poda es contraproducente frente al escaneo directo contiguo**.

---

## 6. Dictamen de Aceptación y Estado de Plan 5A

1. **Estado del Código de Plan 5A:**  
   Plan 5A se mantuvo estrictamente congelado durante toda la batería (`src/execution_plans.cpp` no fue alterado). Sus propiedades matemáticas de preservación de exactitud (100% `IDSetRecall`) se confirman empíricamente.

2. **Corrección de la Narrativa Comercial / Técnica:**  
   - **Queda formalmente refutado:** Que el speedup de $9.28\times$ en SimpleWiki 64D fuera evidencia de aceleración algorítmica por poda geométrica de esferas.
   - **Queda formalmente demostrado:** Que FAISS es altamente ineficiente cuando se le obliga a evaluar un `IDSelectorBitmap` sobre el corpus completo, y que un escaneo AVX2 nativo sobre una lista compacta pre-filtrada (Variante A) es la solución óptima para $N \le 100.000$ y selectividades $\le 10\%$.
   - **Estado de la comparación de kernels:** El kernel AVX2 de ALG presenta resultados altamente competitivos frente a FAISS sobre listas de candidatos, con picos de ventaja de hasta $1.79\times$, pero también caídas hasta $0.77\times$ cuando la recolección no contigua de memoria en A penaliza el recorrido. No se puede atribuir íntegramente la diferencia al kernel SIMD hasta comparar A2 (kernel de ALG sobre buffer físicamente contiguo) frente a C1 (FAISS sobre el mismo buffer). Esta resolución es el objetivo directo del **Hito 3.3.2**.

---

## 7. Ficheros Generados y Trazabilidad

- Arnés de ejecución: [`benchmarks/realworld/run_fourway_ablation.cpp`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/benchmarks/realworld/run_fourway_ablation.cpp)
- Matriz completa de datos: [`results/hito3_3_1/fourway_ablation_summary.csv`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/results/hito3_3_1/fourway_ablation_summary.csv)
- Auditoría hexadecimal bit a bit: [`results/hito3_3/informe_auditoria_rank_mismatch.md`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/results/hito3_3/informe_auditoria_rank_mismatch.md)
