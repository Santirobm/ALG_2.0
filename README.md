# ALG-Flat (ALG 2.0) — Adaptive Geometric Layout Engine

**Exact vector search through physical partitioning, memory layout optimization, certified geometric pruning, and SIMD execution.**

---

## 🎯 Motivación y Reformulación Científica

Tras concluir el Hito 9 del proyecto original (`ALG`), demostramos empíricamente que en regímenes de dimensionalidad moderada ($N = 100.000$, $D = 64$, selectividades del $10\%$ al $25\%$), un escaneo contiguo vectorizado (`ExactScan`) supera sistemáticamente a las arquitecturas basadas en grafos (`HNSW`) y a los despachadores híbridos dinámicos bajo cargas fuera de distribución (*OOD*).

**ALG-Flat (`ALG 2.0`)** parte de una pregunta diferente:
> *¿Cuánto trabajo exacto podemos ahorrar mediante organización física de memoria, paralelismo SIMD y cotas geométricas matemáticamente certificadas, sin sacrificar jamás la exactitud del resultado?*

**ALG-Flat** separa y evalúa incrementalmente cuatro ejes físicos en `float32`:
1. **Ejecución SIMD e ILP (Instruction-Level Parallelism):** Eliminación de cadenas de dependencia en FMA (latencia de 4 ciclos en Intel Golden Cove), reducción horizontal sin `_mm_hadd_ps` y evaluación multi-vectorial simultánea (`4x1 AoS` y `8x1 AoSoA`).
2. **Organización Física de Memoria:** Comparación controlada entre **AoS** (*Array of Structures*), **SoA** (*Structure of Arrays*) y **AoSoA** (*Array of Structures of Arrays* en bloques transpuestos de $8 \times 64 = 2048\text{ B}$).
3. **Particionamiento Físico Geométrico (`IVF-Flat` en espacio original):** Agrupación física en $P=128$ particiones K-Means ordenadas por radio ascendente respecto del centroide, visitadas en orden de cercanía a la consulta $q$.
4. **Poda Geométrica Exacta Conservadora (*Certified Exact Geometric Pruning*):** Descarte determinista de particiones, cáscaras esféricas (*radial shells*) y colas dimensionales en base ortonormal PCA centrada por partición, con **recálculo de distancia final en el espacio original `float32`**.

---

## 🏗️ Arquitectura del Motor

```
ALG_2.0/
├── include/
│   ├── alg_flat_types.h       # TopKHeap estático con desempate determinista (dist_sq, id) y ColdCacheEvictor (36 MB)
│   ├── dataset_loader.h       # Generador determinista y cargador binario compatible con Hito 8/9
│   ├── simd_kernels.h         # Kernels: Scalar estricto, AVX2 AoS (1x1 y 4x1), AVX2 SoA (8x1), AVX2 AoSoA (8x1)
│   ├── physical_layouts.h     # Estructuras físicas: FlatAoS, FlatSoA, FlatAoSoA e IVFPartitionedIndex
│   └── execution_plans.h      # Los 9 planes de ejecución + Auditor de Poda de Nivel 3 (PruningAuditReport)
├── src/
│   ├── dataset_loader.cpp     # Generación reproducible bit-a-bit de escenarios R, D y M50/50
│   ├── physical_layouts.cpp   # K-Means++, Jacobi O(64) + Gram-Schmidt, defecto ||Q*Q^T-I||_F y colas por partición
│   └── execution_plans.cpp    # Implementación instrumentada con RDTSC, cotas conservadoras IEEE-754 y auditoría
├── tests/
│   └── test_exactness.cpp     # Certificación Hito 2A en 3 niveles y 4 suites (incluyendo casos adversariales)
├── benchmarks/
│   └── run_ablation_hito1.cpp # Benchmark canónico Warm Cache vs. Cold Cache (LLC Flush) fijado a P-Core 0
└── results/
    ├── ablation_hito1_summary.csv
    └── run_metadata.txt
```

---

## 🛡️ Hito 2A: Auditoría de Exactitud Matemática en 3 Niveles (`tests/test_exactness.cpp`)

Antes de construir selectores adaptativos o escalar en tamaño, el **Hito 2A** garantiza que ningún mecanismo de poda pueda descartar un vecino verdadero ni alterar el orden exacto de resultados en aritmética finita IEEE-754 `float32`:

### 1. Formulación Exacta de la Cota PCA Centrada por Partición
Para cada vector $x$ perteneciente a la partición $p$ con centroide $c_p$, almacenamos el residuo rotado $\tilde{x}_p = \hat{Q}^T(x - c_p)$ y sus normas de cola **respecto del centroide de su partición**:
$$r^{(48)}_{x,p} = \bigl\|\hat{Q}^T(x - c_p)_{16..63}\bigr\|_2, \qquad r^{(32)}_{x,p} = \bigl\|\hat{Q}^T(x - c_p)_{32..63}\bigr\|_2$$
En tiempo de consulta, al visitar una partición activa $p$, se obtiene el residuo rotado de la consulta respecto de ese mismo centroide $\tilde{q}_p = \hat{Q}^T(q - \mu) - \hat{Q}^T(c_p - \mu) = \hat{Q}^T(q - c_p)$ y sus normas de cola por partición:
$$r^{(48)}_{q,p} = \bigl\|\hat{Q}^T(q - c_p)_{16..63}\bigr\|_2, \qquad r^{(32)}_{q,p} = \bigl\|\hat{Q}^T(q - c_p)_{32..63}\bigr\|_2$$
Por la desigualdad triangular inversa en el subespacio de cola ($s \in \{16, 32\}$):
$$\bigl\|\tilde{q}_{p, s..63} - \tilde{x}_{p, s..63}\bigr\|_2 \ge \Bigl|\, r^{(64-s)}_{q,p} - r^{(64-s)}_{x,p} \,\Bigr|$$
Esta formulación es **exactamente invariante a traslaciones de los centroides lejos del origen** ($\|c_p\|_2 \gg 0$) y elimina la cancelación catastrófica en punto flotante.

### 2. Cotas Numéricamente Conservadoras en IEEE-754 `float32`
* **Defecto de Ortogonalidad Medido:** Al construir $\hat{Q}^T$ en `float32`, se mide en `double` el defecto exacto $\|\hat{Q}\hat{Q}^T - I_{64}\|_F \approx 2.93 \times 10^{-7}$.
* **Radios Conservadores:** `min_radius` se redondea hacia $0$ (`nextafterf(..., 0.0f)`) y `max_radius` hacia $+\infty$ (`nextafterf(..., INFINITY)`) a partir de normas calculadas en `double`.
* **Holgura IEEE-754 por Partición (`fp_slack_p`):** Descuenta el error máximo de redondeo FMA de rotación y sustracción antes de elevar al cuadrado la cota de cola.
* **Distancia Final en Espacio Original:** Todo candidato que sobrevive al filtro en `Plan 5B` (así como todo candidato que supera la compuerta SIMD en `Plan 1..5A`) evalúa su distancia final en el **espacio original `float32`** (`q` contra `part_orig_vectors`) con desempate determinista `(dist_sq ASC, id ASC)`.

### 3. Resultados de los 3 Niveles de Certificación (`test_exactness.exe`)
| Suite de Certificación | Vectores ($N$) | Consultas ($Q$) | Nivel 1: `StrictID@10` / `@100` (0 tolerancia) | Nivel 1B: `RankMatch%` | Nivel 2: `MaxOrigDistErr` | Nivel 3: Vectores Podados Auditados | Violaciones de Cota ($LB^2 > d_{\text{orig}}^2$) | $\min(d_{\text{orig}}^2 / LB^2)$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1_Full_Benchmark_100k_Mix_s009` | $100.000$ | $100$ | **100.0000%** | **100.0000%** | `0.000e+00` | $1.025.773$ | **0** | $1.087924$ |
| `2_Adversarial_FarOffOrigin` ($\|c_p\| \gg 0$) | $20.000$ | $80$ | **100.0000%** | **100.0000%** | `0.000e+00` | $230.486$ | **0** | $1.000088$ |
| `3_Adversarial_Collinear_Tail_And_Ties` | $20.000$ | $60$ | **100.0000%** | **100.0000%** | `0.000e+00` | $111.750$ | **0** | $1.001392$ |
| `4_Full_Benchmark_100k_Tight_s003` | $100.000$ | $100$ | **100.0000%** | **100.0000%** | `0.000e+00` | $1.459.047$ | **0** | $1.057055$ |

---

## 📊 Resultados Canónicos del Hito 1 + 2A (`results/ablation_hito1_summary.csv`)

### Metadatos de Reproducción (`results/run_metadata.txt`)
* **Compilador y Flags:** `Clang 22.1.8` (`zig c++ -O3 -march=native -mavx2 -mfma -std=c++17`)
* **CPU y Afinidad:** Intel Core i5-13500 ($24\,\text{MB}$ Intel Smart Cache L3), fijado a `P-Core 0` (`AffinityMask=0x1`, `HIGH_PRIORITY_CLASS`)
* **Corpus y Semilla:** $N = 100.000$, $D = 64$ (`float32`, $25.6\,\text{MB}$), `Seed = 8`
* **Protocolo de Caché:** `WARM_CACHE` ($500$ consultas tras $20$ consultas de calentamiento por plan) y `COLD_LLC_FLUSH` ($40$ consultas con barrido de $36\,\text{MB}$ + `_mm_mfence` antes de cada consulta)
* **Métrica de Exactitud:** `Strict ID Recall` ($0$ tolerancia de distancia en la frontera)

---

### 1. Escenario Base (`R_k64_s009_Base`, $\sigma = 0.09$, Selectividad 15%)

#### A. Condición Warm Cache (Top-100 y Top-10)
| Plan de Ejecución | Top-K | Prep ($\mu$s) | Calc ($\mu$s) | **p50 ($\mu$s)** | p95 ($\mu$s) | Mean ($\mu$s) | Ciclos/Vec | % Podado Exacto | BW Efectivo (GB/s) | Strict Recall@K |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0_Ref_Hito9_ExactScan` | 100 | 0.0 | 171.4 | **168.5** | 201.6 | 171.5 | 28.4 | 0.0% | 22.86 | **100.000%** |
| `1_Flat_Scalar_AoS` | 100 | 0.0 | 571.4 | **485.4** | 1017.5 | 571.4 | 94.5 | 0.0% | 6.86 | **100.000%** |
| `2A_Flat_AVX2_AoS_1x` | 100 | 0.0 | 157.6 | **146.3** | 196.9 | 157.6 | 26.1 | 0.0% | 24.87 | **100.000%** |
| `2B_Flat_AVX2_AoS_4x` | 100 | 0.0 | 146.3 | **132.3** | 184.8 | 146.3 | 24.2 | 0.0% | 26.80 | **100.000%** |
| `3A_Flat_AVX2_SoA` | 100 | 0.0 | 767.3 | **685.9** | 1165.7 | 767.3 | 126.9 | 0.0% | 24.73 | **100.000%** |
| `3B_Flat_AVX2_AoSoA` | 100 | 0.0 | 714.2 | **606.8** | 1295.1 | 714.2 | 118.2 | 0.0% | 26.57 | **100.000%** |
| `4_IVF_Flat_AVX2` | 100 | 3.3 | 138.7 | **141.1** | 177.9 | 142.0 | 23.5 | 0.0% | 27.60 | **100.000%** |
| **`5A_IVF_PartPrune_AVX2`** | **100** | **3.4** | **123.4** | **113.7** | **167.2** | **126.7** | **20.9** | **0.0%** | **30.93** | **100.000%** |
| **`5B_IVF_FullExactPrune_AVX2`** | **100** | **3.7** | **132.9** | **116.9** | **255.4** | **136.6** | **22.6** | **66.5%** | **10.05** | **100.000%** |
| `0_Ref_Hito9_ExactScan` | 10 | 0.0 | 135.6 | **128.1** | 164.9 | 135.6 | 22.4 | 0.0% | 28.91 | **100.000%** |
| `2B_Flat_AVX2_AoS_4x` | 10 | 0.0 | 111.9 | **101.7** | 151.1 | 111.9 | 18.5 | 0.0% | 35.04 | **100.000%** |
| `5A_IVF_PartPrune_AVX2` | 10 | 3.4 | 116.1 | **118.6** | 154.4 | 119.4 | 19.7 | 0.3% | 32.74 | **100.000%** |
| **`5B_IVF_FullExactPrune_AVX2`** | **10** | **3.5** | **89.0** | **77.2** | **194.5** | **92.5** | **15.3** | **74.3%** | **11.32** | **100.000%** |

#### B. Condición Cold Cache (`COLD_LLC_FLUSH`, Selectividad 15%, Top-100)
| Plan de Ejecución | Prep ($\mu$s) | Calc ($\mu$s) | **p50 ($\mu$s)** | p95 ($\mu$s) | Mean ($\mu$s) | Ciclos/Vec | % Podado Exacto | Strict Recall@100 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0_Ref_Hito9_ExactScan` | 0.0 | 712.8 | **681.8** | 1109.5 | 712.8 | 117.9 | 0.0% | **100.000%** |
| `1_Flat_Scalar_AoS` | 0.0 | 1517.8 | **1368.2** | 2579.5 | 1517.8 | 251.1 | 0.0% | **100.000%** |
| `2B_Flat_AVX2_AoS_4x` | 0.0 | 538.6 | **498.7** | 918.1 | 538.6 | 89.1 | 0.0% | **100.000%** |
| `4_IVF_Flat_AVX2` | 7.8 | 571.1 | **564.1** | 889.8 | 578.9 | 95.7 | 0.0% | **100.000%** |
| `5A_IVF_PartPrune_AVX2` | 7.0 | 495.9 | **473.2** | 789.3 | 502.9 | 83.2 | 0.0% | **100.000%** |
| **`5B_IVF_FullExactPrune_AVX2`** | **9.5** | **363.5** | **352.8** | **575.2** | **373.0** | **61.7** | **72.4%** | **100.000%** |

#### C. Sensibilidad a la Selectividad en `R_k64_s009_Base` (Warm Cache, Top-100)
| Selectividad | `0_Ref_Hito9` p50 ($\mu$s) | `4_IVF_Flat` p50 ($\mu$s) | `5A_PartPrune` p50 ($\mu$s) | `5B_FullExactPrune` p50 ($\mu$s) [% Podado] | Mejor Speedup Exacto |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **10.0%** ($10.120$ vecs) | 114.0 | 78.9 | 77.9 | **77.1 $\mu$s** [**64.5%**] | **$1.48\times$** (`5B`) |
| **15.0%** ($15.077$ vecs) | 168.5 | 141.1 | **113.7 $\mu$s** | **116.9 $\mu$s** [**66.5%**] | **$1.48\times$** (`5A` / `5B`) |
| **25.0%** ($25.127$ vecs) | 280.6 | 223.3 | 246.0 | **176.0 $\mu$s** [**68.5%**] | **$1.59\times$** (`5B`) |

---

### 2. Escenario de Alta Localidad (`R_k64_s003_Tight`, $\sigma = 0.03$, Selectividad 15%)

| Condición | Plan de Ejecución | Prep ($\mu$s) | Calc ($\mu$s) | **p50 ($\mu$s)** | Mean ($\mu$s) | Ciclos/Vec | % Podado Exacto | Speedup (p50) | Strict Recall@K |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Warm (Top-100)** | `0_Ref_Hito9_ExactScan` | 0.0 | 168.4 | **167.8** | 168.4 | 27.9 | 0.0% | $1.00\times$ | **100.000%** |
| **Warm (Top-100)** | `4_IVF_Flat_AVX2` (Sin poda) | 3.8 | 150.0 | **152.5** | 153.8 | 25.4 | 0.0% | $1.10\times$ | **100.000%** |
| **Warm (Top-100)** | **`5A_IVF_PartPrune_AVX2`** | 3.3 | 31.5 | **13.0** | 34.7 | 5.7 | **87.2%** | **$12.91\times$** | **100.000%** |
| **Warm (Top-100)** | **`5B_IVF_FullExactPrune_AVX2`**| 3.6 | 30.9 | **17.3** | 34.5 | 5.7 | **95.0%** | **$9.70\times$** | **100.000%** |
| **Warm (Top-10)** | `0_Ref_Hito9_ExactScan` | 0.0 | 138.9 | **127.0** | 138.9 | 23.0 | 0.0% | $1.00\times$ | **100.000%** |
| **Warm (Top-10)** | **`5A_IVF_PartPrune_AVX2`** | 3.0 | 17.0 | **6.4** | 20.0 | 3.3 | **88.6%** | **$19.84\times$** | **100.000%** |
| **Warm (Top-10)** | **`5B_IVF_FullExactPrune_AVX2`**| 3.4 | 15.3 | **8.3** | 18.7 | 3.1 | **96.0%** | **$15.30\times$** | **100.000%** |
| **Cold (Top-100)** | `0_Ref_Hito9_ExactScan` | 0.0 | 739.8 | **757.2** | 739.8 | 122.4 | 0.0% | $1.00\times$ | **100.000%** |
| **Cold (Top-100)** | **`5A_IVF_PartPrune_AVX2`** | 9.2 | 39.0 | **34.1** | 48.2 | 8.0 | **96.4%** | **$22.21\times$** | **100.000%** |
| **Cold (Top-100)** | **`5B_IVF_FullExactPrune_AVX2`**| 12.4 | 58.4 | **46.3** | 70.8 | 11.7 | **97.9%** | **$16.35\times$** | **100.000%** |

---

### 3. Escenarios Mixto (`M5050_k64_s009`) y Adversario Uniforme (`D_k64_s009_Adv`, Selectividad 15%)

| Escenario | Condición | `0_Ref_Hito9` p50 ($\mu$s) | `2B_AVX2_4x` p50 ($\mu$s) | `4_IVF_Flat` p50 ($\mu$s) | `5A_PartPrune` p50 ($\mu$s) | `5B_FullPrune` p50 ($\mu$s) [% Podado] | Mejor Plan Exacto |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`M5050_k64_s009`** | Warm (Top-100) | 155.7 | 157.2 | **126.8** | 134.0 | 212.3 [45.5%] | **`4_IVF_Flat`** ($1.23\times$) |
| **`M5050_k64_s009`** | Cold (Top-100) | 677.4 | **543.6** | 666.4 | 927.1 | **613.5** [37.5%] | **`2B_AVX2_4x`** ($1.25\times$) |
| **`D_k64_s009_Adv`** | Warm (Top-100) | 159.9 | 145.0 | 122.6 | **119.5** | 211.0 [24.2%] | **`5A_PartPrune`** ($1.34\times$) |
| **`D_k64_s009_Adv`** | Warm (Top-10) | 127.9 | **100.8** | 120.7 | 120.6 | 188.9 [37.8%] | **`2B_AVX2_4x`** ($1.27\times$) |

---

## 🛠️ Compilación y Ejecución Reproducible

```powershell
# 1. Ejecutar la Auditoria Estricta de Exactitud Hito 2A (3 Niveles, 4 Suites)
& "$env:LOCALAPPDATA\Microsoft\WinGet\Links\zig.exe" c++ -O3 -march=native -mavx2 -mfma -std=c++17 `
    tests/test_exactness.cpp src/dataset_loader.cpp src/physical_layouts.cpp src/execution_plans.cpp `
    -o test_exactness.exe
.\test_exactness.exe

# 2. Ejecutar el Benchmark Canonico de Ablacion (Warm Cache + Cold LLC Flush)
& "$env:LOCALAPPDATA\Microsoft\WinGet\Links\zig.exe" c++ -O3 -march=native -mavx2 -mfma -std=c++17 `
    benchmarks/run_ablation_hito1.cpp src/dataset_loader.cpp src/physical_layouts.cpp src/execution_plans.cpp `
    -o run_ablation_hito1.exe
.\run_ablation_hito1.exe
```
