# ALG-Flat (ALG 2.0) — Adaptive Geometric Layout Engine

**Exact vector search through physical partitioning, memory layout optimization, exact geometric pruning, and SIMD execution.**

---

## 🎯 Motivación y Reformulación Científica

Tras concluir el Hito 9 del proyecto original (`ALG`), demostramos empíricamente que en regímenes de dimensionalidad moderada ($N = 100.000$, $D = 64$, selectividades del $10\%$ al $25\%$), un escaneo contiguo vectorizado (`ExactScan`) supera sistemáticamente a las arquitecturas basadas en grafos (`HNSW`) y a los despachadores híbridos dinámicos bajo cargas fuera de distribución (*OOD*).

**ALG-Flat (`ALG 2.0`)** nace de una pregunta fundamental:
> *¿Cuánto rendimiento estamos perdiendo por cómo almacenamos y recorremos físicamente los vectores en memoria, incluso cuando ya elegimos el algoritmo exacto correcto?*

En lugar de sacrificar precisión mediante estructuras aproximadas o pagar el sobrecosto de un selector dinámico, **ALG-Flat** optimiza la ejecución **100% exacta (`float32`, Recall@K = 100.00%)** aislando rigurosamente cuatro ejes físicos:
1. **Ejecución SIMD e ILP (Instruction-Level Parallelism):** Eliminación de cadenas de dependencia en FMA (latencia de 4 ciclos en Intel Golden Cove), eliminación de `_mm_hadd_ps` y evaluación multi-vectorial simultánea (`4x1 AoS` y `8x1 AoSoA`).
2. **Organización Física de Memoria:** Comparación controlada entre **AoS** (*Array of Structures*), **SoA** (*Structure of Arrays*) y **AoSoA** (*Array of Structures of Arrays* en bloques transpuestos de $8 \times 64$).
3. **Particionamiento Físico Geométrico (`IVF-Flat`):** Reutilización de centroides K-Means en *build-time* para agrupar físicamente los vectores en regiones contiguas ordenadas radialmente desde el centroide hacia la periferia.
4. **Poda Geométrica Exacta (*Exact Geometric Pruning*):** Descarte determinista de particiones, cáscaras esféricas (*radial shells*) y colas dimensionales en base ortonormal PCA sin perder un solo vecino exacto.

---

## 🏗️ Arquitectura del Motor

```
ALG_2.0/
├── include/
│   ├── alg_flat_types.h       # TopKHeap estático O(1) rechazo, métricas por fase y ColdCacheEvictor (36 MB)
│   ├── dataset_loader.h       # Generador determinista y cargador binario compatible con Hito 8/9
│   ├── simd_kernels.h         # Kernels: Scalar estricto, AVX2 AoS (1x1 y 4x1), AVX2 SoA (8x1), AVX2 AoSoA (8x1)
│   ├── physical_layouts.h     # Estructuras físicas: FlatAoS, FlatSoA, FlatAoSoA e IVFPartitionedIndex (con PCA O(64))
│   └── execution_plans.h      # Los 9 planes de ejecución de la escalera de ablación
├── src/
│   ├── dataset_loader.cpp     # Generación reproducible bit-a-bit de escenarios R, D y M50/50
│   ├── physical_layouts.cpp   # K-Means++, diagonalización Jacobi O(64), ordenamiento radial y empaquetado
│   └── execution_plans.cpp    # Implementación instrumentada con RDTSC y desglose Prep / Calc / Total
├── tests/
│   └── test_exactness.cpp     # Certificación automática de 100.00% Recall@10 y Recall@100 en los 9 planes
├── benchmarks/
│   └── run_ablation_hito1.cpp # Benchmark Warm Cache vs. Cold Cache (LLC Flush) fijado a P-Core 0
└── results/
    └── ablation_hito1_summary.csv
```

---

## 🔬 Hito 1: Escalera de Ablación Incremental y Diagnóstico Físico

Para evitar mezclar técnicas sin poder aislar la contribución de cada una, evaluamos **9 planes de ejecución incrementales** (todos en `float32` con **100.00% de Recall exacto**):

| Plan | Nombre | Qué aísla y descubre |
| :---: | :--- | :--- |
| **0** | `0_Ref_Hito9_ExactScan` | Baseline original de Hito 8/9 (`std::priority_queue` + 1 acumulador AVX2 + `_mm_hadd_ps`). |
| **1** | `1_Flat_Scalar_AoS` | Coste de referencia escalar real (auto-vectorización LLVM deshabilitada) + `TopKHeap` estático. |
| **2A** | `2A_Flat_AVX2_AoS_1x` | Ganancia pura de SIMD 1x1 (doble acumulador FMA + reducción rápida sin `hadd` + `TopKHeap`). |
| **2B** | `2B_Flat_AVX2_AoS_4x` | Ganancia de paralelismo a nivel de instrucción (procesa 4 vectores elegibles en paralelo reutilizando registros de `q`). |
| **3A** | `3A_Flat_AVX2_SoA` | Layout transpuesto global (*Structure of Arrays*, 64 flujos simultáneos). |
| **3B** | `3B_Flat_AVX2_AoSoA` | Layout transpuesto por bloques (*AoSoA*, bloques contiguos de 8 vectores $\times$ 64 dims = 2 KB). |
| **4** | `4_IVF_Flat_AVX2` | Ganancia pura del particionamiento físico K-Means ($P=128$, visita ordenada por cercanía a `q`, **sin poda**). |
| **5A** | `5A_IVF_PartPrune_AVX2` | Ganancia de la poda exacta a nivel de partición por desigualdad triangular: $d(q, c_p) - r_{\max}(p) > \rho_K$. |
| **5B** | `5B_IVF_FullExactPrune_AVX2` | Poda exacta multi-nivel: Partición + Cáscara Radial $[d_c - \rho_K, d_c + \rho_K]$ + Base Ortogonal PCA $O(64)$ con corte progresivo exacto en 16D/32D. |

---

## 📊 Resultados Experimentales (Intel Core i5-13500, P-Core 0)

### 1. Diagnóstico Térmico de Caché: ¿Por qué `ExactScan` tardaba ~170–184 µs en Hito 9?
El corpus completo de $100.000 \times 64 \times 4\text{ B}$ ocupa **$25.6\,\text{MB}$**, superando los **$24\,\text{MB}$ de Intel Smart Cache (L3)** del i5-13500. Sin embargo, para una selectividad fija del $15\%$, los $15.077$ vectores elegibles representan exactamente **$3.86\,\text{MB}$**.
* En **Warm Cache** (consultas sucesivas sobre el mismo predicado), esos $3.86\,\text{MB}$ permanecen calientes en L3, entregando un ancho de banda efectivo de **$23\text{–}30\,\text{GB/s}$** y una latencia de **$168.6\,\mu\text{s}$ (p50)**.
* En **Cold Cache** (evicción explícita de L3 mediante barrido de $36\,\text{MB}$ + `mfence` antes de cada consulta), `0_Ref_Hito9_ExactScan` sube a **$492\text{–}525\,\mu\text{s}$ (p50)** y el ancho de banda cae a **$7.8\text{–}8.5\,\text{GB/s}$** debido a los accesos indirectos espaciados en DRAM.

---

### 2. Escenario Base (`R_k64_s009_Base`, $\sigma = 0.09$, Selectividad 15%)

#### A. Condición Warm Cache (Top-100 y Top-10)
| Plan de Ejecución | Top-K | Prep ($\mu$s) | Calc ($\mu$s) | **p50 ($\mu$s)** | p95 ($\mu$s) | Ciclos/Vec | % Podado Exacto | BW Efectivo (GB/s) | Recall@K |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0_Ref_Hito9_ExactScan` | 100 | 0.0 | 172.0 | **168.6** | 190.7 | 28.4 | 0.0% | 22.79 | 100.00% |
| `1_Flat_Scalar_AoS` | 100 | 0.0 | 490.7 | **459.6** | 693.6 | 81.2 | 0.0% | 7.99 | 100.00% |
| `2A_Flat_AVX2_AoS_1x` | 100 | 0.0 | 154.5 | **153.8** | 158.5 | 25.5 | 0.0% | 25.37 | 100.00% |
| `2B_Flat_AVX2_AoS_4x` | 100 | 0.0 | 144.4 | **142.2** | 146.9 | 23.9 | 0.0% | 27.13 | 100.00% |
| `3A_Flat_AVX2_SoA` | 100 | 0.0 | 628.8 | **617.2** | 690.7 | 104.0 | 0.0% | 30.18 | 100.00% |
| `3B_Flat_AVX2_AoSoA` | 100 | 0.0 | 597.8 | **552.5** | 809.5 | 98.9 | 0.0% | 31.75 | 100.00% |
| `4_IVF_Flat_AVX2` | 100 | 3.5 | 129.5 | **126.5** | 158.0 | 22.0 | 0.0% | 29.46 | 100.00% |
| `5A_IVF_PartPrune_AVX2` | 100 | 3.5 | 124.1 | **126.4** | 134.5 | 21.1 | 0.0% | 30.72 | 100.00% |
| **`5B_IVF_FullExactPrune_AVX2`** | **100** | **3.7** | **110.3** | **105.7** | **188.5** | **18.8** | **61.6%** | **13.51** | **100.00%** |
| `0_Ref_Hito9_ExactScan` | 10 | 0.0 | 141.7 | **140.3** | 145.9 | 23.4 | 0.0% | 27.67 | 100.00% |
| `2B_Flat_AVX2_AoS_4x` | 10 | 0.0 | 116.4 | **114.9** | 124.6 | 19.2 | 0.0% | 33.68 | 100.00% |
| **`5B_IVF_FullExactPrune_AVX2`** | **10** | **3.6** | **79.7** | **72.0** | **171.3** | **13.7** | **70.9%** | **14.02** | **100.00%** |

#### B. Condición Cold Cache (`COLD_LLC_FLUSH`, Top-100)
| Plan de Ejecución | Prep ($\mu$s) | Calc ($\mu$s) | **p50 ($\mu$s)** | p95 ($\mu$s) | Ciclos/Vec | % Podado Exacto | Recall@100 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0_Ref_Hito9_ExactScan` | 0.0 | 459.3 | **492.2** | 598.0 | 76.0 | 0.0% | 100.00% |
| `1_Flat_Scalar_AoS` | 0.0 | 1085.2 | **1057.5** | 1313.6 | 179.6 | 0.0% | 100.00% |
| `2B_Flat_AVX2_AoS_4x` | 0.0 | 485.3 | **467.8** | 581.2 | 80.3 | 0.0% | 100.00% |
| `4_IVF_Flat_AVX2` | 7.3 | 466.8 | **450.4** | 636.0 | 78.4 | 0.0% | 100.00% |
| **`5B_IVF_FullExactPrune_AVX2`** | **7.3** | **317.6** | **309.7** | **545.3** | **53.7** | **67.8%** | **100.00%** |

---

### 3. Escenario de Alta Localidad (`R_k64_s003_Tight`, $\sigma = 0.03$, Selectividad 15%)

Cuando los clusters tienen menor dispersión intracluster ($\sigma = 0.03$), las cotas geométricas exactas eliminan más del **90% al 97% del trabajo**, reduciendo la latencia en **más de un orden de magnitud** sin perder exactitud:

| Condición | Plan de Ejecución | Prep ($\mu$s) | Calc ($\mu$s) | **p50 ($\mu$s)** | Mean ($\mu$s) | Ciclos/Vec | % Podado Exacto | Speedup (p50) | Recall@K |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Warm (Top-100)** | `0_Ref_Hito9_ExactScan` | 0.0 | 170.5 | **169.5** | 170.5 | 28.2 | 0.0% | $1.00\times$ | 100.00% |
| **Warm (Top-100)** | `4_IVF_Flat_AVX2` (Sin poda) | 3.5 | 127.1 | **126.2** | 130.6 | 21.6 | 0.0% | $1.34\times$ | 100.00% |
| **Warm (Top-100)** | **`5A_IVF_PartPrune_AVX2`** | 3.0 | 22.1 | **11.9** | 25.1 | 4.1 | **87.2%** | **$14.24\times$** | **100.00%** |
| **Warm (Top-100)** | **`5B_IVF_FullExactPrune_AVX2`**| 3.1 | 22.3 | **12.2** | 25.4 | 4.2 | **92.0%** | **$13.89\times$** | **100.00%** |
| **Warm (Top-10)** | `0_Ref_Hito9_ExactScan` | 0.0 | 142.1 | **140.7** | 142.1 | 23.5 | 0.0% | $1.00\times$ | 100.00% |
| **Warm (Top-10)** | **`5A_IVF_PartPrune_AVX2`** | 3.0 | 14.7 | **6.1** | 17.8 | 2.9 | **88.6%** | **$23.07\times$** | **100.00%** |
| **Warm (Top-10)** | **`5B_IVF_FullExactPrune_AVX2`**| 3.3 | 13.9 | **6.7** | 17.3 | 2.9 | **93.5%** | **$21.00\times$** | **100.00%** |
| **Cold (Top-100)** | `0_Ref_Hito9_ExactScan` | 0.0 | 481.1 | **508.7** | 481.1 | 79.6 | 0.0% | $1.00\times$ | 100.00% |
| **Cold (Top-100)** | **`5A_IVF_PartPrune_AVX2`** | 10.5 | 36.2 | **33.8** | 46.6 | 7.7 | **96.4%** | **$15.05\times$** | **100.00%** |
| **Cold (Top-100)** | **`5B_IVF_FullExactPrune_AVX2`**| 10.7 | 36.3 | **33.9** | 47.0 | 7.8 | **97.3%** | **$15.01\times$** | **100.00%** |

---

### 4. Escenario Adversario Uniforme (`D_k64_s009_Adv`, Consultas Uniformes en $S^{63}$)

En una distribución adversaria uniforme en $\mathbb{R}^{64}$, toda consulta $q$ es casi ortogonal a los centroides ($\|q - c_p\|_2 \approx \sqrt{2}$), por lo que la desigualdad triangular a nivel de partición no poda particiones ($0.0\%$). Aun así:
* **`4_IVF_Flat_AVX2` y `5A_IVF_PartPrune_AVX2`** reducen la latencia de **$169.8\,\mu\text{s}$ a $132.5\,\mu\text{s}$ (p50)** en Top-100 (un **22% más rápido** que el `ExactScan` de Hito 9) gracias al kernel `4x1` y al ordenamiento físico que satura temprano el umbral del `TopKHeap`.
* En cambio, `5B_IVF_FullExactPrune_AVX2` solo logra podar el $17.1\%$ del trabajo en $D$, pagando $185.6\,\mu\text{s}$ por las comprobaciones progresivas de 16D/32D. Esto demuestra empíricamente por qué **aislar `5A` y `5B`** es indispensable: `5A` nunca degrada en regímenes adversarios (+3.5 $\mu$s de preparación ampliamente compensados por la ganancia de localidad), mientras que el strip-mining de `5B` brilla cuando la poda supera el ~35%.

---

### 5. Lección Arquitectónica Clave: AoS vs. SoA vs. AoSoA bajo Filtrado

La comparación de los tres diseños físicos reveló dos fenómenos fundamentales:
1. **`AoSoA` supera consistentemente a `SoA` entre un 15% y un 28%** tanto en Warm como en Cold Cache, alcanzando hasta **$36.53\,\text{GB/s}$** de ancho de banda efectivo frente a $22\text{–}30\,\text{GB/s}$ de `SoA`. La razón física es que `SoA` abre 64 flujos de memoria simultáneos separados por $400\,\text{KB}$, saturando los descriptores del *hardware prefetcher* L1/L2 y el TLB, mientras que `AoSoA` lee un único flujo contiguo de bloques de $2\,\text{KB}$.
2. **Dilución de Carriles SIMD (*SIMD Lane Dilution*) en bloques estáticos bajo filtros no correlacionados:** Cuando el filtro tiene selectividad aleatoria $s = 15\%$, la probabilidad de que un bloque estático de 8 vectores en `SoA`/`AoSoA` contenga al menos un vector elegible es $1 - (1 - 0.15)^8 = 72.75\%$. Esto obliga a `SoA`/`AoSoA` estáticos a evaluar el $72.75\%$ del corpus, mientras que **`2B_Flat_AVX2_AoS_4x`** (e **`IVF_Flat_AVX2`**) recolectan dinámicamente 4 vectores elegibles contiguos y los procesan en paralelo sin desperdiciar un solo carril SIMD.

---

## 🛠️ Compilación y Ejecución

```powershell
# 1. Test de Exactitud Matemática (100.00% Recall@10 y Recall@100)
& "$env:LOCALAPPDATA\Microsoft\WinGet\Links\zig.exe" c++ -O3 -std=c++17 -mavx2 -mfma -I include `
  src/dataset_loader.cpp src/physical_layouts.cpp src/execution_plans.cpp tests/test_exactness.cpp `
  -o tests/test_exactness.exe
.\tests\test_exactness.exe

# 2. Benchmark de Ablación Hito 1 (Warm Cache & Cold LLC Flush)
& "$env:LOCALAPPDATA\Microsoft\WinGet\Links\zig.exe" c++ -O3 -std=c++17 -mavx2 -mfma -I include `
  src/dataset_loader.cpp src/physical_layouts.cpp src/execution_plans.cpp benchmarks/run_ablation_hito1.cpp `
  -o benchmarks/run_ablation_hito1.exe
.\benchmarks\run_ablation_hito1.exe
```
