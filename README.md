# ALG-Flat 2.0: Motor Vectorial Exacto en Espacio Plano con Poda Geométrica Certificada en IEEE-754

> **Estado del Proyecto:** **Hito 1 + Hito 2A.2 Completados y Certificados** (Ablación Física de 9 Planes + Derivación Formal IEEE-754 con **Redondeo Dirigido Conservador** `round_up_f32` / `round_down_f32` mediante `std::nextafterf`, profundidades de redondeo separadas por etapa $\gamma_{17}, \gamma_{33}, \gamma_{67}$, y Auditoría Exhaustiva de las **5 Etapas de Poda** con **Cero Tolerancia Oculta** $\Delta = LB_{\text{safe}}^2 - d_{\text{orig}}^2 \le 0$).  
> **Barrera Metodológica:** Congelado para revisión de **Hito 2A.2** antes de iniciar **Hito 2B** (Selector Adaptativo Ligero) e **Hito 2C** (Escalamiento de Corpus y Dimensión).

---

## 1. Motivación Física: Del Diagnóstico de Hito 9 a ALG-Flat 2.0

En el cierre del proyecto anterior ([`ALG` Hito 9](https://github.com/Santirobm/ALG)), demostramos que sobre un corpus de $N = 100{,}000$ vectores en $D = 64$ dimensiones (`float32`), un escaneo plano secuencial (`ExactScan`) promediaba **~168–173 µs**, superando ampliamente la navegación aproximada sobre grafos `HNSW` (**~450–830 µs**).

El análisis físico de **ALG-Flat 2.0** parte de una precisión fundamental sobre la jerarquía de memoria del procesador utilizado (**Intel Core i5-13500**, con **24 MB de Intel Smart Cache L3**):
- **El corpus completo ocupa $100{,}000 \times 64 \times 4\text{ bytes} = \mathbf{25.6\text{ MB}}$**, lo cual **excede ligeramente los 24 MB de caché L3** (especialmente al sumar metadatos, listas de elegibles, buffers del proceso y competencia por vías de caché).
- Sin embargo, bajo una selectividad del $15\%$ ($15{,}077$ vectores elegibles), los vectores candidatos ocupan únicamente **$3.86\text{ MB}$**, y aun en recorridos más amplios, una fracción sustancial reside en L3 mientras el resto se beneficia del prefetcher de hardware L2/L3 y de la ausencia de *pointer chasing* aleatorio.

**Hipótesis Central de ALG-Flat 2.0:**
> En lugar de usar la geometría de los datos para navegar un grafo aproximado (sufriendo fallos de caché L3/DRAM impredecibles y pérdida de exactitud), podemos usar **organización física contigua en memoria + cotas geométricas exactas certificadas en IEEE-754 `float32`** para **descartar particiones y dimensiones redundantes sin perder un solo vecino verdadero (`100.0000%` `IDSetRecall` y `100.0000%` `RankMatch`)**.

---

## 2. Certificación Matemática de Exactitud (Hito 2A → Hito 2A.1 → Hito 2A.2)

Para que un motor con poda geométrica (`Plan 5A` y `Plan 5B`) pueda declararse **matemáticamente exacto** en aritmética finita IEEE-754 `binary32` (`float32`), en el cierre definitivo **Hito 2A.2** ([`include/ieee754_bounds.h`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/include/ieee754_bounds.h), [`src/physical_layouts.cpp`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/src/physical_layouts.cpp), [`src/execution_plans.cpp`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/src/execution_plans.cpp), [`tests/test_exactness.cpp`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/tests/test_exactness.cpp)) se implementaron y verificaron seis garantías formales:

### 2.1. Invariancia Traslacional: Residuos Rotados por Partición
Cada partición $p$ con centroide $c_p$ almacena directamente el **residuo local rotado respecto de su propio centroide**:
$$\tilde{x}_p = \hat{Q}^T (x - c_p), \qquad r_{48}(x, p) = \|\tilde{x}_{p, 16..63}\|_2, \qquad r_{32}(x, p) = \|\tilde{x}_{p, 32..63}\|_2$$
En tiempo de consulta, una única rotación global $\tilde{q} = \hat{Q}^T (q - \mu)$ permite obtener en $\mathcal{O}(D)$ por partición visitada:
$$\tilde{q}_p = \tilde{q} - \widehat{Q^T(c_p - \mu)} \approx \hat{Q}^T (q - c_p)$$
Así, tanto en las primeras $16/32$ dimensiones como en las colas $16..63$ y $32..63$, la resta se realiza en el mismo sistema de referencia local centrado en $c_p$:
$$\tilde{q}_p - \tilde{x}_p = \hat{Q}^T(q - c_p) - \hat{Q}^T(x - c_p) = \hat{Q}^T(q - x)$$

### 2.2. Redondeo Dirigido Conservador (`round_up_f32` / `round_down_f32`) con `std::nextafterf`
En **Hito 2A.2** se eliminó la conversión implícita al más cercano (`static_cast<float>`) al almacenar cotas y holguras calculadas en `double` (`binary64`, $u_{64} = 2^{-53}$). En [`include/ieee754_bounds.h`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/include/ieee754_bounds.h) se implementaron dos operadores de conversión dirigida certificada que primero absorben el error de evaluación intermedia en `binary64` ($\pm 16 u_{64}$) y luego garantizan la dirección del redondeo en `binary32` mediante `std::nextafterf`:
- **`round_up_f32(double x)` (hacia $+\infty$):** Aplicado a toda cota superior o término de holgura que se **resta** en una cota inferior (`delta_vec_p`, `fp_slack_p`, `sq_slack_16_p`, `sq_slack_32_p`, `sq_slack_64_p`, `dc_high`, `rho_upper`, `r_pad`, `r_high`, `q_mean_norm`, `max_radius`). Garantiza $\operatorname{round\_up\_f32}(x) \ge x$.
- **`round_down_f32(double x)` (hacia $0$):** Aplicado a todo factor multiplicativo de escala o cota inferior final (`scale_safe_16`, `scale_safe_32`, `scale_safe_64`, `dc_low`, `lb_part_sq`, `r_low`, `lb_rad_sq`, `min_radius`). Garantiza $0 \le \operatorname{round\_down\_f32}(x) \le x$.

### 2.3. Derivación Formal por Etapa (`16D` $\to \gamma_{17}$, `32D` $\to \gamma_{33}$, `64D` $\to \gamma_{67}$)
En [`include/ieee754_bounds.h`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/include/ieee754_bounds.h), con redondeo unitario $u = 2^{-24} = 5.9604644775390625 \times 10^{-8}$ y $\gamma_n = \frac{n u}{1 - n u}$, se separaron explícitamente las profundidades de acumulación SIMD de las tres etapas progresivas del kernel `5B`:

1. **Distancia canónica en espacio original (`l2_hito9_ref`):** Resta por coordenada ($1u$) + acumulación FMA + reducción horizontal, acotada por profundidad secuencial $n \le 66$ ($\gamma_{66} = 3.93392 \times 10^{-6}$):
   $$\bigl|\operatorname{fl}_{\text{orig}}(\|q - x\|_2^2) - \|q - x\|_2^2\bigr| \le \gamma_{66} \|q - x\|_2^2$$
2. **Defecto de ortogonalidad medido en `binary64`:** Al construir el índice se mide exactamente $\delta_{\text{ortho}} = \|\hat{Q}\hat{Q}^T - I_{64}\|_F \approx 2.93 \times 10^{-7}$, garantizando para todo $v \in \mathbb{R}^{64}$:
   $$(1 - \delta_{\text{ortho}})\|v\|_2^2 \le \|\hat{Q}^T v\|_2^2 \le (1 + \delta_{\text{ortho}})\|v\|_2^2$$
3. **Perturbación L2 del vector diferencia rotado $\delta_{\text{vec}}(q, p)$:**
   $$\bigl\|(\tilde{q}_p - \tilde{x}_p)_S - (\hat{Q}^T(q - x))_S\bigr\|_2 \le \delta_{\text{vec}}(q, p) = \operatorname{round\_up\_f32}\Bigl(\bigl(8\gamma_{64}(1+u) + 4u\bigr)\|q - \mu\|_2 + 4u\|c_p - \mu\|_2 + 2u\,r_{\max}(p)\Bigr)$$
4. **Holgura lineal de cola (`fp_slack_p`):**
   $$\text{fp\_slack}_p = \operatorname{round\_up\_f32}\Bigl((1 + 4u)\bigl(\delta_{\text{vec}}(q, p) + \gamma_{64}(d_c^{\text{high}} + \delta_{\text{vec}}(q, p)) + u\,r_{\max}(p)\bigr)\Bigr)$$
5. **Holguras cuadráticas y escalas seguras separadas por etapa ($s \in \{16, 32, 64\}$):**
   Sea $M = 2\,\delta_{\text{vec}}(q, p)\,\bigl(d_c^{\text{high}} + r_{\max}(p) + \delta_{\text{vec}}(q, p)\bigr)$:
   - **Etapa 1 (`PCA 16D` + cola `48D`, profundidad $n = 17$, $\gamma_{17} = 1.01328 \times 10^{-6}$):**
     $$\text{sq\_slack}_{16,p} = \operatorname{round\_up\_f32}\!\left(\frac{M}{1 - \gamma_{17}}\right), \qquad s_{\text{safe},16} = \operatorname{round\_down\_f32}\!\left(\frac{(1 - \gamma_{66})(1 - \gamma_{17})(1 - 6u)}{1 + \delta_{\text{ortho}}}\right)$$
   - **Etapa 2 (`PCA 32D` + cola `32D`, profundidad $n = 33$, $\gamma_{33} = 1.96696 \times 10^{-6}$):**
     $$\text{sq\_slack}_{32,p} = \operatorname{round\_up\_f32}\!\left(\frac{M}{1 - \gamma_{33}}\right), \qquad s_{\text{safe},32} = \operatorname{round\_down\_f32}\!\left(\frac{(1 - \gamma_{66})(1 - \gamma_{33})(1 - 6u)}{1 + \delta_{\text{ortho}}}\right)$$
   - **Etapa 3 (`PCA 64D` completo en espacio rotado, sin cola geométrica, profundidad $n = 67$, $\gamma_{67} = 3.99353 \times 10^{-6}$):**
     $$\text{sq\_slack}_{64,p} = \operatorname{round\_up\_f32}\!\left(\frac{M}{1 - \gamma_{67}}\right), \qquad s_{\text{safe},64} = \operatorname{round\_down\_f32}\!\left(\frac{(1 - \gamma_{66})(1 - \gamma_{67})(1 - 4u)}{1 + \delta_{\text{ortho}}}\right)$$

### 2.4. Distancia Final Canónica y Orden Total Determinista
En todos los planes (`0` a `5B`), todo candidato que ingresa o actualiza el heap Top-$K$ evalúa su distancia final con la **función de distancia original canónica** `l2_hito9_ref(q, x_orig, 64)` sobre las coordenadas originales `float32`, aplicando el orden total estricto:
$$\text{precedes}(a, b) \iff (a.\text{dist\_sq} < b.\text{dist\_sq}) \lor \bigl(a.\text{dist\_sq} == b.\text{dist\_sq} \land a.\text{id} < b.\text{id}\bigr)$$

### 2.5. Auditoría Exhaustiva de las 5 Etapas de Poda con Cero Tolerancia Oculta (`test_exactness.cpp`)
En **Hito 2A.2**, [`audit_plan5b_pruning_soundness`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/src/execution_plans.cpp#L938-L1344) reproduce **bit a bit** el camino de producción de `Plan 5B` (incluyendo el bucle SIMD 4x con `_mm256_hadd_ps` y `_mm_fmadd_ps`, más la cola escalar) y audita las **5 etapas reales de descarte**:
1. **Etapa 1 (`Part`):** Poda de partición completa ($LB_{\text{part}}^2 > \theta_K^2$).
2. **Etapa 2 (`Radial`):** Poda por cascarón radial $[r_{\text{low}}, r_{\text{high}}]$ respecto de $c_p$.
3. **Etapa 3 (`PCA16`):** Prefijo de 16 dimensiones principales + cota de cola de 48 dimensiones.
4. **Etapa 4 (`PCA32`):** Prefijo de 32 dimensiones principales + cota de cola de 32 dimensiones.
5. **Etapa 5 (`PCA64`):** Distancia rotada completa de 64 dimensiones ($LB_{64,\text{safe}}^2 = s_{\text{safe},64} \cdot \max(0, d_{64,\text{rot}} - \text{sq\_slack}_{64,p}) > \theta_K^2$), donde ya **no existe holgura geométrica por coordenadas omitidas** y la cota depende exclusivamente de las holguras numéricas IEEE-754 ($\text{sq\_slack}_{64,p}$, $s_{\text{safe},64}$ y $\delta_{\text{ortho}}$).

Para cada candidato descartado en cualquiera de las 5 etapas, el auditor mide el **exceso con signo exacto**:
$$\Delta(q, x) = LB_{\text{safe}}^2(q, x) - d_{\text{orig}}^2(q, x)$$
y verifica con `assert` la conservación exacta del flujo de candidatos:
$$\text{Part} + \text{Radial} + \text{PCA16} + \text{PCA32} + \text{PCA64} + \text{EvaluadosOrig} == \text{ElegiblesAuditados}$$

| Suite de Certificación (`test_exactness.cpp`) | Elegibles Auditados | Podados `Part` | Podados `Radial` | Podados `PCA16` | Podados `PCA32` | Podados `PCA64` | Evaluados `Orig 64D` | Violaciones Estrictas ($\Delta > 0$, 5 etapas) | Máx. Exceso con Signo ($\max \Delta$) | Ratio Mín. $d_{\text{orig}}^2 / LB_{\text{safe}}^2$ | `IDSetRecall` / `RankMatch` / `DupIDs` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1_Full_Benchmark_100k_Mix_s009`** | $1{,}507{,}700$ | $21$ | $32{,}202$ | $572{,}144$ | $421{,}407$ | **$459{,}408$** | $22{,}518$ | **`0 / 0 / 0 / 0 / 0`** | **`-8.333e-05`** | **`1.000057`** | **`100.0000%` / `100.0000%` / `0`** |
| **`2_Adversarial_FarOffOrigin_Centroids`** | $240{,}480$ | $229{,}206$ | $0$ | $890$ | $50$ | **$8$** | $10{,}326$ | **`0 / 0 / 0 / 0 / 0`** | **`-1.820e+00`** | **`1.000045`** | **`100.0000%` / `100.0000%` / `0`** |
| **`3_Adversarial_TrueCentroidCollinear_And_Ties`** | $696{,}660$ | $546{,}793$ | $40{,}596$ | $44{,}920$ | $36{,}376$ | **$16{,}728$** | $11{,}247$ | **`0 / 0 / 0 / 0 / 0`** | **`-2.727e-05`** | **`1.000062`** | **`100.0000%` / `100.0000%` / `0`** |
| **`4_Full_Benchmark_100k_Tight_s003`** | $1{,}507{,}700$ | $1{,}285{,}446$ | $54{,}391$ | $80{,}346$ | $38{,}860$ | **$30{,}840$** | $17{,}817$ | **`0 / 0 / 0 / 0 / 0`** | **`-4.165e-05`** | **`1.000059`** | **`100.0000%` / `100.0000%` / `0`** |

> **Observación crítica sobre la Etapa 5 (`PCA64`):** Al auditar los **$506{,}984$ descartes en `PCA 64D`** (donde la holgura geométrica es exactamente cero), el ratio mínimo observado entre la distancia original y la cota inferior certificada pasa a ser **`1.000045`–`1.000062`** y el máximo exceso con signo se sitúa entre **`-2.727e-05` y `-8.333e-05`**, confirmando que las cotas derivadas con redondeo dirigido son simultáneamente **ajustadas** ($\sim 5 \times 10^{-5}$ de margen relativo) y **100% libres de violaciones ($\Delta \le 0$ en los $3{,}890{,}632$ descartes totales de las 5 etapas)**.

---

## 3. Taxonomía de los 9 Planes de Ejecución (Escalera de Ablación Hito 1)

| Plan | Nombre en Código | Layout Físico | Kernel de Distancia | Poda Geométrica | Qué Aísla Experimentalmente |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Plan 0** | `0_Ref_Hito9_ExactScan` | `AoS` Base | AVX2 1x + `_mm256_hadd_ps` | Ninguna (0%) | Línea base exacta e idéntica a `ExactScan` de Hito 9. |
| **Plan 1** | `1_Flat_Scalar_AoS` | `AoS` Base | Escalar puro (`O3` sin SIMD) | Ninguna (0%) | Costo real de renunciar a vectorización SIMD en 64D. |
| **Plan 2A** | `2A_Flat_AVX2_AoS_1x` | `AoS` Base | AVX2 FMA + reducción rápida | Ninguna (0%) | Ganancia por eliminar `_mm256_hadd_ps` en el mismo layout. |
| **Plan 2B** | `2B_Flat_AVX2_AoS_4x` | `AoS` Base | AVX2 FMA **4-way loop unrolling** + prefetch | Ninguna (0%) | Ganancia por ocultar latencia de FMA (4 ciclos) procesando 4 vectores en paralelo. |
| **Plan 3A** | `3A_Flat_AVX2_SoA` | `SoA` Global | AVX2 FMA vertical (8 vecs/reg) | Ninguna (0%) | Penalización por pérdida de localidad espacial y máscara de predicado dispersa en SoA. |
| **Plan 3B** | `3B_Flat_AVX2_AoSoA` | `AoSoA` (8x64) | AVX2 FMA por bloque de 2 KB | Ninguna (0%) | Mitigación de TLB misses respecto a SoA, pero aún penalizado por carriles SIMD inactivos bajo filtrado del 15%. |
| **Plan 4** | `4_IVF_Flat_AVX2` | `IVF` Contiguo | AVX2 FMA 4x por partición | **Ninguna (0%)** | Ganancia pura por **localidad física de memoria** (evaluando exactamente los mismos 15,077 vectores). |
| **Plan 5A** | `5A_IVF_PartPrune_AVX2` | `IVF` Contiguo | AVX2 FMA 4x por partición | **Esfera de Partición** ($LB_{\text{part}}^2 > \theta_K^2$) | Ganancia por saltar particiones completas sin tocar sus vectores. |
| **Plan 5B** | `5B_IVF_FullExactPrune_AVX2` | `IVF` Centrado + PCA | Cascada `16D -> 32D -> 64D` + dist. original | **5 Etapas: Partición + Radial + PCA 16D/32D/64D** | Ganancia por poda geométrica multinivel exacta certificada en IEEE-754. |

---

## 4. Resultados Experimentales Canónicos Sincronizados (`Seed=8`)

> **Metadatos de Ejecución ([`results/run_metadata.txt`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/results/run_metadata.txt)):**  
> - **CPU:** Intel Core i5-13500 (24 MB L3 Smart Cache), fijado a `P-Core 0` (`AffinityMask=0x1`, `HIGH_PRIORITY_CLASS`).  
> - **SO:** `Windows 10 x86_64 (NT 10.0 Build 19045)`.  
> - **Compilador:** `Clang 22.1.8` (`-O3 -march=native -mavx2 -mfma -std=c++17`).  
> - **Protocolo:** `WARM_CACHE` = 500 consultas medidas tras 20 consultas de calentamiento por plan; `COLD_LLC_FLUSH` = 40 consultas precedidas cada una por un barrido de escritura sobre un buffer de **36 MB** (`> 24 MB L3`) + `_mm_mfence()`.  
> - **Archivo fuente de datos:** Todos los valores de las tablas siguientes provienen exactamente de [`results/ablation_hito1_summary.csv`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/results/ablation_hito1_summary.csv).

### 4.1. Escenario 1: Alta Localidad Geométrica (`R_k64_s003_Tight`, $\sigma = 0.03$, Selectividad $15\%$, Top-$100$)

| Plan de Ejecución | WARM `Prep` (µs) | WARM `Calc` (µs) | WARM `p50` (µs) | WARM `p95` (µs) | WARM `Mean` (µs) | Ciclos / Vec | Vectores Podados | COLD `p50` (µs) | COLD `Mean` (µs) | `IDSetRecall@100` / `RankMatch@100` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0_Ref_Hito9_ExactScan`** | 0.0 | 173.8 | 168.6 | 202.3 | 173.8 | 28.7 | 0.0% | 458.0 | 444.1 | `100.0%` / `100.0%` |
| **`1_Flat_Scalar_AoS`** | 0.0 | 468.0 | 465.7 | 482.4 | 468.1 | 77.5 | 0.0% | 1052.6 | 1066.7 | `100.0%` / `100.0%` |
| **`2A_Flat_AVX2_AoS_1x`** | 0.0 | 169.7 | 158.8 | 219.7 | 169.7 | 28.1 | 0.0% | 495.5 | 505.0 | `100.0%` / `100.0%` |
| **`2B_Flat_AVX2_AoS_4x`** | 0.0 | 157.2 | 152.6 | 180.8 | 157.2 | 26.0 | 0.0% | 472.8 | 491.3 | `100.0%` / `100.0%` |
| **`3A_Flat_AVX2_SoA`** | 0.0 | 710.4 | 680.6 | 892.8 | 710.4 | 117.5 | 0.0% | 1489.9 | 1519.2 | `100.0%` / `100.0%` |
| **`3B_Flat_AVX2_AoSoA`** | 0.0 | 594.6 | 584.6 | 648.5 | 594.6 | 98.4 | 0.0% | 1271.6 | 1292.5 | `100.0%` / `100.0%` |
| **`4_IVF_Flat_AVX2`** | 3.1 | 129.7 | 126.4 | 156.3 | 132.9 | 22.0 | 0.0% | 463.4 | 471.9 | `100.0%` / `100.0%` |
| **`5A_IVF_PartPrune_AVX2`** | 7.6 | 22.8 | **17.2** | **121.7** | **30.4** | **5.0** | **87.2%** | **31.3** | **43.2** | **`100.0%` / `100.0%`** |
| **`5B_IVF_FullExactPrune_AVX2`** | 7.8 | 25.6 | **19.4** | **134.4** | **33.4** | **5.5** | **95.0%** | **39.6** | **46.4** | **`100.0%` / `100.0%`** |

- **Para `Top-10` en este mismo escenario (`WARM_CACHE`):** `5A` alcanza **`p50 = 10.7 µs`** (`Mean = 22.0 µs`, **13.1x más rápido** que `Plan 0` en `p50`), y `5B` alcanza **`p50 = 13.0 µs`** (`Mean = 23.6 µs`, `p95 = 101.9 µs`) podando el **96.0%** de los vectores con `100.000%` de `IDSetRecall` y `RankMatch`.
- **En `COLD_LLC_FLUSH` (tras expulsar L3 con 36 MB):** `Plan 0` se degrada a **458.0 µs** (`p50`), mientras que `5A` (**31.3 µs**, `Mean = 43.2 µs`) y `5B` (**39.6 µs**, `Mean = 46.4 µs`, `p95 = 54.9 µs`) logran un speedup de **11.6x–14.6x en `p50`** (**~10x en media**) porque leen de DRAM únicamente el ~3.4% de las particiones relevantes.

---

### 4.2. Escenario 2: Localidad Moderada (`R_k64_s009_Base`, $\sigma = 0.09$, Selectividad $15\%$, Top-$100$)

En este régimen, el radio del Top-$100$ supera la distancia a los bordes de la mayoría de las particiones, por lo que la poda gruesa por partición (`5A`) logra descartar casi cero vectores ($0.002\%$). Aquí es donde la **poda fina multinivel (cascarón radial + cascada PCA 16D/32D/64D centrada en $c_p$) de `Plan 5B`** demuestra su valor diferencial:

| Plan de Ejecución | WARM `Prep` (µs) | WARM `Calc` (µs) | WARM `p50` (µs) | WARM `p95` (µs) | WARM `Mean` (µs) | Ciclos / Vec | Vectores Podados | COLD `p50` (µs) | COLD `Mean` (µs) | `IDSetRecall@100` / `RankMatch@100` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0_Ref_Hito9_ExactScan`** | 0.0 | 172.8 | 168.3 | 185.2 | 172.8 | 28.6 | 0.0% | 477.5 | 448.1 | `100.0%` / `100.0%` |
| **`2B_Flat_AVX2_AoS_4x`** | 0.0 | 147.0 | 145.6 | 148.7 | 147.0 | 24.3 | 0.0% | 472.0 | 488.8 | `100.0%` / `100.0%` |
| **`4_IVF_Flat_AVX2`** | 3.1 | 129.5 | 126.1 | 155.0 | 132.6 | 21.9 | 0.0% | 461.6 | 471.4 | `100.0%` / `100.0%` |
| **`5A_IVF_PartPrune_AVX2`** | 7.7 | 125.2 | 131.2 | 139.5 | 132.9 | 22.0 | 0.0% | 452.0 | 456.9 | `100.0%` / `100.0%` |
| **`5B_IVF_FullExactPrune_AVX2`** | 8.1 | **124.5** | **117.9** | 237.5 | **132.6** | **21.9** | **66.5%** | **290.2** | **308.2** | **`100.0%` / `100.0%`** |

- **Para `Top-10` en `R_k64_s009_Base` (`WARM_CACHE`):** `5B` poda el **74.3%** de los vectores y reduce la latencia mediana `p50` a **98.7 µs** (`Mean = 113.1 µs`) frente a **142.9 µs** (`Mean = 156.5 µs`) de `Plan 0` y **120.9 µs** de `Plan 4`.
- **Para `Selectividad 25%` (`25,127` elegibles, `Top-100`, `WARM_CACHE`):** `5B` poda el **68.5%** de los candidatos y baja la mediana `p50` a **173.4 µs** (`Mean = 198.3 µs`) frente a **260.4 µs** (`Mean = 268.3 µs`) de `Plan 0` y **206.1 µs** (`Mean = 210.1 µs`) de `Plan 4`.
- **En `COLD_LLC_FLUSH` (`Top-100`, Sel $15\%$):** Mientras `5A` tarda **452.0 µs** (`p50`) / **456.9 µs** (`Mean`) porque no logra podar particiones, `5B` poda el **72.4%** de las evaluaciones completas de 64 dimensiones y baja la latencia en frío a **290.2 µs** (`p50`) / **308.2 µs** (`Mean`).

---

### 4.3. Escenarios Mixto (`M5050_k64_s009`) y Adversarial Difuso (`D_k64_s009_Adv`, Top-$100$, Sel $15\%$)

| Escenario | Plan | WARM `p50` (µs) | WARM `p95` (µs) | WARM `Mean` (µs) | Vectores Podados | COLD `p50` (µs) | COLD `Mean` (µs) | `IDSetRecall@100` / `RankMatch@100` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`M5050_k64_s009`** | `0_Ref_Hito9_ExactScan` | 168.4 | 214.6 | 173.4 | 0.0% | 457.7 | 445.6 | `100.0%` / `100.0%` |
| **`M5050_k64_s009`** | `2B_Flat_AVX2_AoS_4x` | 145.4 | 148.9 | 146.1 | 0.0% | 474.4 | 495.5 | `100.0%` / `100.0%` |
| **`M5050_k64_s009`** | `4_IVF_Flat_AVX2` | **129.7** | 159.2 | **133.2** | 0.0% | 457.9 | 472.2 | `100.0%` / `100.0%` |
| **`M5050_k64_s009`** | `5A_IVF_PartPrune_AVX2` | 134.5 | **142.0** | 135.1 | 0.0% | **447.0** | **464.0** | `100.0%` / `100.0%` |
| **`M5050_k64_s009`** | `5B_IVF_FullExactPrune_AVX2` | 217.8 | 251.0 | 184.5 | 45.5% | 591.4 | 526.5 | `100.0%` / `100.0%` |
| **`D_k64_s009_Adv`** | `0_Ref_Hito9_ExactScan` | 168.6 | 176.6 | 172.1 | 0.0% | 462.5 | 445.9 | `100.0%` / `100.0%` |
| **`D_k64_s009_Adv`** | `2B_Flat_AVX2_AoS_4x` | 144.8 | 148.9 | 145.3 | 0.0% | 471.3 | 490.7 | `100.0%` / `100.0%` |
| **`D_k64_s009_Adv`** | `4_IVF_Flat_AVX2` | **132.7** | 160.1 | **137.1** | 0.0% | 459.3 | 480.4 | `100.0%` / `100.0%` |
| **`D_k64_s009_Adv`** | `5A_IVF_PartPrune_AVX2` | 137.7 | **143.6** | 140.1 | 0.0% | **457.3** | **471.3** | `100.0%` / `100.0%` |
| **`D_k64_s009_Adv`** | `5B_IVF_FullExactPrune_AVX2` | 232.8 | 246.3 | 231.2 | 24.2% | 559.3 | 572.4 | `100.0%` / `100.0%` |

---

## 5. Caracterización de Latencia de Cola (`p50` vs `p95`/`p99`) y Conclusiones Físicas para Hito 2B

1. **Por qué `SoA` (`3A`) y `AoSoA` (`3B`) pierden bajo filtrado del 15%:**  
   Cuando el predicado selecciona el $15\%$ de los vectores, un bloque SIMD vertical de 8 vectores consecutivos contiene en promedio solo $8 \times 0.15 = 1.2$ vectores elegibles. Por tanto, los planes `3A` y `3B` desperdician el **$85\%$ de sus carriles AVX2** calculando coordenadas de vectores no elegibles (evaluando $72{,}568$ vectores en bloques activos frente a los $15{,}077$ elegibles). En cambio, el recorrido indexado por `eligible_ids` en `AoS` (`2B` y `4`) aprovecha el $100\%$ del cómputo en vectores elegibles.
2. **Ganancia pura por localidad física (`Plan 4` vs `Plan 2B`):**  
   Sin podar un solo vector (`0.0%` podados), reordenar físicamente los vectores por partición geométrica (`Plan 4`) reduce el tiempo de cómputo en `WARM_CACHE` de **145.3–157.2 µs (`2B`) a 129.5–134.0 µs (`4`)** (y el total a **132.6–137.1 µs** en todos los escenarios, incluso el adversarial difuso), elevando el ancho de banda efectivo a **~28.6–29.6 GB/s**.
3. **Caracterización del régimen `p50` vs `p95`/`p99` en `Plan 5A` y `Plan 5B`:**  
   - **En `R_k64_s003_Tight` (`WARM_CACHE`):** `5A` alcanza `p50 = 17.2 µs` y `Mean = 30.4 µs`, pero `p95 = 121.7 µs` (`p99 = 134.0 µs`). Esto ocurre porque la mayoría de las consultas caen cerca del núcleo de un cluster (cerrando rápidamente la cota $r_K$ en la primera partición y podando $>95\%$ del corpus en `~17 µs`), mientras que el $5\%$ de las consultas cae en regiones inter-cluster donde $r_K$ es mayor y obliga a recorrer múltiples particiones vecinas. Aun así, incluso en `p95` (`121.7 µs`) y `p99` (`134.0 µs`), `5A` sigue siendo significativamente más rápido que `Plan 0` (`202.3 µs` / `291.7 µs`).
   - **En `R_k64_s009_Base` (`WARM_CACHE`):** `5B` mejora la mediana (`p50 = 117.9 µs` vs `126.1 µs` de `Plan 4` y `168.3 µs` de `Plan 0`) e iguala la media de `Plan 4` (`Mean = 132.6 µs`), pero su cola `p95` sube a **`237.5 µs`** (`p99 = 248.8 µs`) frente a **`139.5 µs`** de `5A` (o **`148.7 µs`** de `2B`). La causa física es directa: para el $\sim 10\%$ de las consultas situadas en zonas difusas de alta superposición, la cascada por candidato (`radio -> 16D -> 32D -> 64D` + `l2_hito9_ref`) no logra podar suficientes vectores para amortizar su sobrecarga de instrucciones y predicción de saltos.
   - **En `M5050_k64_s009` (bimodal):** El $50\%$ de las consultas son agrupadas y el $50\%$ son difusas, produciendo una distribución bimodal donde `5B` (`p50 = 217.8 µs`, `p95 = 251.0 µs`, `Mean = 184.5 µs`) pierde frente a `4` (`p50 = 129.7 µs`, `Mean = 133.2 µs`) y `5A` (`p50 = 134.5 µs`, `p95 = 142.0 µs`).
4. **Implicación directa para el Selector Adaptativo de Hito 2B:**  
   La divergencia entre `p50` y `p95` en `5B` demuestra que un selector adaptativo en el **Hito 2B** no debe elegir un plan estático por dataset, sino **por consulta individual** usando las distancias de la consulta a los centroides $d(q, c_{(1)}), d(q, c_{(2)}), \dots$ calculadas en los primeros `~3–7 µs` de `Prep`:
   - Si $d(q, c_{(1)})$ indica que la consulta cae dentro de un núcleo compacto ($d(q, c_{(1)}) \ll d(q, c_{(2)})$), despachar `5A`/`5B`.
   - Si la brecha de cuantización $d(q, c_{(2)}) / d(q, c_{(1)}) \approx 1$ o la tasa de poda en las primeras 2 particiones visitadas cae por debajo del umbral de amortización ($\sim 55\%$), **conmutar inmediatamente al barrido AVX2 4x sin sobrecarga de poda (`Plan 4` / `Plan 5A`)**, acotando el `p95`/`p99` al nivel de `Plan 4/5A` (`~139–160 µs`) mientras se conserva la aceleración de `5B` en `p50` y `COLD_LLC_FLUSH`.

---

## 6. Reproducción Exacta

```powershell
# 1. Compilar y ejecutar la Suite de Certificación Matemática de Exactitud de 5 Etapas (Hito 2A.2)
& "$env:LOCALAPPDATA\Microsoft\WinGet\Links\zig.exe" c++ -O3 -march=native -mavx2 -mfma -std=c++17 `
    tests/test_exactness.cpp src/dataset_loader.cpp src/physical_layouts.cpp src/execution_plans.cpp `
    -o test_exactness.exe
.\test_exactness.exe

# 2. Compilar y ejecutar el Benchmark Canónico de Ablación (WARM + COLD LLC Flush)
& "$env:LOCALAPPDATA\Microsoft\WinGet\Links\zig.exe" c++ -O3 -march=native -mavx2 -mfma -std=c++17 `
    benchmarks/run_ablation_hito1.cpp src/dataset_loader.cpp src/physical_layouts.cpp src/execution_plans.cpp `
    -o run_ablation_hito1.exe
.\run_ablation_hito1.exe
```
