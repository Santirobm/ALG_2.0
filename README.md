# ALG-Flat 2.0: Motor Vectorial Exacto en Espacio Plano con Poda Geométrica Certificada en IEEE-754

> **Estado del Proyecto:** **Hito 1 + Hito 2A.1 Completados y Certificados** (Ablación Física de 9 Planes + Derivación Formal IEEE-754 mediante factores $\gamma_n$ de Higham + Auditoría de Poda con **Cero Tolerancia Oculta** $\Delta = LB_{\text{safe}}^2 - d_{\text{orig}}^2 \le 0$).  
> **Barrera Metodológica:** Congelado para revisión de **Hito 2A.1** antes de iniciar **Hito 2B** (Selector Adaptativo Ligero) e **Hito 2C** (Escalamiento de Corpus y Dimensión).

---

## 1. Motivación Física: Del Diagnóstico de Hito 9 a ALG-Flat 2.0

En el cierre del proyecto anterior ([`ALG` Hito 9](https://github.com/Santirobm/ALG)), demostramos que sobre un corpus de $N = 100{,}000$ vectores en $D = 64$ dimensiones (`float32`), un escaneo plano secuencial (`ExactScan`) promediaba **~168 µs**, superando ampliamente la navegación aproximada sobre grafos `HNSW` (**~450–830 µs**).

El análisis físico de **ALG-Flat 2.0** parte de una precisión fundamental sobre la jerarquía de memoria del procesador utilizado (**Intel Core i5-13500**, con **24 MB de Intel Smart Cache L3**):
- **El corpus completo ocupa $100{,}000 \times 64 \times 4\text{ bytes} = \mathbf{25.6\text{ MB}}$**, lo cual **excede ligeramente los 24 MB de caché L3** (especialmente al sumar metadatos, listas de elegibles, buffers del proceso y competencia por vías de caché).
- Sin embargo, bajo una selectividad del $15\%$ ($15{,}077$ vectores elegibles), los vectores candidatos ocupan únicamente **$3.86\text{ MB}$**, y aun en recorridos más amplios, una fracción sustancial reside en L3 mientras el resto se beneficia del prefetcher de hardware L2/L3 y de la ausencia de *pointer chasing* aleatorio.

**Hipótesis Central de ALG-Flat 2.0:**
> En lugar de usar la geometría de los datos para navegar un grafo aproximado (sufriendo fallos de caché L3/DRAM impredecibles y pérdida de exactitud), podemos usar **organización física contigua en memoria + cotas geométricas exactas certificadas en IEEE-754 `float32`** para **descartar particiones y dimensiones redundantes sin perder un solo vecino verdadero (`100.0000%` `IDSetRecall` y `100.0000%` `RankMatch`)**.

---

## 2. Certificación Matemática de Exactitud (Hito 2A + Hito 2A.1)

Para que un motor con poda geométrica (`Plan 5A` y `Plan 5B`) pueda declararse **matemáticamente exacto** en aritmética finita IEEE-754 `binary32` (`float32`), en el **Hito 2A** y su cierre **Hito 2A.1** ([`include/ieee754_bounds.h`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/include/ieee754_bounds.h), [`src/physical_layouts.cpp`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/src/physical_layouts.cpp), [`src/execution_plans.cpp`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/src/execution_plans.cpp), [`tests/test_exactness.cpp`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/tests/test_exactness.cpp)) se implementaron y verificaron cinco garantías formales:

### 2.1. Invariancia Traslacional: Residuos Rotados por Partición
En la versión inicial de Hito 1, las colas dimensionales (`16..63` y `32..63`) se almacenaban respecto de la media global $\mu$, lo que introducía dependencia de la posición absoluta del centroide $\|c_p - \mu\|_2$.  
En **Hito 2A**, cada partición $p$ con centroide $c_p$ almacena directamente el **residuo local rotado respecto de su propio centroide**:
$$\tilde{x}_p = \hat{Q}^T (x - c_p), \qquad r_{48}(x, p) = \|\tilde{x}_{p, 16..63}\|_2, \qquad r_{32}(x, p) = \|\tilde{x}_{p, 32..63}\|_2$$
En tiempo de consulta, una única rotación global $\tilde{q} = \hat{Q}^T (q - \mu)$ permite obtener en $\mathcal{O}(D)$ por partición visitada:
$$\tilde{q}_p = \tilde{q} - \widehat{Q^T(c_p - \mu)} \approx \hat{Q}^T (q - c_p)$$
Así, tanto en las primeras $16/32$ dimensiones como en las colas $16..63$ y $32..63$, la resta se realiza en el mismo sistema de referencia local centrado en $c_p$:
$$\tilde{q}_p - \tilde{x}_p = \hat{Q}^T(q - c_p) - \hat{Q}^T(x - c_p) = \hat{Q}^T(q - x)$$

### 2.2. Derivación Formal de Holguras IEEE-754 mediante Factores $\gamma_n$ de Higham
En **Hito 2A.1** se eliminaron todas las constantes empíricas y se reemplazaron en [`include/ieee754_bounds.h`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/include/ieee754_bounds.h) por una derivación cerrada basada en el análisis de redondeo de Higham (*Accuracy and Stability of Numerical Algorithms*, 2nd Ed.), con redondeo unitario $u = 2^{-24} = 5.9604644775390625 \times 10^{-8}$ y $\gamma_n = \frac{n u}{1 - n u}$:

1. **Hipótesis de rango numérico:** Dimensión $D = 64$, coordenadas finitas `binary32` libres de *overflow* ($> 10^{38}$) y *underflow* subnormal ($< 10^{-38}$).
2. **Distancia canónica en espacio original (`l2_hito9_ref`):** Resta por coordenada ($1u$) + acumulación FMA + reducción horizontal, acotada por profundidad secuencial $n \le 66$ ($\gamma_{66} = 3.934 \times 10^{-6}$):
   $$\bigl|\operatorname{fl}_{\text{orig}}(\|q - x\|_2^2) - \|q - x\|_2^2\bigr| \le \gamma_{66} \|q - x\|_2^2$$
3. **Defecto de ortogonalidad medido en `binary64`:** Al construir el índice se mide exactamente $\delta_{\text{ortho}} = \|\hat{Q}\hat{Q}^T - I_{64}\|_F \approx 2.93 \times 10^{-7}$, garantizando para todo $v \in \mathbb{R}^{64}$:
   $$(1 - \delta_{\text{ortho}})\|v\|_2^2 \le \|\hat{Q}^T v\|_2^2 \le (1 + \delta_{\text{ortho}})\|v\|_2^2$$
4. **Rotación en build-time acumulada en `binary64`:** Tanto $\tilde{x}_p = \operatorname{fl}_{32}(\hat{Q}^T(x - c_p))$ como $\tilde{c}_p = \operatorname{fl}_{32}(\hat{Q}^T(c_p - \mu))$ se acumulan en doble precisión (`double`) y se redondean una sola vez a `float32`, acotando su error L2 en $64$ dimensiones por:
   $$\|\tilde{x}_p - \hat{Q}^T(x - c_p)\|_2 \le 2u \cdot r_{\max}(p), \qquad \|\tilde{c}_p - \hat{Q}^T(c_p - \mu)\|_2 \le 2u \cdot \|c_p - \mu\|_2$$
5. **Rotación de la consulta en query-time y resta por partición:** Cada fila de $\tilde{q} = \operatorname{fl}(\hat{Q}^T(q - \mu))$ tiene error acotado por $\gamma_{64}(1+u)\|q - \mu\|_2$; sumando en norma L2 sobre las $64$ filas (factor $\sqrt{64} = 8$) y añadiendo el redondeo de $\tilde{q}_p = \operatorname{fl}(\tilde{q} - \tilde{c}_p)$, la perturbación L2 total $\delta_{\text{vec}}(q, p)$ en cualquier subconjunto de coordenadas $S \subseteq \{0..63\}$ cumple:
   $$\bigl\|(\tilde{q}_p - \tilde{x}_p)_S - (\hat{Q}^T(q - x))_S\bigr\|_2 \le \delta_{\text{vec}}(q, p) = \bigl(8\gamma_{64}(1+u) + 4u\bigr)\|q - \mu\|_2 + 4u\|c_p - \mu\|_2 + 2u\,r_{\max}(p)$$
6. **Holgura lineal de cola (`fp_slack_p`) y holgura cuadrática de prefijo (`sq_slack_p`):**
   $$\text{fp\_slack}_p = (1 + 4u)\Bigl(\delta_{\text{vec}}(q, p) + \gamma_{64}\bigl(d_c^{\text{high}} + \delta_{\text{vec}}(q, p)\bigr) + u\,r_{\max}(p)\Bigr)$$
   $$\text{sq\_slack}_p = \frac{2\,\delta_{\text{vec}}(q, p)\,\bigl(d_c^{\text{high}} + r_{\max}(p) + \delta_{\text{vec}}(q, p)\bigr)}{1 - \gamma_{16}}, \qquad s_{\text{safe}} = \frac{(1 - \gamma_{66})(1 - \gamma_{16})(1 - 6u)}{1 + \delta_{\text{ortho}}}$$
7. **Cota inferior certificada por etapa ($s \in \{16, 32\}$):**
   $$LB_{s,\text{safe}}^2(q, x) = s_{\text{safe}} \cdot \max\Bigl(0,\; D_s^2 + \max\bigl(0,\, |r_{\text{tail}}(q,p) - r_{\text{tail}}(x,p)| - \text{fp\_slack}_p\bigr)^2 - \text{sq\_slack}_p\Bigr) \le \operatorname{fl}_{\text{orig}}\bigl(\|q - x\|_2^2\bigr)$$

### 2.3. Distancia Final Canónica y Orden Total Determinista
En todos los planes (`0` a `5B`), todo candidato que ingresa o actualiza el heap Top-$K$ evalúa su distancia final con la **función de distancia original canónica** `l2_hito9_ref(q, x_orig, 64)` sobre las coordenadas originales `float32`, aplicando el orden total estricto:
$$\text{ precedes }(a, b) \iff (a.\text{dist\_sq} < b.\text{dist\_sq}) \lor \bigl(a.\text{dist\_sq} == b.\text{dist\_sq} \land a.\text{id} < b.\text{id}\bigr)$$

### 2.4. Auditoría de Exactitud en 3 Niveles con Cero Tolerancia Oculta (`test_exactness.cpp`)
En **Hito 2A.1**, [`audit_plan5b_pruning_soundness`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/src/execution_plans.cpp#L483-L580) eliminó cualquier tolerancia en la comparación y registra para cada uno de los millones de descartes el **exceso con signo exacto**:
$$\Delta(q, x) = LB_{\text{safe}}^2(q, x) - d_{\text{orig}}^2(q, x)$$
separando explícitamente:
- **`strict_violations_*` ($\Delta > 0.0$, cero tolerancia):** Cualquier caso donde $LB_{\text{safe}}^2(q, x) > d_{\text{orig}}^2(q, x)$ se contabiliza como violación estricta y aborta la certificación.
- **`diag_margin_events_*` ($0.0 < \Delta \le \epsilon_{\text{diag}}$)** y **`diag_exceed_violations_*` ($\Delta > \epsilon_{\text{diag}}$)**.
- **`IDSetRecall@K` (con rechazo de IDs duplicados)** y **`RankMatch@K` (identidad posición a posición)** reportados por separado.
- **Verificación explícita de colinealidad real respecto de $c_p$ en la Suite 3:** Las consultas adversariales de la Suite 3 se construyen una vez conocidas las particiones reales de K-Means y sus centroides asignados $c_p$, inyectando 50 pares donde $(q - c_p)$ y $(x^* - c_p)$ alcanzan $\cos\theta = 1.000000000$ en `64D` (poda radial), `Cola 16..63` (con cota radial nula para forzar la llegada a `PCA 16D`) y `Cola 32..63` (con cotas radial y `PCA 16D` nulas para forzar la llegada a `PCA 32D`), verificando con `assert` que el gap relativo de la desigualdad triangular inversa sea $< 10^{-6}$ y que $|d_{\text{orig}}^2 / LB_{\text{exact}}^2 - 1| < 10^{-5}$.

| Suite de Certificación (`test_exactness.cpp`) | Vectores ($N$) | Consultas ($Q$) | Elegibles Auditados | Vectores Podados (`5B`) | Violaciones Estrictas ($\Delta > 0$) | Eventos Margen ($0 < \Delta \le \epsilon$) | Falsos Descartes Top-$K$ | Máx. Exceso con Signo ($\max \Delta$) | Ratio Mínimo $d_{\text{orig}}^2 / LB_{\text{safe}}^2$ | `IDSetRecall` / `RankMatch` / `DupIDs` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1_Full_Benchmark_100k_Mix_s009`** | $100{,}000$ | $100$ | $1{,}507{,}700$ | $1{,}025{,}774$ ($68.0\%$) | **`0`** | **`0`** | **`0`** | **`-1.342e-01`** | **`1.087921`** | **`100.0000%` / `100.0000%` / `0`** |
| **`2_Adversarial_FarOffOrigin_Centroids`** ($\|c_p\| \gg 0$) | $20{,}000$ | $80$ | $240{,}480$ | $230{,}152$ ($95.7\%$) | **`0`** | **`0`** | **`0`** | **`-2.066e+00`** | **`1.000051`** | **`100.0000%` / `100.0000%` / `0`** |
| **`3_Adversarial_TrueCentroidCollinear_And_Ties`** ($\cos\theta_{c_p}=1.0$) | $20{,}000$ | $60$ | $696{,}660$ | $668{,}685$ ($96.0\%$) | **`0`** | **`0`** | **`0`** | **`-4.779e-05`** | **`1.000140`** | **`100.0000%` / `100.0000%` / `0`** |
| **`4_Full_Benchmark_100k_Tight_s003`** | $100{,}000$ | $100$ | $1{,}507{,}700$ | $1{,}459{,}045$ ($96.8\%$) | **`0`** | **`0`** | **`0`** | **`-4.105e-02`** | **`1.057056`** | **`100.0000%` / `100.0000%` / `0`** |

> En los **$3{,}952{,}540$ candidatos elegibles auditados** y **$3{,}383{,}656$ descartes efectivos**, se registraron **0 violaciones estrictas ($\Delta > 0$)**, **0 eventos en margen diagnóstico**, **0 falsos descartes del Top-$K$**, **0 IDs duplicados**, **`100.0000%` `IDSetRecall`** y **`100.0000%` `RankMatch`**. Incluso en los pares de colinealidad exacta respecto de $c_p$ (`Suite 3`, donde $|d_{\text{orig}}^2 / LB_{\text{exact}}^2 - 1| \le 1.54 \times 10^{-8}$), la cota certificada mantiene un margen positivo seguro ($d_{\text{orig}}^2 / LB_{\text{safe}}^2 = 1.000140$, $\max \Delta = -4.779 \times 10^{-5} < 0$).

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
| **Plan 5A** | `5A_IVF_PartPrune_AVX2` | `IVF` Contiguo | AVX2 FMA 4x por partición | **Esfera de Partición** ($d_{\text{low}}(q,c_p) - R_{\max} > r_K$) | Ganancia por saltar particiones completas sin tocar sus vectores. |
| **Plan 5B** | `5B_IVF_FullExactPrune_AVX2` | `IVF` Centrado + PCA | Cascada `16D -> 32D -> 64D` + dist. original | **Partición + Cascarón Radial + Cola PCA centrada en $c_p$** | Ganancia por poda geométrica multinivel exacta certificada en IEEE-754. |

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
| **`0_Ref_Hito9_ExactScan`** | 0.0 | 168.4 | 168.2 | 171.2 | 168.5 | 27.9 | 0.0% | 480.3 | 452.8 | `100.0%` / `100.0%` |
| **`1_Flat_Scalar_AoS`** | 0.0 | 477.9 | 469.8 | 520.1 | 478.0 | 79.1 | 0.0% | 1059.6 | 1094.9 | `100.0%` / `100.0%` |
| **`2A_Flat_AVX2_AoS_1x`** | 0.0 | 172.5 | 161.5 | 217.5 | 172.5 | 28.5 | 0.0% | 488.1 | 504.3 | `100.0%` / `100.0%` |
| **`2B_Flat_AVX2_AoS_4x`** | 0.0 | 155.3 | 151.9 | 174.3 | 155.3 | 25.7 | 0.0% | 471.1 | 491.6 | `100.0%` / `100.0%` |
| **`3A_Flat_AVX2_SoA`** | 0.0 | 704.8 | 679.1 | 870.3 | 704.9 | 116.6 | 0.0% | 1456.0 | 1507.4 | `100.0%` / `100.0%` |
| **`3B_Flat_AVX2_AoSoA`** | 0.0 | 626.5 | 602.7 | 736.6 | 626.5 | 103.7 | 0.0% | 1281.5 | 1327.0 | `100.0%` / `100.0%` |
| **`4_IVF_Flat_AVX2`** | 3.2 | 127.2 | 127.2 | 156.0 | 130.4 | 21.6 | 0.0% | 450.8 | 461.3 | `100.0%` / `100.0%` |
| **`5A_IVF_PartPrune_AVX2`** | 2.9 | 23.6 | **12.6** | **117.6** | **26.5** | **4.4** | **87.2%** | **27.3** | **45.6** | **`100.0%` / `100.0%`** |
| **`5B_IVF_FullExactPrune_AVX2`** | 3.1 | 24.4 | **14.6** | **122.2** | **27.5** | **4.5** | **95.0%** | **36.4** | **44.4** | **`100.0%` / `100.0%`** |

- **Para `Top-10` en este mismo escenario (`WARM_CACHE`):** `5A` alcanza **`p50 = 6.0 µs`** (`Mean = 17.5 µs`, **23.4x más rápido** que `Plan 0` en `p50`), y `5B` alcanza **`p50 = 8.1 µs`** (`Mean = 17.7 µs`, `p95 = 85.7 µs`) podando el **96.0%** de los vectores con `100.000%` de `IDSetRecall` y `RankMatch`.
- **En `COLD_LLC_FLUSH` (tras expulsar L3 con 36 MB):** `Plan 0` se degrada a **480.3 µs** (`p50`), mientras que `5A` (**27.3 µs**) y `5B` (**36.4 µs**, `Mean = 44.4 µs`) logran un speedup de **13.2x–17.6x en `p50`** (**10.2x en media**) porque leen de DRAM únicamente el ~3.4% de las particiones relevantes.

---

### 4.2. Escenario 2: Localidad Moderada (`R_k64_s009_Base`, $\sigma = 0.09$, Selectividad $15\%$, Top-$100$)

En este régimen, el radio del Top-$100$ supera la distancia a los bordes de la mayoría de las particiones, por lo que la poda gruesa por partición (`5A`) logra descartar casi cero vectores ($0.002\%$). Aquí es donde la **poda fina multinivel (cascarón radial + colas PCA 16D/32D centradas en $c_p$) de `Plan 5B`** demuestra su valor diferencial:

| Plan de Ejecución | WARM `Prep` (µs) | WARM `Calc` (µs) | WARM `p50` (µs) | WARM `p95` (µs) | WARM `Mean` (µs) | Ciclos / Vec | Vectores Podados | COLD `p50` (µs) | COLD `Mean` (µs) | `IDSetRecall@100` / `RankMatch@100` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0_Ref_Hito9_ExactScan`** | 0.0 | 168.1 | 167.8 | 171.0 | 168.1 | 27.8 | 0.0% | 464.3 | 452.9 | `100.0%` / `100.0%` |
| **`2B_Flat_AVX2_AoS_4x`** | 0.0 | 145.2 | 144.7 | 149.2 | 145.2 | 24.0 | 0.0% | 472.7 | 492.9 | `100.0%` / `100.0%` |
| **`4_IVF_Flat_AVX2`** | 3.2 | 128.1 | 127.2 | 155.4 | 131.2 | 21.7 | 0.0% | 450.2 | 467.5 | `100.0%` / `100.0%` |
| **`5A_IVF_PartPrune_AVX2`** | 3.2 | 125.6 | 128.0 | 136.0 | 128.8 | 21.3 | 0.0% | 450.9 | 464.8 | `100.0%` / `100.0%` |
| **`5B_IVF_FullExactPrune_AVX2`** | 3.6 | 117.8 | **108.0** | 229.7 | **121.4** | **20.1** | **66.5%** | **286.4** | **290.8** | **`100.0%` / `100.0%`** |

- **Para `Top-10` en `R_k64_s009_Base` (`WARM_CACHE`):** `5B` poda el **74.3%** de los vectores y reduce la latencia mediana `p50` a **74.8 µs** (`Mean = 92.5 µs`) frente a **153.0 µs** de `Plan 0` (**2.05x más rápido**).
- **En `COLD_LLC_FLUSH` (`Top-100`):** Mientras `5A` tarda **450.9 µs** porque no logra podar particiones, `5B` poda el **72.4%** de las evaluaciones completas de 64 dimensiones y baja la latencia en frío a **286.4 µs** (`p50`) / **290.8 µs** (`Mean`).

---

### 4.3. Escenarios Mixto (`M5050_k64_s009`) y Adversarial Difuso (`D_k64_s009_Adv`, Top-$100$, Sel $15\%$)

| Escenario | Plan | WARM `p50` (µs) | WARM `p95` (µs) | WARM `Mean` (µs) | Vectores Podados | COLD `p50` (µs) | COLD `Mean` (µs) | `IDSetRecall@100` / `RankMatch@100` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`M5050_k64_s009`** | `0_Ref_Hito9_ExactScan` | 168.4 | 171.9 | 169.2 | 0.0% | 488.2 | 483.6 | `100.0%` / `100.0%` |
| **`M5050_k64_s009`** | `2B_Flat_AVX2_AoS_4x` | 145.0 | 148.7 | 145.3 | 0.0% | 486.9 | 511.9 | `100.0%` / `100.0%` |
| **`M5050_k64_s009`** | `4_IVF_Flat_AVX2` | 130.4 | 151.0 | 132.4 | 0.0% | 578.5 | 578.9 | `100.0%` / `100.0%` |
| **`M5050_k64_s009`** | `5A_IVF_PartPrune_AVX2` | **130.8** | **138.2** | **131.3** | 0.0% | **450.4** | **472.1** | `100.0%` / `100.0%` |
| **`M5050_k64_s009`** | `5B_IVF_FullExactPrune_AVX2` | 206.3 | 234.5 | 168.8 | 45.5% | 594.8 | 536.6 | `100.0%` / `100.0%` |
| **`D_k64_s009_Adv`** | `0_Ref_Hito9_ExactScan` | 167.6 | 170.5 | 168.1 | 0.0% | 500.7 | 498.8 | `100.0%` / `100.0%` |
| **`D_k64_s009_Adv`** | `2B_Flat_AVX2_AoS_4x` | 144.5 | **148.3** | 145.1 | 0.0% | 469.4 | 493.3 | `100.0%` / `100.0%` |
| **`D_k64_s009_Adv`** | `4_IVF_Flat_AVX2` | **133.2** | 160.5 | **137.0** | 0.0% | 468.7 | 494.6 | `100.0%` / `100.0%` |
| **`D_k64_s009_Adv`** | `5A_IVF_PartPrune_AVX2` | 138.2 | 192.7 | 152.9 | 0.0% | **445.2** | **461.4** | `100.0%` / `100.0%` |
| **`D_k64_s009_Adv`** | `5B_IVF_FullExactPrune_AVX2` | 220.6 | 236.0 | 220.4 | 24.2% | 549.1 | 561.2 | `100.0%` / `100.0%` |

---

## 5. Caracterización de Latencia de Cola (`p50` vs `p95`/`p99`) y Conclusiones Físicas para Hito 2B

1. **Por qué `SoA` (`3A`) y `AoSoA` (`3B`) pierden bajo filtrado del 15%:**  
   Cuando el predicado selecciona el $15\%$ de los vectores, un bloque SIMD vertical de 8 vectores consecutivos contiene en promedio solo $8 \times 0.15 = 1.2$ vectores elegibles. Por tanto, los planes `3A` y `3B` desperdician el **$85\%$ de sus carriles AVX2** calculando coordenadas de vectores no elegibles (evaluando $72{,}568$ vectores en bloques activos frente a los $15{,}077$ elegibles). En cambio, el recorrido indexado por `eligible_ids` en `AoS` (`2B` y `4`) aprovecha el $100\%$ del cómputo en vectores elegibles.
2. **Ganancia pura por localidad física (`Plan 4` vs `Plan 2B`):**  
   Sin podar un solo vector (`0.0%` podados), reordenar físicamente los vectores por partición geométrica (`Plan 4`) reduce el tiempo de cómputo en `WARM_CACHE` de **145.2 µs (`2B`) a 128.1 µs (`4`)** (y el total a **130.4–137.0 µs** en todos los escenarios, incluso el adversarial difuso), elevando el ancho de banda efectivo a **~30–31.5 GB/s**.
3. **Caracterización del régimen `p50` vs `p95`/`p99` en `Plan 5A` y `Plan 5B`:**  
   - **En `R_k64_s003_Tight` (`WARM_CACHE`):** `5A` alcanza `p50 = 12.6 µs` y `Mean = 26.5 µs`, pero `p95 = 117.6 µs` (`p99 = 132.6 µs`). Esto ocurre porque la mayoría de las consultas caen cerca del núcleo de un cluster (cerrando rápidamente la cota $r_K$ en la primera partición y podando $>95\%$ del corpus en `~12 µs`), mientras que el $5\%$ de las consultas cae en regiones inter-cluster donde $r_K$ es mayor y obliga a recorrer múltiples particiones vecinas. Aun así, incluso en `p95` (`117.6 µs`) y `p99` (`132.6 µs`), `5A` sigue siendo significativamente más rápido que `Plan 0` (`171.2 µs` / `174.1 µs`).
   - **En `R_k64_s009_Base` (`WARM_CACHE`):** `5B` mejora tanto la mediana (`p50 = 108.0 µs` vs `127.2 µs` de `Plan 4` y `167.8 µs` de `Plan 0`) como la media (`Mean = 121.4 µs` vs `131.2 µs`), pero su cola `p95` sube a **`229.7 µs`** (`p99 = 241.9 µs`) frente a **`136.0 µs`** de `5A` (o **`149.2 µs`** de `2B`). La causa física es directa: para el $\sim 10\%$ de las consultas situadas en zonas difusas de alta superposición, la cascada escalar por candidato (`radio -> 16D -> 32D -> 64D` + `l2_hito9_ref`) no logra podar suficientes vectores para amortizar su sobrecarga de instrucciones y predicción de saltos.
   - **En `M5050_k64_s009` (bimodal):** El $50\%$ de las consultas son agrupadas (donde `5B` tarda `~95–115 µs`) y el $50\%$ son difusas (donde `5B` tarda `~206–235 µs`), produciendo una distribución bimodal donde la media de `5B` (`168.8 µs`) iguala a `Plan 0` pero su mediana (`206.3 µs`) y `p95` (`234.5 µs`) pierden frente a `5A` (`p50 = 130.8 µs`, `p95 = 138.2 µs`).
4. **Implicación directa para el Selector Adaptativo de Hito 2B:**  
   La divergencia entre `p50` y `p95` en `5B` demuestra que un selector adaptativo en el **Hito 2B** no debe elegir un plan estático por dataset, sino **por consulta individual** usando las distancias de la consulta a los centroides $d(q, c_{(1)}), d(q, c_{(2)}), \dots$ calculadas en los primeros `~3 µs` de `Prep`:
   - Si $d(q, c_{(1)})$ indica que la consulta cae dentro de un núcleo compacto ($d(q, c_{(1)}) \ll d(q, c_{(2)})$), despachar `5A`/`5B`.
   - Si la brecha de cuantización $d(q, c_{(2)}) / d(q, c_{(1)}) \approx 1$ o la tasa de poda en las primeras 2 particiones visitadas cae por debajo del umbral de amortización ($\sim 55\%$), **conmutar inmediatamente al barrido AVX2 4x sin sobrecarga de poda (`Plan 4` / `Plan 5A`)**, acotando el `p95`/`p99` al nivel de `Plan 4/5A` (`~136–155 µs`) mientras se conserva la aceleración de `5B` en `p50` y `Mean`.

---

## 6. Reproducción Exacta

```powershell
# 1. Compilar y ejecutar la Suite de Certificación Matemática de Exactitud (Hito 2A.1)
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
