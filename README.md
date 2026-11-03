# ALG-Flat 2.0: Motor Vectorial Exacto en Espacio Plano con Poda Geométrica Certificada en IEEE-754

> **Estado del Proyecto:** **Hito 1 + Hito 2A.3 Completados y Certificados** (Ablación Física de 9 Planes + Contrato Formal de Entorno IEEE-754 `MXCSR` y Dominio de Coordenadas con Fallback Automático a `ExactScan` + Aritmética de Intervalos con Redondeo hacia Afuera en `binary64` verificada contra Oráculo en `long double` + Kernel Unificado Templado de 5 Etapas con **Cero Tolerancia Oculta** $\Delta = LB_{\text{safe}}^2 - d_{\text{orig}}^2 \le 0$).  
> **Barrera Metodológica:** **Hito 2A.3** cerrado en `main` antes de iniciar **Hito 2B** (Selector Adaptativo Ligero en rama separada) y manteniendo **Hito 2C** bloqueado.

---

## 1. Motivación Física: Del Diagnóstico de Hito 9 a ALG-Flat 2.0

En el cierre del proyecto anterior ([`ALG` Hito 9](https://github.com/Santirobm/ALG)), demostramos que sobre un corpus de $N = 100{,}000$ vectores en $D = 64$ dimensiones (`float32`), un escaneo plano secuencial (`ExactScan`) promediaba **~159–172 µs**, superando ampliamente la navegación aproximada sobre grafos `HNSW` (**~450–830 µs**).

El análisis físico de **ALG-Flat 2.0** parte de una precisión fundamental sobre la jerarquía de memoria del procesador utilizado (**Intel Core i5-13500**, con **24 MB de Intel Smart Cache L3**):
- **El corpus completo ocupa $100{,}000 \times 64 \times 4\text{ bytes} = \mathbf{25.6\text{ MB}}$**, lo cual **excede ligeramente los 24 MB de caché L3** (especialmente al sumar metadatos, listas de elegibles, buffers del proceso y competencia por vías de caché).
- Sin embargo, bajo una selectividad del $15\%$ ($15{,}077$ vectores elegibles), los vectores candidatos ocupan únicamente **$3.86\text{ MB}$**, y aun en recorridos más amplios, una fracción sustancial reside en L3 mientras el resto se beneficia del prefetcher de hardware L2/L3 y de la ausencia de *pointer chasing* aleatorio.

**Hipótesis Central de ALG-Flat 2.0:**
> En lugar de usar la geometría de los datos para navegar un grafo aproximado (sufriendo fallos de caché L3/DRAM impredecibles y pérdida de exactitud), podemos usar **organización física contigua en memoria + cotas geométricas exactas certificadas en IEEE-754 `float32`** para **descartar particiones y dimensiones redundantes sin perder un solo vecino verdadero (`100.0000%` `IDSetRecall` y `100.0000%` `RankMatch`)**.

---

## 2. Certificación Matemática de Exactitud (Hito 2A → Hito 2A.1 → Hito 2A.2 → Hito 2A.3)

Para que un motor con poda geométrica (`Plan 5A` y `Plan 5B`) pueda declararse **matemáticamente exacto** en aritmética finita IEEE-754 `binary32` (`float32`), en el cierre definitivo **Hito 2A.3** ([`include/ieee754_bounds.h`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/include/ieee754_bounds.h), [`src/physical_layouts.cpp`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/src/physical_layouts.cpp), [`src/execution_plans.cpp`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/src/execution_plans.cpp), [`tests/test_exactness.cpp`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/tests/test_exactness.cpp)) se implementaron y verificaron ocho garantías formales:

### 2.1. Contrato Formal en Tiempo de Ejecución: Estado `MXCSR` y Dominio de Coordenadas con Fallback a `ExactScan`
Toda demostración en punto flotante IEEE-754 depende de hipótesis sobre el entorno de ejecución y el rango numérico de las entradas. En **Hito 2A.3**, estas hipótesis dejan de ser supuestos implícitos y se verifican activamente en tiempo de ejecución:
1. **Validación del registro `MXCSR` ([`check_ieee754_mxcsr_state()`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/include/ieee754_bounds.h#L103-L115)):** Verifica mediante `_mm_getcsr()` antes de construir el índice y al inicio de cada consulta con poda (`Plan 5A` y `Plan 5B`) que:
   - El modo de redondeo activo sea **Round-to-Nearest-Even** (`bits [14:13] == 0x0000`).
   - **Flush-To-Zero (`FTZ`, bit 15 `0x8000`)** esté desactivado (`0`).
   - **Denormals-Are-Zero (`DAZ`, bit 6 `0x0040`)** esté desactivado (`0`).
2. **Validación SIMD del dominio de coordenadas ([`check_vector_domain_certified()`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/include/ieee754_bounds.h#L123-L152)):** Verifica con AVX2 que toda coordenada $x_d$ del corpus y de la consulta sea finita (`!isnan && !isinf`) y cumpla $|x_d| \le M_{\max} = 1.0 \times 10^{15}$ (`CERTIFIED_MAX_COORD_ABS`). Esto garantiza que $64 (2 M_{\max})^2 = 2.56 \times 10^{32} \ll \text{FLT\_MAX} \approx 3.4028 \times 10^{38}$ (imposibilitando *overflow* en distancias o rotaciones PCA), mientras que cualquier *underflow* gradual en cuadrados de diferencias subnormales queda cubierto por el término aditivo exacto $\text{SUBNORMAL\_SQ\_SLACK\_F64} = 64 \times \text{FLT\_TRUE\_MIN} \approx 8.97 \times 10^{-44}$.
3. **Fallback automático seguro ([`execute_ivf_exact_scan_fallback()`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/src/execution_plans.cpp#L649-L699)):** Si un proceso host altera `MXCSR` (por ejemplo, activando `FTZ=1` o cambiando el modo de redondeo) o si una consulta contiene coordenadas fuera de $[-10^{15}, 10^{15}]$, `Plan 5A` y `Plan 5B` desactivan inmediatamente toda poda geométrica, ejecutan el barrido exacto sobre las coordenadas originales y registran `metrics.fallback_triggered = 1`.

### 2.2. Invariancia Traslacional: Residuos Rotados por Partición
Cada partición $p$ con centroide $c_p$ almacena directamente el **residuo local rotado respecto de su propio centroide**:
$$\tilde{x}_p = \hat{Q}^T (x - c_p), \qquad r_{48}(x, p) = \|\tilde{x}_{p, 16..63}\|_2, \qquad r_{32}(x, p) = \|\tilde{x}_{p, 32..63}\|_2$$
En tiempo de consulta, una única rotación global $\tilde{q} = \hat{Q}^T (q - \mu)$ permite obtener en $\mathcal{O}(D)$ por partición visitada:
$$\tilde{q}_p = \tilde{q} - \widehat{Q^T(c_p - \mu)} \approx \hat{Q}^T (q - c_p)$$
Así, tanto en las primeras $16/32$ dimensiones como en las colas $16..63$ y $32..63$, la resta se realiza en el mismo sistema de referencia local centrado en $c_p$:
$$\tilde{q}_p - \tilde{x}_p = \hat{Q}^T(q - c_p) - \hat{Q}^T(x - c_p) = \hat{Q}^T(q - x)$$

### 2.3. Aritmética de Intervalos con Redondeo hacia Afuera en `binary64` y Conversión Dirigida `round_up_f32` / `round_down_f32`
En **Hito 2A.3** se reemplazó cualquier margen heurístico fijo en `binary64` por **aritmética de intervalos con redondeo dirigido hacia afuera en cada operación elemental `binary64`** ([`include/ieee754_bounds.h`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/include/ieee754_bounds.h#L166-L246)), seguida de conversión dirigida a `binary32`:
- **Primitivas de intervalo en `binary64` (`1 ULP` exacto por operación elemental):** Como según IEEE-754 toda operación elemental (`+`, `-`, `*`, `/`, `sqrt`) en modo Round-to-Nearest-Even tiene error $\le \frac{1}{2}\text{ULP}$, avanzar o retroceder exactamente `1 ULP` en la representación entera IEEE-754 de 64 bits (`next_ulp_up_pos_f64` / `next_ulp_down_pos_f64`) garantiza rigurosamente:
  $$\operatorname{add\_up\_f64}(a, b) \ge a + b, \quad \operatorname{mul\_up\_f64}(a, b) \ge a \cdot b, \quad \operatorname{div\_up\_f64}(a, b) \ge \frac{a}{b}, \quad \operatorname{sqrt\_up\_f64}(a) \ge \sqrt{a}$$
  $$\operatorname{sub\_down\_f64}(a, b) \le \max(0, a - b), \quad \operatorname{mul\_down\_f64}(a, b) \le a \cdot b, \quad \operatorname{div\_down\_f64}(a, b) \le \frac{a}{b}, \quad \operatorname{sqrt\_down\_f64}(a) \le \sqrt{a}$$
- **Cotas superiores certificadas para todas las entradas de construcción ([`src/physical_layouts.cpp`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/src/physical_layouts.cpp#L279-L497)):**
  - $\delta_{\text{ortho}} = \|\hat{Q}\hat{Q}^T - I_{64}\|_F$ se acumula con `add_up_f64` y `sqrt_up_f64`, y se almacena con `round_up_f32`.
  - $\|c_p - \mu\|_2$, $\|q - \mu\|_2$, $r_{\max}(p)$, $r_{48}(x, p)$ y $r_{32}(x, p)$ se acumulan con `add_up_f64`, `mul_up_f64`, `sqrt_up_f64` y `round_up_f32`, garantizando que toda magnitud que alimenta una holgura sea una **cota superior matemática verdadera**.
  - $r_{\min}(p)$ se acumula hacia abajo y se almacena con `round_down_f32`, garantizando una **cota inferior matemática verdadera**.

### 2.4. Presupuesto Explícito de Operaciones en `binary64` y Derivación Formal por Etapa (`16D` $\to \gamma_{17}$, `32D` $\to \gamma_{33}$, `64D` $\to \gamma_{67}$)
En [`include/ieee754_bounds.h`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/include/ieee754_bounds.h#L271-L495), cada expresión tiene documentado su conteo exacto de operaciones `binary64` ($\le 7$ operaciones elementales por expresión, todas envueltas en primitivas de intervalo hacia afuera):
1. **Constantes $\gamma_n = \operatorname{div\_up\_f64}(n u, \operatorname{sub\_down\_f64}(1, n u))$ (2 ops `binary64`):** Con $u = 2^{-24}$, se obtienen cotas superiores estrictas para $\gamma_{17} = 1.01328 \times 10^{-6}$, $\gamma_{33} = 1.96696 \times 10^{-6}$, $\gamma_{64} = 3.81471 \times 10^{-6}$, $\gamma_{66} = 3.93392 \times 10^{-6}$ y $\gamma_{67} = 3.99353 \times 10^{-6}$.
2. **Perturbación L2 del vector diferencia rotado $\delta_{\text{vec}}(q, p)$ (7 ops `binary64` hacia $+\infty$):**
   $$\bigl\|(\tilde{q}_p - \tilde{x}_p)_S - (\hat{Q}^T(q - x))_S\bigr\|_2 \le \delta_{\text{vec}}(q, p) = \operatorname{round\_up\_f32}\Bigl(\bigl(8\gamma_{64}(1+u) + 4u\bigr)\|q - \mu\|_2^{\text{up}} + 4u\|c_p - \mu\|_2^{\text{up}} + 2u\,r_{\max}^{\text{up}}(p)\Bigr)$$
3. **Holgura lineal de cola (`fp_slack_p`, 6 ops `binary64` hacia $+\infty$):**
   $$\text{fp\_slack}_p = \operatorname{round\_up\_f32}\Bigl((1 + 4u)\bigl(\delta_{\text{vec}}(q, p) + \gamma_{64}(d_c^{\text{high}} + \delta_{\text{vec}}(q, p)) + u\,r_{\max}^{\text{up}}(p)\bigr)\Bigr)$$
4. **Holguras cuadráticas y escalas seguras separadas por etapa ($s \in \{16, 32, 64\}$, 5 ops `binary64` por etapa):**
   Sea $M^{\text{up}} = \operatorname{add\_up\_f64}\bigl(2\,\delta_{\text{vec}}(q, p)\,(d_c^{\text{high}} + r_{\max}^{\text{up}}(p) + \delta_{\text{vec}}(q, p)),\, \text{SUBNORMAL\_SQ\_SLACK\_F64}\bigr)$:
   $$\text{sq\_slack}_{s,p} = \operatorname{round\_up\_f32}\!\left(\operatorname{div\_up\_f64}\!\bigl(M^{\text{up}},\, \operatorname{sub\_down\_f64}(1, \gamma_{\text{depth}(s)})\bigr)\right)$$
   $$s_{\text{safe},s} = \operatorname{round\_down\_f32}\!\left(\operatorname{div\_down\_f64}\!\bigl(\operatorname{mul\_down\_f64}((1 - \gamma_{66})(1 - \gamma_{\text{depth}(s)}),\, 1 - c_s u),\, \operatorname{add\_up\_f64}(1, \delta_{\text{ortho}}^{\text{up}})\bigr)\right)$$
   donde $\text{depth}(16)=17, c_{16}=6$; $\text{depth}(32)=33, c_{32}=6$; y $\text{depth}(64)=67, c_{64}=4$.

### 2.5. Unificación Estructural entre Kernel de Producción y Auditor (`run_plan5b_unified_core<AuditMode>`)
En **Hito 2A.3** se eliminó toda duplicación de código entre `execute_plan5b_ivf_full_exact_prune_avx2` y `audit_plan5b_pruning_soundness`. Ambos invocan exactamente la misma plantilla unificada ([`run_plan5b_unified_core<AuditMode>`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/src/execution_plans.cpp#L911-L1065), [`eval_4x_pca_progressive_unified<AuditMode>`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/src/execution_plans.cpp#L710-L839) y [`eval_1x_pca_progressive_unified<AuditMode>`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/src/execution_plans.cpp#L846-L905)):
- En producción (`AuditMode = false`), las ramas `if constexpr (AuditMode)` desaparecen en tiempo de compilación sin añadir una sola instrucción al camino crítico.
- En auditoría (`AuditMode = true`), el mismo código fuente ejecuta las **5 etapas reales de descarte** (`Part`, `Radial`, `PCA16`, `PCA32`, `PCA64`), haciendo estructuralmente imposible que producción y auditoría diverjan.

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

> **Verificación contra Oráculo en Precisión Extendida (`long double`):** Además de las 4 suites sobre 3.95 millones de candidatos, [`verify_directed_rounding_and_interval_oracle()`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/tests/test_exactness.cpp#L47-L148) somete las cotas a **40 regímenes de escala a lo largo de 12 órdenes de magnitud ($10^{-6}$ a $10^{13}$)**, incluyendo normas nulas, escalas asimétricas y perturbaciones de $\pm 1\text{ ULP}$, comprobando contra un oráculo en `long double` que ninguna cota inferior supera jamás el valor exacto y ninguna holgura queda por debajo del valor exacto.

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
| **Plan 5B** | `5B_IVF_FullExactPrune_AVX2` | `IVF` Centrado + PCA | Cascada unificada `16D -> 32D -> 64D` + dist. original | **5 Etapas: Partición + Radial + PCA 16D/32D/64D** | Ganancia por poda geométrica multinivel exacta certificada en IEEE-754. |

---

## 4. Resultados Experimentales Canónicos Sincronizados (`Seed=8`)

> **Metadatos de Ejecución ([`results/run_metadata.txt`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/results/run_metadata.txt)):**  
> - **CPU:** Intel Core i5-13500 (24 MB L3 Smart Cache), fijado a `P-Core 0` (`AffinityMask=0x1`, `HIGH_PRIORITY_CLASS`).  
> - **SO:** `Windows 10 x86_64 (NT 10.0 Build 19045)`.  
> - **Compilador:** `Clang 22.1.8` (`-O3 -march=native -mavx2 -mfma -std=c++17`).  
> - **Contrato IEEE-754 Verificado:** `MXCSR = 0x1f80` (`RoundToNearestEven=0, FTZ=0, DAZ=0, Contract_Valid=true`), `MaxCoordAbs = 1e+15`.  
> - **Protocolo:** `WARM_CACHE` = 500 consultas medidas tras 20 consultas de calentamiento por plan; `COLD_LLC_FLUSH` = 40 consultas precedidas cada una por un barrido de escritura sobre un buffer de **36 MB** (`> 24 MB L3`) + `_mm_mfence()`.  
> - **Archivo fuente de datos:** Todos los valores de las tablas siguientes provienen exactamente de [`results/ablation_hito1_summary.csv`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/results/ablation_hito1_summary.csv).

### 4.1. Escenario 1: Alta Localidad Geométrica (`R_k64_s003_Tight`, $\sigma = 0.03$, Selectividad $15\%$, Top-$100$)

| Plan de Ejecución | WARM `Prep` (µs) | WARM `Calc` (µs) | WARM `p50` (µs) | WARM `p95` (µs) | WARM `Mean` (µs) | Ciclos / Vec | Vectores Podados | COLD `p50` (µs) | COLD `Mean` (µs) | `IDSetRecall@100` / `RankMatch@100` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0_Ref_Hito9_ExactScan`** | 0.0 | 158.8 | 154.4 | 174.9 | 158.9 | 26.3 | 0.0% | 534.6 | 596.3 | `100.0%` / `100.0%` |
| **`1_Flat_Scalar_AoS`** | 0.0 | 466.8 | 453.8 | 517.8 | 466.8 | 77.2 | 0.0% | 1055.5 | 1090.9 | `100.0%` / `100.0%` |
| **`2A_Flat_AVX2_AoS_1x`** | 0.0 | 157.5 | 157.2 | 165.6 | 157.5 | 26.1 | 0.0% | 513.1 | 525.9 | `100.0%` / `100.0%` |
| **`2B_Flat_AVX2_AoS_4x`** | 0.0 | 164.3 | 160.2 | 206.7 | 164.3 | 27.2 | 0.0% | 486.3 | 511.3 | `100.0%` / `100.0%` |
| **`3A_Flat_AVX2_SoA`** | 0.0 | 753.0 | 702.5 | 998.2 | 753.0 | 124.6 | 0.0% | 1532.6 | 1646.7 | `100.0%` / `100.0%` |
| **`3B_Flat_AVX2_AoSoA`** | 0.0 | 607.4 | 587.4 | 706.4 | 607.4 | 100.5 | 0.0% | 1284.6 | 1303.7 | `100.0%` / `100.0%` |
| **`4_IVF_Flat_AVX2`** | 3.1 | 117.4 | 113.2 | 147.6 | 120.5 | 19.9 | 0.0% | 465.4 | 482.2 | `100.0%` / `100.0%` |
| **`5A_IVF_PartPrune_AVX2`** | 8.4 | 21.4 | **17.9** | **110.1** | **29.8** | **4.9** | **87.2%** | **31.5** | **44.0** | **`100.0%` / `100.0%`** |
| **`5B_IVF_FullExactPrune_AVX2`** | 8.7 | 25.9 | **20.4** | **133.9** | **34.6** | **5.7** | **95.0%** | **40.1** | **52.5** | **`100.0%` / `100.0%`** |

- **Para `Top-10` en este mismo escenario (`WARM_CACHE`):** `5A` alcanza **`p50 = 11.5 µs`** (`Mean = 21.9 µs`, `p95 = 94.1 µs`, **10.9x más rápido** que `Plan 0` en `p50`), y `5B` alcanza **`p50 = 14.0 µs`** (`Mean = 25.7 µs`, `p95 = 101.9 µs`) podando el **96.0%** de los vectores con `100.000%` de `IDSetRecall` y `RankMatch`.
- **En `COLD_LLC_FLUSH` (tras expulsar L3 con 36 MB):** `Plan 0` se degrada a **534.6 µs** (`p50`) / **596.3 µs** (`Mean`), mientras que `5A` (**31.5 µs**, `Mean = 44.0 µs`) y `5B` (**40.1 µs**, `Mean = 52.5 µs`, `p95 = 74.9 µs`) logran un speedup de **13.3x–17.0x en `p50`** (**11.4x–13.5x en media**) porque leen de DRAM únicamente el ~3.4% de las particiones relevantes.

---

### 4.2. Escenario 2: Localidad Moderada (`R_k64_s009_Base`, $\sigma = 0.09$, Selectividad $15\%$, Top-$100$)

En este régimen, el radio del Top-$100$ supera la distancia a los bordes de la mayoría de las particiones, por lo que la poda gruesa por partición (`5A`) logra descartar casi cero vectores ($0.002\%$). Aquí es donde la **poda fina multinivel (cascarón radial + cascada PCA 16D/32D/64D centrada en $c_p$) de `Plan 5B`** demuestra su valor diferencial:

| Plan de Ejecución | WARM `Prep` (µs) | WARM `Calc` (µs) | WARM `p50` (µs) | WARM `p95` (µs) | WARM `Mean` (µs) | Ciclos / Vec | Vectores Podados | COLD `p50` (µs) | COLD `Mean` (µs) | `IDSetRecall@100` / `RankMatch@100` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0_Ref_Hito9_ExactScan`** | 0.0 | 159.3 | 153.9 | 179.2 | 159.4 | 26.4 | 0.0% | 481.7 | 461.4 | `100.0%` / `100.0%` |
| **`2B_Flat_AVX2_AoS_4x`** | 0.0 | 133.0 | 129.2 | 152.9 | 133.1 | 22.0 | 0.0% | 483.2 | 520.3 | `100.0%` / `100.0%` |
| **`4_IVF_Flat_AVX2`** | 3.2 | 125.3 | **127.2** | 149.4 | **128.5** | 21.2 | 0.0% | 501.6 | 516.2 | `100.0%` / `100.0%` |
| **`5A_IVF_PartPrune_AVX2`** | 8.6 | 126.8 | 132.9 | **145.5** | 135.4 | 22.4 | 0.0% | 445.6 | 471.7 | `100.0%` / `100.0%` |
| **`5B_IVF_FullExactPrune_AVX2`** | 9.9 | 139.7 | 132.2 | 260.7 | 149.6 | 24.7 | **66.5%** | **326.5** | **366.2** | **`100.0%` / `100.0%`** |

- **Para `Top-10` en `R_k64_s009_Base` (`WARM_CACHE`):** `5B` poda el **74.3%** de los vectores y reduce la latencia mediana `p50` a **93.0 µs** (`Mean = 107.2 µs`) frente a **125.6 µs** (`Mean = 132.1 µs`) de `Plan 0` y **106.3 µs** (`Mean = 114.2 µs`) de `Plan 4`.
- **Para `Selectividad 25%` (`25,127` elegibles, `Top-100`, `WARM_CACHE`):** `5B` poda el **68.5%** de los candidatos y baja la mediana `p50` a **173.2 µs** (`Mean = 197.9 µs`) frente a **236.5 µs** (`Mean = 248.4 µs`) de `Plan 0` y **181.7 µs** (`Mean = 190.1 µs`) de `Plan 4`.
- **En `COLD_LLC_FLUSH` (`Top-100`, Sel $15\%$):** Mientras `5A` tarda **445.6 µs** (`p50`) / **471.7 µs** (`Mean`) porque no logra podar particiones, `5B` poda el **72.4%** de las evaluaciones completas de 64 dimensiones y baja la latencia en frío a **326.5 µs** (`p50`) / **366.2 µs** (`Mean`).

---

### 4.3. Escenarios Mixto (`M5050_k64_s009`) y Adversarial Difuso (`D_k64_s009_Adv`, Top-$100$, Sel $15\%$)

| Escenario | Plan | WARM `p50` (µs) | WARM `p95` (µs) | WARM `Mean` (µs) | Vectores Podados | COLD `p50` (µs) | COLD `Mean` (µs) | `IDSetRecall@100` / `RankMatch@100` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`M5050_k64_s009`** | `0_Ref_Hito9_ExactScan` | 154.1 | 172.1 | 156.9 | 0.0% | 502.2 | 510.8 | `100.0%` / `100.0%` |
| **`M5050_k64_s009`** | `2B_Flat_AVX2_AoS_4x` | **129.9** | **153.9** | **133.5** | 0.0% | 507.7 | 561.1 | `100.0%` / `100.0%` |
| **`M5050_k64_s009`** | `4_IVF_Flat_AVX2` | 132.3 | 160.6 | 136.0 | 0.0% | **465.4** | **487.3** | `100.0%` / `100.0%` |
| **`M5050_k64_s009`** | `5A_IVF_PartPrune_AVX2` | 137.1 | 163.7 | 140.6 | 0.0% | 467.5 | 490.7 | `100.0%` / `100.0%` |
| **`M5050_k64_s009`** | `5B_IVF_FullExactPrune_AVX2` | 227.0 | 277.0 | 197.1 | 45.5% | 612.3 | 544.4 | `100.0%` / `100.0%` |
| **`D_k64_s009_Adv`** | `0_Ref_Hito9_ExactScan` | 168.7 | 194.7 | 172.4 | 0.0% | 493.5 | 462.9 | `100.0%` / `100.0%` |
| **`D_k64_s009_Adv`** | `2B_Flat_AVX2_AoS_4x` | 129.4 | 183.2 | 142.1 | 0.0% | 479.7 | 499.0 | `100.0%` / `100.0%` |
| **`D_k64_s009_Adv`** | `4_IVF_Flat_AVX2` | **122.0** | 163.7 | 130.7 | 0.0% | **471.4** | **488.0** | `100.0%` / `100.0%` |
| **`D_k64_s009_Adv`** | `5A_IVF_PartPrune_AVX2` | 124.1 | **145.5** | **127.3** | 0.0% | 471.8 | 496.1 | `100.0%` / `100.0%` |
| **`D_k64_s009_Adv`** | `5B_IVF_FullExactPrune_AVX2` | 238.3 | 276.4 | 240.7 | 24.2% | 585.6 | 599.5 | `100.0%` / `100.0%` |

---

## 5. Caracterización de Latencia de Cola (`p50` vs `p95`/`p99`) y Conclusiones Físicas para Hito 2B

1. **Por qué `SoA` (`3A`) y `AoSoA` (`3B`) pierden bajo filtrado del 15%:**  
   Cuando el predicado selecciona el $15\%$ de los vectores, un bloque SIMD vertical de 8 vectores consecutivos contiene en promedio solo $8 \times 0.15 = 1.2$ vectores elegibles. Por tanto, los planes `3A` y `3B` desperdician el **$85\%$ de sus carriles AVX2** calculando coordenadas de vectores no elegibles (evaluando $72{,}568$ vectores en bloques activos frente a los $15{,}077$ elegibles). En cambio, el recorrido indexado por `eligible_ids` en `AoS` (`2B` y `4`) aprovecha el $100\%$ del cómputo en vectores elegibles.
2. **Ganancia pura por localidad física (`Plan 4` vs `Plan 0` / `Plan 2B`):**  
   Sin podar un solo vector (`0.0%` podados), reordenar físicamente los vectores por partición geométrica (`Plan 4`) reduce el tiempo total en `WARM_CACHE` a **120.5–136.0 µs** en todos los escenarios (frente a **156.9–172.4 µs** de `Plan 0`), elevando el ancho de banda efectivo a **~28.8–32.5 GB/s**.
3. **Caracterización del régimen `p50` vs `p95`/`p99` en `Plan 5A` y `Plan 5B`:**  
   - **En `R_k64_s003_Tight` (`WARM_CACHE`):** `5A` alcanza `p50 = 17.9 µs` y `Mean = 29.8 µs`, con `p95 = 110.1 µs` (`p99 = 122.3 µs`). Esto ocurre porque la mayoría de las consultas caen cerca del núcleo de un cluster (cerrando rápidamente la cota $r_K$ en la primera partición y podando $>95\%$ del corpus en `~18 µs`), mientras que el $5\%$ de las consultas cae en regiones inter-cluster donde $r_K$ es mayor y obliga a recorrer múltiples particiones vecinas. Aun así, incluso en `p95` (`110.1 µs`) y `p99` (`122.3 µs`), `5A` sigue siendo significativamente más rápido que `Plan 0` (`174.9 µs` / `188.7 µs`).
   - **En `R_k64_s009_Base` (`WARM_CACHE`):** `5B` domina en `Top-10` (`p50 = 93.0 µs`, `Mean = 107.2 µs` vs `114.2 µs` de `Plan 4`), en `Selectividad 25%` (`p50 = 173.2 µs` vs `181.7 µs` de `Plan 4`) y en `COLD_LLC_FLUSH` (`p50 = 326.5 µs` vs `501.6 µs` de `Plan 4`), pero en `Top-100 WARM` su cola `p95` sube a **`260.7 µs`** frente a **`145.5 µs`** de `5A` (o **`149.4 µs`** de `4`). La causa física es directa: para las consultas situadas en zonas difusas de alta superposición, la cascada por candidato (`radio -> 16D -> 32D -> 64D` + `l2_hito9_ref`) no logra podar suficientes vectores para amortizar su sobrecarga de instrucciones y predicción de saltos.
   - **En `M5050_k64_s009` (bimodal):** El $50\%$ de las consultas son agrupadas y el $50\%$ son difusas, produciendo una distribución bimodal donde `5B` (`p50 = 227.0 µs`, `p95 = 277.0 µs`, `Mean = 197.1 µs`) pierde en promedio frente a `4` (`p50 = 132.3 µs`, `Mean = 136.0 µs`) y `5A` (`p50 = 137.1 µs`, `p95 = 163.7 µs`).
4. **Implicación directa para el Selector Adaptativo de Hito 2B:**  
   La divergencia entre `p50` y `p95` en `5B` demuestra que un selector adaptativo en el **Hito 2B** no debe elegir un plan estático por dataset, sino **por consulta individual** usando las señales geométricas gratuitas disponibles tras la fase inicial de consulta o tras las primeras particiones visitadas:
   - Si la consulta cae dentro de un núcleo compacto o la tasa de poda inicial supera el umbral de amortización, ejecutar `5A`/`5B`.
   - Si la tasa de poda observada cae por debajo del umbral de amortización, **conmutar inmediatamente al barrido AVX2 4x sin sobrecarga de poda (`Plan 4` / `Plan 5A`)**, acotando el `p95`/`p99` al nivel de `Plan 4/5A` mientras se conserva la aceleración de `5A`/`5B` en consultas agrupadas y `COLD_LLC_FLUSH`.

---

## 6. Reproducción Exacta

```powershell
# 1. Compilar y ejecutar la Suite de Certificación Matemática de Exactitud y Contrato IEEE-754 (Hito 2A.3)
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
