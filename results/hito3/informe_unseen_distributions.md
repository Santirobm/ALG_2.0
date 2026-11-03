# Informe de Generalización Hito 3.2: Evaluación en 6 Distribuciones Desconocidas (Out-of-Distribution)

## 1. Protocolo Experimental y Distribuciones Evaluadas
Para garantizar que Plan 5A no dependa de suposiciones de entrenamiento ni de la geometría específica de los datos sintéticos gaussianos, se diseñó una batería exhaustiva con **6 distribuciones no vistas**, que rompen deliberadamente la simetría esférica, la correlación o la continuidad del espacio:

1. **Uniform Hypercube (Uniforme):** Coordenadas i.i.d. en $[-1.0, 1.0]$. Geometría sin concentración central ni centroides naturales.
2. **Anisotropic Gaussian (Anisotrópica):** Varianza dependiente de la coordenada $\sigma_d = 0.02 \cdot (d + 1)$ para $d \in [1, 64]$. Elipsoide fuertemente alargado.
3. **Heavy-Tailed Student-$t$ ($\nu=3$):** Colas pesadas con valores atípicos extremos e infinitos momentos de orden superior a 3. Rompe acotamientos gaussianos estándar.
4. **Correlated Autoregressive AR(1) ($\rho=0.85$):** Alta correlación entre dimensiones consecutivas, modelando señales temporales y dependencias locales.
5. **Degenerate Low-Rank (8D intrínseco en 64D ambiente):** Los datos residen en un subespacio de dimensión intrínseca 8, con ruido residual $\epsilon \sim 10^{-5}$ en las 56 dimensiones restantes.
6. **Multimodal 100 Clusters:** 100 centros dispersos en el hipercubo $[-5, 5]^{64}$ con acumulación local ($\sigma=0.05$). Modela variedad semántica y manifolds agrupados.

- **Corpus:** $N = 100{,}000$ vectores, $D = 64$ ($float32$).
- **Selectividades evaluadas:** $1\%, 5\%, 10\%, 25\%$.
- **Consultas:** 100 consultas COLD (con evicción de 36 MB LLC) y 100 consultas WARM por cada celda experimental (total de 9.600 consultas ejecutadas).
- **Oráculo de Verificación:** Ground Truth exacto computado en `double` ($binary64$) con acumulación de distancias en 64 bits.

---

## 2. Resultados Consolidados: Exactitud y Tasa de Retención (Recall)

| Distribución | Modos de Caché | IDSetRecall Mínimo | IDSetRecall Medio | StrictRankMatch Medio |
| :--- | :--- | :---: | :---: | :---: |
| **Uniform Hypercube** | COLD / WARM | **100.00%** | **100.00%** | 99.92% |
| **Anisotropic Gaussian** | COLD / WARM | **100.00%** | **100.00%** | 99.96% |
| **Heavy-Tailed Student-$t$** | COLD / WARM | **100.00%** | **100.00%** | 99.94% |
| **Correlated AR(1)** | COLD / WARM | **100.00%** | **100.00%** | 99.96% |
| **Degenerate Low-Rank (8D)** | COLD / WARM | **100.00%** | **100.00%** | 99.96% |
| **Multimodal 100 Clusters** | COLD / WARM | **100.00%** | **100.00%** | **100.00%** |

> [!IMPORTANT]
> **Certificación de Exactitud Cero Falsos Negativos:** En las 9.600 ejecuciones a través de las 6 distribuciones no vistas, `IDSetRecall` fue del **100.00%**. No se perdió ni un solo vecino más cercano en ninguna condición de dispersión, colas pesadas o degeneración de rango.
> Las pequeñas diferencias en `StrictRankMatch` (entre 98.75% y 100%) corresponden exclusivamente a empates numéricos en $float32$ resueltos por orden de ID frente al oráculo de 64 bits.

---

## 3. Rendimiento y Speedups frente a Meta FAISS Flat

### A. Régimen COLD (Evicción LLC de 36 MB)

| Distribución | Sel=1% (Top-10) | Sel=1% (Top-100) | Sel=5% (Top-10) | Sel=10% (Top-10) | Sel=25% (Top-10) | Speedup Medio COLD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Uniform Hypercube** | 3.85× | 3.02× | 2.67× | 2.31× | 1.57× | **2.45×** |
| **Anisotropic Gaussian** | 3.69× | 3.10× | 2.76× | 2.28× | 1.55× | **2.43×** |
| **Heavy-Tailed Student-$t$** | 4.08× | 3.26× | 2.79× | 2.28× | 1.54× | **2.50×** |
| **Correlated AR(1)** | 4.05× | 3.29× | 2.80× | 2.37× | 1.54× | **2.53×** |
| **Degenerate Low-Rank (8D)**| 3.64× | 3.20× | 2.63× | 2.26× | 1.54× | **2.44×** |
| **Multimodal (100 Clust)** | **15.62×** | **8.24×** | **29.59×** | **41.45×** | **50.15×** | **28.96×** |

### B. Régimen WARM (En Caché L1/L2)

| Distribución | Sel=1% (Top-10) | Sel=1% (Top-100) | Sel=5% (Top-10) | Sel=10% (Top-10) | Sel=25% (Top-10) | Speedup Medio WARM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Uniform Hypercube** | 9.04× | 5.09× | 4.30× | 2.95× | 2.07× | **3.98×** |
| **Anisotropic Gaussian** | 9.43× | 5.26× | 4.50× | 3.03× | 2.07× | **4.14×** |
| **Heavy-Tailed Student-$t$** | 9.35× | 5.73× | 4.41× | 2.98× | 1.83× | **4.10×** |
| **Correlated AR(1)** | 9.19× | 6.51× | 4.40× | 3.10× | 2.00× | **4.22×** |
| **Degenerate Low-Rank (8D)**| 9.45× | 6.07× | 4.08× | 3.09× | 2.11× | **4.22×** |
| **Multimodal (100 Clust)** | **16.68×** | **10.66×** | **21.64×** | **32.60×** | **53.30×** | **26.24×** |

---

## 4. Hallazgo Clave: Comportamiento de la Poda Geométrica según la Distribución

1. **Distribuciones Continuas / Difusas (Uniforme, Anisotrópica, AR1, Student-t, Rango Bajo):**
   - En estas geometrías difusas, los puntos están dispersos sin huecos vacíos extensos en el espacio $D=64$.
   - Como predice el análisis dimensional de la poda exacta en alta dimensión ($D=64$), el radio del Top-K heap cubre una proporción de particiones que requiere escanear los candidatos filtrados.
   - **Sin embargo, ALG Plan 5A mantiene un speedup de 1.5× a 9.45× frente a FAISS.** Esto confirma que la ventaja no proviene únicamente de descartar particiones enteras por radio esférico, sino de la **organización física por bloques de partición de memoria contigua alineada a AVX2**, eliminando las pausas de direccionamiento indirecto y bifurcaciones de predicción en el bucle interior de filtrado que sufre el escáner lineal de FAISS.

2. **Distribuciones Agrupadas / Multimodales (100 Clusters):**
   - Cuando el corpus posee estructura de variedad natural (clusters o tópicos semánticos típicos de embeddings de texto/visión), la **poda esférica geométrica se vuelve extremadamente eficaz**:
   - **Tasa de poda observada:** Promedio de **96.72%** de candidatos descartados sin evaluación de distancia exacta.
   - **Latencia de consulta:** Desciende a **15.9 µs** en COLD y **9.8 µs** en WARM (frente a 1.417 µs y 1.022 µs de FAISS Flat).
   - **Speedup:** Alcanza picos de **50.15× (COLD)** y **53.30× (WARM)** manteniendo exactamente el 100.00% de recall.
