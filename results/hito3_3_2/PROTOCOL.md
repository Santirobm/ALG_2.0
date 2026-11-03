# Hito 3.3.2 — Protocolo de Aislamiento de Kernel y Escalabilidad Dimensional

**Fecha:** 3 de Noviembre de 2026  
**Rama:** `hito3.3.2-kernel-isolation`  
**Objetivo:** Experimento de decisión para responder de forma definitiva si el motor de cálculo y selección de ALG posee una ventaja intrínseca reproducible frente a FAISS cuando ambos operan bajo idénticas condiciones físicas de memoria, o si la ventaja dependía de asimetrías de layout o filtrado.

---

## 1. Variantes en Evaluación Simétrica

Para eliminar toda asimetría de acceso a memoria, se definen tres variantes con roles claramente delimitados:

| Variante | Fuente de Datos | Motor | Rol Metodológico |
| :--- | :--- | :--- | :--- |
| **A1** | `corpus_vectors` original vía array `eligible_ids` | ALG AVX2 4x + TopKHeap | Mide el coste de recolección de memoria (gather/strided) sobre corpus completo. |
| **A2** | Buffer contiguo `compact_vectors` ($M \times D$) | ALG AVX2 4x + TopKHeap | **Núcleo de decisión:** Mide el rendimiento puro de ALG con localidad física controlada. |
| **C1** | El **mismo** buffer contiguo `compact_vectors` | FAISS Flat L2 (`IndexFlatL2`) | **Control simétrico:** Búsqueda nativa de FAISS sobre idéntica disposición de memoria. |

### Controles No Negociables
1. **Identidad Binaria de Entrada:** Antes de cronometrar, una rutina de aserción valida que para cada candidato $j \in [0..M-1]$, el vector consumido por A2 y C1 es idéntico byte a byte (`memcmp == 0`).
2. **Idéntico Mapeo de Identificadores:** Ambos motores devuelven índices locales que se remapean exactamente a través del mismo array `eligible_ids`.
3. **Oráculo binary64 Corregido:** Ground Truth en doble precisión con resolución determinista de empates (`DoubleTopKHeap` con comparador estricto por ID).
4. **Plan 5A Congelado:** Ningún archivo de Plan 5A es modificado.

---

## 2. Metodología de Medición Independiente (Sin Contaminación)

Para evitar que la instrumentación altere el comportamiento de la caché L1/L2 y los cauces de ejecución superscalares:
- **Medición Primaria (End-to-End):** Se ejecuta sin cronómetros internos intermedios. Mide la latencia operacional completa de cada consulta bajo rotación balanceada ($A_1 \to A_2 \to C_1$).
- **Medición de Componentes (Micro-benchmark Separado):**
  - Pasada de **Distancia Pura:** Evalúa únicamente la distancia euclídea a todos los candidatos sin inserción en heap.
  - Pasada de **Selección Top-K Pura:** Mide el tiempo de filtrado y ordenación.
- **Prohibición Expresa:** Queda prohibido inferir el coste del heap restando cronómetros dentro de la misma llamada ($T_{\text{heap}} \ne T_{\text{total}} - T_{\text{dist}}$).

---

## 3. Dimensiones y Espacio Experimental

- **Dimensiones:** $D \in \{64, 128, 256\}$ con embeddings reales de SimpleWiki MiniLM-L6-v2.
- **Selectividades:** $1\%$, $10\%$, $100\%$.
- **Valores de $K$:** $K \in \{10, 100\}$.
- **Regímenes de Caché:** `WARM_CACHE` y `COLD_LLC_FLUSH` (36 MB).
- **Volumen Estadístico:** $10.000$ consultas reservadas por condición en la fase de evaluación, registrando la distribución completa de latencias:
  - Media, Desviación Estándar, Mediana ($p50$), $p95$, $p99$, QPS.
  - `IDSetRecall` y `StrictRankMatch` frente al oráculo binary64.

---

## 4. Criterio de Cierre y Regla de Parada

- Si **A2 supera a C1** de forma consistente y reproducible ($\ge 1.15\times$) en la mayoría de las condiciones simétricas, se valida la existencia de un núcleo de cálculo defendible en ALG para continuar su desarrollo.
- Si **A2 empata o pierde contra C1** ($A_2 / C_1 \le 1.05\times$), se certifica que ALG no posee ventaja computacional en el nivel Flat, y se detiene la inversión en micro-optimizaciones del kernel para pivotar la dirección del proyecto.
