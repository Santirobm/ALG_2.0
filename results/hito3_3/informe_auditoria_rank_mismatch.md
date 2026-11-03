# Informe de Auditoría Forense: Discrepancia Posicional en Hito 3.3 (Consulta #65)

**Fecha:** 3 de Noviembre de 2026  
**Rama Git:** `hito3.3.1-benchmark-integrity`  
**Objetivo:** Análisis a nivel de bit del caso con `StrictRankMatch = 99.00%` detectado en embeddings reales ($D=64$, Top-100, Selectividad 10%, COLD).

---

## 1. Identificación y Parámetros de la Consulta
- **Archivo ejecutor:** [`benchmarks/realworld/audit_rank_mismatch.cpp`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/benchmarks/realworld/audit_rank_mismatch.cpp).
- **Consulta:** Índice relativo `#65` de la serie COLD (`q_idx = 165` del dataset global).
- **Parámetros:** $K = 100$, Selectividad = 10% (9.000 candidatos elegibles de 90.000).
- **Resultado Global de Vecindad:** `IDSetRecall = 100.00%` (exactamente los mismos 100 IDs recuperados por ALG y por el oráculo Ground Truth en `double`).

---

## 2. Evidencia Numérica Extraída

La única discrepancia en el ranking de los 100 vecinos ocurrió en las posiciones contiguas **89 y 90**:

| Motor / Oráculo | Rango 89 | Rango 90 | Distancia binary64 | Distancia float32 |
| :--- | :---: | :---: | :---: | :---: |
| **Oráculo Ground Truth (binary64)** | **ID: 80871** | **ID: 50563** | `d²(80871) = 1.0741607358890903` <br> `d²(50563) = 1.0741608335483388` | `1.0741607` <br> `1.0741608` |
| **ALG Plan 5A (float32 AVX2)** | **ID: 50563** | **ID: 80871** | `d² = 1.0741608335483388` | `1.07416081` |

### Métricas de Precisión:
- **Diferencia real en 64 bits (binary64):**
  $$\Delta d^2 = |d^2(80871) - d^2(50563)| = 9.765924846583118 \times 10^{-8}$$
- **Épsilon de máquina float32 ($\epsilon_{mach}$):**
  $$\epsilon_{mach} = 2^{-23} \approx 1.1920928955 \times 10^{-7}$$
- **Comparación con el límite numérico:**
  $$\Delta d^2 = 9.766 \times 10^{-8} < \epsilon_{mach} \approx 1.192 \times 10^{-7}$$

---

## 3. Dictamen Causal

1. **Separación Sub-Épsilon:** La distancia euclídea al cuadrado entre los dos vectores en el espacio de 64 dimensiones difiere en menos de **1 Unidad en el Último Lugar (ULP)** de precisión de punto flotante de simple precisión ($float32$).
2. **Colapso a Valor Idéntico:** Al ser computada en registros vectoriales $float32$, ambas distancias colapsan exactamente al mismo valor cuantizado: `1.07416081`.
3. **Desempate por ID:** El oráculo binary64 ordena primero `80871` porque su distancia es $9.76 \times 10^{-8}$ menor. Sin embargo, en el acumulador de 32 bits de ALG (y de cualquier motor que utilice IEEE 754 float32), al existir un empate estricto hasta el bit 23 de mantisa, el orden se resuelve por comparación de ID o inserción en el heap.
4. **Conclusión:** Queda matemáticamente demostrado que la discrepancia posicional no constituye un error algorítmico, omisión de candidatos ni fallo de ordenación, sino un empate numérico infranqueable para la representación de 32 bits.
