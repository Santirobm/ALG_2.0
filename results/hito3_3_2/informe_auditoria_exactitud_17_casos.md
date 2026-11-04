# Auditoría Forense Exhaustiva de Exactitud (17 Casos Discrepantes)

**Fecha:** 4 de Noviembre de 2026  
**Kernel Verificado:** `l2_sq_avx2_4x_aos()` de producción  
**Oráculo:** binary64 (`double`) con desempate determinista por ID  
**Total Consultas Evaluadas:** 9.000 ($3 \text{ dimensiones} \times 3 \text{ selectividades} \times 1.000 \text{ queries}$, Top-100)  

## 1. Tabla de Discrepancias Identificadas

| # | Dim | Sel | Consulta | Rango | ID (GT) | ID (A2) | Distancia binary64 (GT) | Distancia binary64 (A2) | Hex float32 4x (GT) | Hex float32 4x (A2) | $\Delta$ ULP | Set Match (100 Vecinos) |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 1 | 64 | 10% | #165 | 89 | 80871 | 50563 | 1.074160736 | 1.074160834 | 0x3f897e1a | 0x3f897e1a | 0 ULP | 100% (Identico) |
| 2 | 64 | 10% | #655 | 79 | 70470 | 184 | 1.178799421 | 1.178799508 | 0x3f96e2e6 | 0x3f96e2e6 | 0 ULP | 100% (Identico) |
| 3 | 64 | 10% | #701 | 29 | 40162 | 30705 | 1.13805864 | 1.138058681 | 0x3f91abe8 | 0x3f91abe8 | 0 ULP | 100% (Identico) |
| 4 | 64 | 100% | #618 | 68 | 63768 | 24371 | 0.6685276198 | 0.6685276419 | 0x3f2b24a0 | 0x3f2b24a0 | 0 ULP | 100% (Identico) |
| 5 | 128 | 1% | #69 | 82 | 10046 | 80093 | 1.722981991 | 1.722982062 | 0x3fdc8aad | 0x3fdc8aac | 1 ULP | 100% (Identico) |
| 6 | 128 | 1% | #523 | 29 | 80000 | 30075 | 1.619576566 | 1.61957663 | 0x3fcf4e4a | 0x3fcf4e4a | 0 ULP | 100% (Identico) |
| 7 | 128 | 10% | #215 | 53 | 50200 | 20291 | 1.353944129 | 1.353944297 | 0x3fad4e0c | 0x3fad4e0c | 0 ULP | 100% (Identico) |
| 8 | 128 | 10% | #899 | 74 | 50365 | 10016 | 1.337559277 | 1.337559292 | 0x3fab3524 | 0x3fab3524 | 0 ULP | 100% (Identico) |
| 9 | 128 | 100% | #694 | 94 | 61998 | 21656 | 1.224905943 | 1.224906059 | 0x3f9cc9b8 | 0x3f9cc9b8 | 0 ULP | 100% (Identico) |
| 10 | 128 | 100% | #777 | 65 | 72389 | 46572 | 1.071002 | 1.071002044 | 0x3f891698 | 0x3f891698 | 0 ULP | 100% (Identico) |
| 11 | 256 | 1% | #357 | 7 | 20032 | 27 | 1.201551162 | 1.201551316 | 0x3f99cc6f | 0x3f99cc6f | 0 ULP | 100% (Identico) |
| 12 | 256 | 1% | #984 | 73 | 50040 | 23 | 1.740717066 | 1.740717158 | 0x3fdecfd2 | 0x3fdecfd2 | 0 ULP | 100% (Identico) |
| 13 | 256 | 10% | #357 | 41 | 20032 | 27 | 1.201551162 | 1.201551316 | 0x3f99cc6f | 0x3f99cc6f | 0 ULP | 100% (Identico) |
| 14 | 256 | 10% | #444 | 89 | 60026 | 873 | 1.279037494 | 1.279037546 | 0x3fa3b780 | 0x3fa3b780 | 0 ULP | 100% (Identico) |
| 15 | 256 | 100% | #226 | 20 | 34592 | 15878 | 1.018961876 | 1.018961907 | 0x3f826d58 | 0x3f826d58 | 0 ULP | 100% (Identico) |
| 16 | 256 | 100% | #649 | 60 | 88748 | 46937 | 1.293269464 | 1.293269504 | 0x3fa589da | 0x3fa589da | 0 ULP | 100% (Identico) |
| 17 | 256 | 100% | #921 | 34 | 77752 | 5754 | 1.164496544 | 1.164496736 | 0x3f950e3a | 0x3f950e3a | 0 ULP | 100% (Identico) |

## 2. Resumen Estadístico y Dictamen Numérico

- **Total de Consultas Evaluadas:** 9.000
- **Coincidencia Exacta de Ranking (StrictRankMatch):** 8983 / 9.000 (**99.81%**)
- **Consultas con IDSetRecall = 100% entre las 17 discrepantes:** 17 / 17 (**100.0%**)
- **Clasificación por Precisión Numérica:**
  - **0 ULP (Empates Binarios Idénticos en float32 AVX2 4x):** 16 casos
  - **1 ULP (Redondeo de Simple Precisión en Frontera IEEE-754):** 1 casos
  - **2 ULP (Acumulación dimensional de precisión finita):** 0 casos

### Dictamen de Exactitud
1. **Cero Vecinos Perdidos:** En las 17 consultas con inversión posicional, los 100 identificadores devueltos son exactamente los mismos 100 que calculó el oráculo binary64. No se omitió ningún candidato relevante.
2. **Resolución de Empates:** Cuando dos vectores distan en binary64 por una cantidad menor a $\epsilon_{mach} \approx 1.192 \times 10^{-7}$, el acumulador vectorizado `_mm256_fmadd_ps` produce exactamente la misma palabra de 32 bits (0 ULP) o bits adyacentes (1 ULP). El orden devuelto es consecuencia directa de la resolución por ID sobre palabras idénticas.
