# Informe de Auditoría Forense: Discrepancia Posicional en Hito 3.3 (Consulta #65)

**Fecha:** 3 de Noviembre de 2026  
**Rama Git:** `hito3.3.1-benchmark-integrity`  
**Objetivo:** Análisis a nivel de bit y palabras hexadecimales IEEE 754 del caso con `StrictRankMatch = 99.00%` detectado en embeddings reales ($D=64$, Top-100, Selectividad 10%, COLD).

---

## 1. Identificación de la Consulta y Candidatos
- **Archivo ejecutor de auditoría:** [`benchmarks/realworld/inspect_hex_bits.cpp`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/benchmarks/realworld/inspect_hex_bits.cpp).
- **Consulta:** Índice relativo `#65` de la serie COLD (`q_idx = 165` del dataset global).
- **Candidatos discrepantes:** Vector `A` (`id=50563`) y Vector `B` (`id=80871`) en las posiciones 89 y 90 del Top-100.
- **Resultado Global de Vecindad:** `IDSetRecall = 100.00%` (los 100 vecinos recuperados son idénticos).

---

## 2. Inspección Hexadecimal Bit a Bit (Palabras IEEE 754)

Se evaluó la acumulación de distancias cuadradas sobre los mismos vectores ejecutando directamente los kernels de producción de ALG y el oráculo:

| Modo de Cómputo | Candidato A (ID: 50563) | Candidato B (ID: 80871) | Diferencia Hexadecimal | Diferencia Numérica |
| :--- | :---: | :---: | :---: | :---: |
| **Double binary64 (Oráculo)** | `1.0741608335483388`<br>`0x3ff12fc3452bbf58` | `1.0741607358890903`<br>`0x3ff12fc32af4a7ad` | Distintas en bits de mantisa 25..52 | $\Delta d^2 = 9.7659 \times 10^{-8}$ |
| **Kernel Producción 4x (`l2_sq_avx2_4x_aos`)** | `1.07416081`<br>**`0x3f897e1a`** | `1.07416081`<br>**`0x3f897e1a`** | **COINCIDENCIA EXACTA BIT A BIT** | **0 ULP (Empate estricto)** |
| **Kernel Producción 1x (`l2_sq_avx2_1x_aos`)** | `1.07416081`<br>**`0x3f897e1a`** | `1.07416081`<br>**`0x3f897e1a`** | **COINCIDENCIA EXACTA BIT A BIT** | **0 ULP (Empate estricto)** |
| **Float32 Escalar (sin FMA)** | `1.07416081`<br>`0x3f897e1a` | `1.07416070`<br>`0x3f897e19` | Diferencia en bit menos significativo | 1 ULP ($\approx 1.192 \times 10^{-7}$) |

---

## 3. Resolución Definitiva del Mecanismo Causal

1. **Empate Estricto en el Kernel AVX2 de Plan 5A:**
   El kernel vectorizado de Plan 5A emplea instrucciones fusionadas `_mm256_fmadd_ps` (FMA). Debido a que FMA realiza un único redondeo final en lugar de redondeos intermedios en la multiplicación y suma, el cálculo de la distancia cuadrada euclídea para ambos candidatos produce **exactamente la misma palabra binaria de 32 bits: `0x3f897e1a`**.
   
2. **Origen de la Aparente Discrepancia en la Tabla:**
   La visualización de `1.0741607` vs `1.0741608` provenía de la rutina auxiliar de verificación escalar que realizaba sumas no fusionadas (`mul + add` por separado), cruzando la frontera de redondeo en 1 ULP (`0x3f897e19` vs `0x3f897e1a`). En el hardware AVX2 real de ALG Plan 5A, **la distancia almacenada en el heap para ambos candidatos es idéntica bit a bit**.

3. **Desempate por ID vs Desempate Binary64:**
   - Para el oráculo de 64 bits (`double`), el Candidato B (`80871`) es menor por $9.7659 \times 10^{-8}$, ubicándose en el rango 89.
   - Para ALG Plan 5A (AVX2), ambas distancias son `0x3f897e1a`. Al detectar igualdad numérica estricta, la política de inserción del heap desempata por orden de inserción/ID, colocando primero al Candidato A (`50563`).

**Conclusión:** Se certifica empíricamente que la permutación posicional en la consulta #65 no es un error de ordenamiento ni de aproximación, sino un **empate binario exacto (0 ULP)** en la precisión $float32$ del hardware AVX2.
