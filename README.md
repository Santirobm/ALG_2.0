# ALG Vector Engine

**ALG Vector Engine** es una biblioteca de software independiente de alto rendimiento en C/C++ diseñada para la búsqueda vectorial filtrada exacta en espacios continuos (Euclidean Squared Distance, $L_2^2$).

El motor incorpora despacho dinámico de instrucciones en tiempo de ejecución (SIMD AVX2+FMA frente a fallback escalar portable) y almacenamiento contiguo alineado a 64 bytes (AoS), alcanzando rendimientos superiores a las implementaciones de escaneo vectorial estándar como FAISS Flat sin dependencias externas (cero dependencia de Python, NumPy o FAISS).

---

## 1. Arquitectura y Componentes del Sistema

El diseño de la biblioteca sigue una separación estricta de responsabilidades:

```
ALG_2.0/
├── include/
│   └── alg_engine.h          # C ABI pública estable (manejadores opacos, enums, contratos)
├── src/
│   ├── platform/
│   │   ├── cpu_features.h    # Detección de ISA en runtime vía CPUID/XGETBV (AVX2, FMA)
│   │   ├── cpu_features.cpp
│   │   ├── allocator.h       # Asignador de memoria contigua alineada a 64 bytes
│   │   └── allocator.cpp
│   ├── kernels/
│   │   ├── kernels.h         # Punteros a funciones de kernel y tablas de despacho
│   │   ├── simd_avx2.cpp     # Kernel AVX2+FMA 4x-unrolled (compilado con -mavx2 -mfma)
│   │   └── simd_scalar.cpp   # Fallback escalar portable IEEE-754 (compilado con flags base)
│   └── core/
│       ├── alg_heap.h        # Bounded Max-Heap Top-K con desempate determinista por ID
│       ├── alg_index.h       # Representación interna del índice y almacenamiento contiguo
│       ├── alg_index.cpp     # Gestión del ciclo de vida y asignación del corpus
│       ├── alg_search.h      # Declaraciones de algoritmos de búsqueda y orquestación
│       └── alg_search.cpp    # Búsqueda filtrada (sparse/compact) y barrido completo
├── examples/
│   ├── c_example.c           # Ejemplo de integración en C puro
│   └── cpp_example.cpp       # Ejemplo de integración en C++ moderno (wrapper RAII)
├── tests/
│   └── test_engine_unit.cpp  # Suite de verificación unitaria y regresión numérica
├── tools/
│   └── alg_benchmark_cli.cpp # Arnés de benchmarking CLI de latencia y throughput (QPS)
├── CMakeLists.txt            # Configuración para CMake
└── build.ps1                 # Script de compilación y ejecución directa en Windows (Zig Clang)
```

### Aislamiento de SIMD y Portabilidad
- El núcleo (`core`) y la plataforma (`platform`) se compilan con la ISA base de la arquitectura destino.
- El kernel optimizado de distancia 4x (`src/kernels/simd_avx2.cpp`) se encuentra aislado en una unidad de compilación dedicada que se compila con `-mavx2 -mfma`.
- Durante la inicialización, la biblioteca ejecuta una comprobación CPUID (`alg_platform_detect_cpu_features`) y verifica que el sistema operativo gestione adecuadamente los registros YMM mediante `XGETBV`. Si el procesador anfitrión soporta AVX2 y FMA, se despachan los punteros vectorizados; de lo contrario, se conmuta transparentemente al kernel escalar portable.

---

## 2. Contrato Numérico y Especificación de Ranking

### 2.1 Modo `ALG_SEARCH_FAST_FLOAT32`
1. **Cálculo de Distancia:**
   - La distancia euclídea al cuadrado ($d^2(q, v) = \sum_{d=0}^{D-1} (q_d - v_d)^2$) se evalúa en precisión estándar IEEE 754 `float32`.
   - En CPUs compatibles con AVX2, se evalúan simultáneamente 4 vectores candidatos mediante 4 acumuladores independientes `__m256`, reducidos al final a través de adiciones vectoriales horizontales.
   - El orden de acumulación es fijo y determinista para cada dimensión.

2. **Criterio de Ordenamiento Estricto:**
   - **Clave Primaria:** Menor distancia cuadrática `float32` primero (orden ascendente).
   - **Desempate Determinista:** En caso de igualdad exacta de distancias en float32 (diferencia de 0 ULP), el empate se resuelve de forma determinista favoreciendo al **menor identificador de vector (`id`)**.

3. **Garantía y Precisión frente a `binary64`:**
   - Debido a la no asociatividad de las operaciones en punto flotante a través de múltiples canales SIMD, la máquina $\epsilon_{\text{mach}}$ no constituye una cota de error universal que garantice equivalencia con el cálculo en doble precisión (`binary64`) para distribuciones arbitrarias sin una pasada explícita de intervalo de error.
   - En auditorías exhaustivas sobre embeddings semánticos reales (**SimpleWiki MiniLM-L6-v2**, $D \in \{64, 128, 256\}$ con 9,000 consultas evaluadas), el motor certificó **100.0% de `IDSetRecall`** frente al oráculo `binary64`, confirmándose que las escasas discrepancias de orden estricto correspondieron a empates exactos de 0 ULP de bit o redondeos de 1 ULP adyacente.

### 2.2 Modo `ALG_SEARCH_EXACT_TOPK`
- Reservado para futuras extensiones que implementen refinamiento adaptativo por bandas de incertidumbre numérica. En la versión actual, retorna `ALG_INVALID_ARGUMENT`.

---

## 3. Especificación de la C ABI Pública (`include/alg_engine.h`)

La interfaz pública expone un contrato C99/C11 puro libre de símbolos decorados de C++:

```c
// Tipos de Estado y Retorno
typedef enum AlgStatus {
    ALG_OK                 = 0,
    ALG_INVALID_ARGUMENT   = 1,
    ALG_UNSUPPORTED_ISA    = 2,
    ALG_OUT_OF_MEMORY      = 3,
    ALG_NOT_INITIALIZED    = 4,
    ALG_CAPACITY_EXCEEDED  = 5
} AlgStatus;

// Configuración del Índice
typedef struct AlgIndexConfig {
    uint32_t dimension;        // Dimensión de los vectores (D > 0)
    uint32_t initial_capacity; // Capacidad inicial preasignada (0 para automático)
    AlgMetric metric;          // Solo ALG_METRIC_L2_SQR
} AlgIndexConfig;

// Creación y Destrucción
ALG_API AlgStatus alg_create_index(const AlgIndexConfig* config, AlgIndex** out_index);
ALG_API void alg_free_index(AlgIndex* index);

// Ingesta de Vectores
ALG_API AlgStatus alg_set_vectors(AlgIndex* index, uint32_t num_vectors, const float* vectors);

// Búsqueda Filtrada sobre Conjunto de Candidatos
ALG_API AlgStatus alg_search_compact(
    const AlgIndex* index,
    const float* query_vector,
    const uint32_t* eligible_ids,
    uint32_t eligible_count,
    uint32_t top_k,
    AlgSearchMode search_mode,
    AlgSearchResult* out_results,
    uint32_t* out_count);

// Búsqueda No Filtrada (100% Selectividad)
ALG_API AlgStatus alg_search(
    const AlgIndex* index,
    const float* query_vector,
    uint32_t top_k,
    AlgSearchMode search_mode,
    AlgSearchResult* out_results,
    uint32_t* out_count);

// Búsqueda sobre Buffer Externo Pre-Compactado
ALG_API AlgStatus alg_search_buffer(
    const float* query_vector,
    const float* contiguous_vectors,
    const uint32_t* candidate_ids,
    uint32_t candidate_count,
    uint32_t dimension,
    uint32_t top_k,
    AlgSearchMode search_mode,
    AlgSearchResult* out_results,
    uint32_t* out_count);
```

---

## 4. Compilación y Construcción

### 4.1 Compilación Rápida con Zig C++ (Windows / PowerShell)
El repositorio incluye el script de compilación directa [`build.ps1`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/build.ps1):

```powershell
# Compilar bibliotecas (libalg.a y alg.dll), ejecutar tests y construir ejemplos
.\build.ps1

# Ejecutar adicionalmente el benchmark CLI de rendimiento
.\build.ps1 -RunBenchmark
```

### 4.2 Compilación con CMake (Linux / macOS / Windows)
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
ctest --output-on-failure
```

Artefactos generados en `bin/` o `build/`:
- `libalg.a` / `alg.lib`: Biblioteca estática.
- `alg.dll` / `libalg.so`: Biblioteca dinámica compartida.
- `test_engine_unit`: Ejecutable de pruebas de regresión.
- `c_example`: Ejemplo de consumo en C.
- `cpp_example`: Ejemplo de consumo en C++.
- `alg_benchmark_cli`: Arnés de benchmarking independiente.

---

## 5. Rendimiento y Resultados de Benchmarking

Resultados reproducibles ejecutados con [`tools/alg_benchmark_cli.cpp`](file:///c:/Users/sgi.lm/Desktop/BTC/ALG_2.0/tools/alg_benchmark_cli.cpp) sobre procesador con AVX2+FMA (corpus $N = 100{,}000$ vectores, 200 consultas, Top-10):

| Dimensión | Selectividad | Candidatos | Latencia Media (µs) | p50 (µs) | p95 (µs) | p99 (µs) | QPS |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **64** | 1% | 1,000 | **4.58** | 4.40 | 6.20 | 7.10 | **218,293** |
| **64** | 10% | 10,000 | **77.47** | 72.00 | 102.40 | 128.20 | **12,909** |
| **64** | 100% | 100,000 | **913.48** | 715.20 | 1,543.60 | 1,671.70 | **1,095** |
| **128** | 1% | 1,000 | **11.33** | 11.20 | 12.10 | 13.40 | **88,261** |
| **128** | 10% | 10,000 | **134.47** | 129.10 | 160.70 | 170.90 | **7,437** |
| **128** | 100% | 100,000 | **2,836.26** | 2,462.60 | 4,512.90 | 7,402.70 | **353** |
| **256** | 1% | 1,000 | **24.57** | 24.60 | 25.10 | 26.40 | **40,703** |
| **256** | 10% | 10,000 | **331.02** | 289.90 | 661.90 | 812.20 | **3,021** |
| **256** | 100% | 100,000 | **5,662.05** | 5,388.10 | 7,019.70 | 7,439.70 | **177** |

---

## 6. Historial de Investigación

Las investigaciones exploratorias que fundamentaron la selección de la arquitectura actual se preservan en las ramas históricas del repositorio:
- `main` / `hito2a`: Certificación de cotas geométricas y aritmética de intervalos IEEE-754.
- `hito3.2-generalization-scalability`: Escalabilidad dimensional y distribuciones sintéticas.
- `hito3.3-realworld-validation`: Evaluación sobre embeddings reales de SimpleWiki MiniLM-L6-v2.
- `hito3.3.1-benchmark-integrity`: Descomposición de speedup y ablación de cuatro vías.
- `hito3.3.2-kernel-isolation`: Experimento de decisión final de aislamiento de kernel (A1/A2/C1).
- `product/alg-vector-engine`: **Línea activa de producción de la biblioteca independiente**.
