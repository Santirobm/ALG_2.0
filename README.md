# ALG Vector Engine

ALG Vector Engine es una biblioteca de software en C y C++ para busqueda vectorial exacta (k-NN) bajo distancia euclidea al cuadrado (L2). Esta disenada especificamente para entornos donde las consultas combinan filtros relacionales con similitud vectorial y se requiere maxima velocidad sin la complejidad de dependencias externas.

La biblioteca detecta en tiempo de ejecucion el conjunto de instrucciones del procesador (SIMD AVX2 con FMA o fallback escalar portable), opera sobre memoria contigua alineada a 64 bytes y no depende de Python, BLAS, OpenMP ni frameworks pesados.

## Problema que resuelve

En sistemas modernos de busqueda semantica, recomendacion y analitica, la mayoria de las consultas no se ejecutan sobre el corpus completo, sino sobre subconjuntos filtrados por usuario, fecha, categoria, estado o atributos de negocio.

Bajo estas condiciones:
- Los indices aproximados basados en grafos (como HNSW) pierden eficiencia cuando los filtros reducen la cantidad de candidatos (por ejemplo, al 1% o 10%), debido a caminos desconectados en el grafo y accesos dispersos a memoria.
- Las bibliotecas genericas orientadas a Python introducen sobrecarga de pasaje de datos, gestion de memoria y costo de despacho que superan el tiempo propio del calculo vectorial.

ALG Vector Engine resuelve este escenario concentrandose en el calculo masivo y la seleccion Top-K sobre vectores continuos en memoria, ofreciendo tiempos de respuesta en el orden de microsegundos y una integracion nativa directa en cualquier lenguaje.

## Caracteristicas principales

- Interfaz C ABI pura: Cabecera C99 estable (alg_engine.h) compatible directamente con C, C++, Rust, Go, C# y llamadas FFI.
- Despacho dinamico de instrucciones: Selecciona automaticamente el kernel AVX2+FMA o el camino escalar portable al iniciar, sin necesidad de compilar variantes binarias separadas.
- Cero dependencias externas: Compila directamente con cualquier compilador C++14/C11 estandar (GCC, Clang, MSVC). No requiere OpenMP, MKL, BLAS ni dependencias de terceros.
- Memoria contigua alineada: Los vectores se almacenan alineados a 64 bytes para aprovechar las lineas de cache L1/L2/L3 y los mecanismos de prefetch por hardware.
- Seguridad y robustez: Validaciones de punteros nulos, limites de dimensiones, rechazo estricto de valores no finitos (NaN, Infinito) y proteccion aritmetica contra desbordamiento de enteros (overflow).
- Concurrencia nativa: Los indices son de solo lectura durante la busqueda, permitiendo consultas simultaneas desde multiples hilos sin bloqueos ni contencion.
- Multiplataforma: Soporte verificado para Windows x64 (MSVC, Clang/LLVM, MinGW) y Linux x64 (GCC, Clang).

## Rendimiento

Mediciones obtenidas con el arnes de pruebas incluido (tools/alg_benchmark_cli.cpp) sobre un procesador Intel Core i5-13500 con AVX2+FMA activado.

Corpus: 100,000 vectores sinteticos, 200 consultas, Top-10:

| Dimension | Selectividad | Candidatos evaluados | Latencia media (us) | p50 (us) | p95 (us) | QPS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 64 | 1% | 1,000 | 4.58 | 4.40 | 6.20 | 218,293 |
| 64 | 10% | 10,000 | 77.47 | 72.00 | 102.40 | 12,909 |
| 64 | 100% | 100,000 | 913.48 | 715.20 | 1,543.60 | 1,095 |
| 128 | 1% | 1,000 | 11.33 | 11.20 | 12.10 | 88,261 |
| 128 | 10% | 10,000 | 134.47 | 129.10 | 160.70 | 7,437 |
| 128 | 100% | 100,000 | 2,836.26 | 2,462.60 | 4,512.90 | 353 |
| 256 | 1% | 1,000 | 24.57 | 24.60 | 25.10 | 40,703 |
| 256 | 10% | 10,000 | 331.02 | 289.90 | 661.90 | 3,021 |
| 256 | 100% | 100,000 | 5,662.05 | 5,388.10 | 7,019.70 | 177 |

En conjuntos reducidos (1,000 a 10,000 vectores), las latencias inferiores a 100 microsegundos permiten realizar miles de consultas por segundo por nucleo sin saturar el sistema anfitrion.

## Ejemplo de uso en C

```c
#include <stdio.h>
#include <stdlib.h>
#include "alg_engine.h"

int main(void) {
    const uint32_t dimension = 64;
    const uint32_t num_vectors = 1000;
    const uint32_t top_k = 5;

    // 1. Configuracion y creacion del indice
    AlgIndexConfig config = {
        .dimension = dimension,
        .initial_capacity = num_vectors,
        .metric = ALG_METRIC_L2_SQR
    };

    AlgIndex* index = NULL;
    if (alg_create_index(&config, &index) != ALG_OK) {
        fprintf(stderr, "Error al crear el indice\n");
        return 1;
    }

    // 2. Carga de vectores contiguos en memoria
    float* corpus = (float*)malloc(num_vectors * dimension * sizeof(float));
    // ... inicializar vectores ...
    alg_set_vectors(index, num_vectors, corpus);

    // 3. Busqueda filtrada sobre un subconjunto de candidatos elegibles
    float query[64] = {0};
    uint32_t eligible_ids[] = {10, 25, 42, 99, 150, 320};
    uint32_t eligible_count = sizeof(eligible_ids) / sizeof(eligible_ids[0]);

    AlgSearchResult results[5];
    uint32_t result_count = 0;

    AlgStatus status = alg_search_compact(
        index,
        query,
        eligible_ids,
        eligible_count,
        top_k,
        ALG_SEARCH_FAST_FLOAT32,
        results,
        &result_count
    );

    if (status == ALG_OK) {
        for (uint32_t i = 0; i < result_count; ++i) {
            printf("Rank %u: ID %u, Distancia L2^2: %f\n",
                   i + 1, results[i].id, results[i].distance);
        }
    }

    // 4. Liberacion de recursos
    free(corpus);
    alg_free_index(index);
    return 0;
}
```

## Ejemplo de uso en C++

```cpp
#include <iostream>
#include <vector>
#include "alg_engine.h"

int main() {
    const uint32_t dim = 128;
    const uint32_t count = 5000;

    AlgIndexConfig config = { dim, count, ALG_METRIC_L2_SQR };
    AlgIndex* index = nullptr;
    if (alg_create_index(&config, &index) != ALG_OK) {
        return 1;
    }

    std::vector<float> data(count * dim, 0.5f);
    alg_set_vectors(index, count, data.data());

    std::vector<float> query(dim, 0.5f);
    std::vector<AlgSearchResult> results(10);
    uint32_t returned = 0;

    if (alg_search(index, query.data(), 10, ALG_SEARCH_FAST_FLOAT32, results.data(), &returned) == ALG_OK) {
        for (uint32_t i = 0; i < returned; ++i) {
            std::cout << "Top " << (i + 1) << ": ID " << results[i].id
                      << " (distancia: " << results[i].distance << ")\n";
        }
    }

    alg_free_index(index);
    return 0;
}
```

## Estructura del repositorio

```
ALG_2.0/
|-- include/
|   `-- alg_engine.h          Cabecera publica de la C ABI
|-- src/
|   |-- core/                 Gestion del indice, heap y logica de busqueda
|   |-- kernels/              Kernels SIMD (AVX2+FMA) y fallback escalar
|   `-- platform/             Deteccion de CPU y asignador alineado
|-- examples/                 Ejemplos funcionales en C y C++ con scripts de compilacion
|-- tests/                    Suite de 14 pruebas unitarias y de regresion numerica
|-- tools/                    Herramienta de benchmark en linea de comandos
|-- dist/                     Paquete de distribucion del SDK (binarios e interfaces)
`-- CMakeLists.txt            Configuracion de compilacion del proyecto
```

## Compilacion y pruebas

### Requisitos
- Compilador C++14 y C11: GCC 7+, Clang 6+ o MSVC 2017+
- CMake 3.16 o superior (opcional si se compila con scripts directos)

### Construccion con CMake (Linux y Windows)

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
ctest --output-on-failure
```

### Construccion directa en Windows (PowerShell)

```powershell
.\build.ps1
```

El script genera las bibliotecas en bin/, compila y ejecuta los 14 tests unitarios, compila los ejemplos y empaqueta el SDK en dist/alg-vector-engine-sdk.

## Contenido del SDK empaquetado

El directorio dist/alg-vector-engine-sdk/ contiene los artefactos listos para su distribucion e integracion directa:

- include/alg_engine.h: Cabecera C unica y autocontenida.
- lib/windows-x64/: Biblioteca estatica (libalg.a), biblioteca dinamica (alg.dll) y libreria de importacion (libalg_dll.a).
- lib/linux-x64/: Biblioteca estatica (libalg.a) y biblioteca compartida (libalg.so).
- examples/: Ejemplos minimos en C y C++ con archivos CMakeLists.txt, build_examples.bat y build_examples.sh.

## Licencia

Distribuido bajo la licencia Apache 2.0. Consulte el archivo LICENSE para mas detalles.
