// run_pure_distance_comparison.cpp
// Micro-benchmark simétrico de Distancia Pura (ALG AVX2 4x vs FAISS fvec_L2sqr_ny)
// Ambos ejecutan sobre el MISMO puntero de memoria contigua sin selección Top-K

#include "../../include/simd_kernels.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <windows.h>

using namespace alg_flat;

extern "C" {
    void faiss_fvec_L2sqr_ny(float* dis, const float* x, const float* y, size_t d, size_t ny)
        asm("?fvec_L2sqr_ny@faiss@@YAXPEAMPEBM1_K2@Z");
}

static inline uint64_t get_highres_time_ns() {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    _mm_lfence();
    QueryPerformanceCounter(&counter);
    _mm_lfence();
    return static_cast<uint64_t>((counter.QuadPart * 1000000000ULL) / freq.QuadPart);
}

static inline void run_pure_distance_avx2(
    const float* compact_vectors,
    uint32_t dim,
    uint32_t count,
    const float* query_values,
    float* out_dists) {

    uint32_t i = 0;
    for (; i + 4 <= count; i += 4) {
        const float* v0 = compact_vectors + static_cast<size_t>(i + 0) * dim;
        const float* v1 = compact_vectors + static_cast<size_t>(i + 1) * dim;
        const float* v2 = compact_vectors + static_cast<size_t>(i + 2) * dim;
        const float* v3 = compact_vectors + static_cast<size_t>(i + 3) * dim;
        __m128 d4 = l2_sq_avx2_4x_aos(query_values, v0, v1, v2, v3, dim);
        _mm_storeu_ps(out_dists + i, d4);
    }
    for (; i < count; ++i) {
        const float* v = compact_vectors + static_cast<size_t>(i) * dim;
        out_dists[i] = l2_sq_avx2_1x_aos(query_values, v, dim);
    }
}

int main() {
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(), 0x1);

    std::vector<uint32_t> dims = {64, 128, 256};
    std::vector<uint32_t> counts = {900, 9000, 90000};
    const uint32_t num_queries = 500;

    std::cout << "================================================================================" << std::endl;
    std::cout << "MICRO-BENCHMARK DE DISTANCIA PURA (ALG AVX2 4x vs FAISS fvec_L2sqr_ny)" << std::endl;
    std::cout << "Mismo buffer fisico, sin heap, sin seleccion Top-K, 500 consultas por celda" << std::endl;
    std::cout << "================================================================================" << std::endl;

    for (uint32_t dim : dims) {
        std::string dim_str = "D" + std::to_string(dim);
        std::string corpus_file = "data/hito3_3/" + dim_str + "/corpus.f32";
        std::string queries_file = "data/hito3_3/" + dim_str + "/queries.f32";

        std::vector<float> corpus(90000 * dim);
        std::ifstream fc(corpus_file, std::ios::binary);
        if (!fc) continue;
        fc.read(reinterpret_cast<char*>(corpus.data()), corpus.size() * sizeof(float));
        fc.close();

        std::vector<float> queries(10000 * dim);
        std::ifstream fq(queries_file, std::ios::binary);
        if (!fq) continue;
        fq.read(reinterpret_cast<char*>(queries.data()), queries.size() * sizeof(float));
        fq.close();

        for (uint32_t count : counts) {
            std::vector<float> dists_alg(count);
            std::vector<float> dists_faiss(count);

            // Validar que ambos calculan distancias numericamente consistentes
            run_pure_distance_avx2(corpus.data(), dim, count, &queries[0], dists_alg.data());
            faiss_fvec_L2sqr_ny(dists_faiss.data(), &queries[0], corpus.data(), dim, count);

            double max_diff = 0.0;
            for (uint32_t i = 0; i < count; ++i) {
                double diff = std::abs(static_cast<double>(dists_alg[i]) - static_cast<double>(dists_faiss[i]));
                if (diff > max_diff) max_diff = diff;
            }

            // Warmup
            for (uint32_t q = 0; q < 50; ++q) {
                run_pure_distance_avx2(corpus.data(), dim, count, &queries[q * dim], dists_alg.data());
                faiss_fvec_L2sqr_ny(dists_faiss.data(), &queries[q * dim], corpus.data(), dim, count);
            }

            // Medicion con rotacion alternada para evitar sesgo termico
            std::vector<double> lat_alg, lat_faiss;
            lat_alg.reserve(num_queries);
            lat_faiss.reserve(num_queries);

            for (uint32_t q = 0; q < num_queries; ++q) {
                const float* q_vec = &queries[q * dim];
                if (q % 2 == 0) {
                    uint64_t t0 = get_highres_time_ns();
                    run_pure_distance_avx2(corpus.data(), dim, count, q_vec, dists_alg.data());
                    uint64_t t1 = get_highres_time_ns();
                    lat_alg.push_back((t1 - t0) / 1000.0);

                    uint64_t tf0 = get_highres_time_ns();
                    faiss_fvec_L2sqr_ny(dists_faiss.data(), q_vec, corpus.data(), dim, count);
                    uint64_t tf1 = get_highres_time_ns();
                    lat_faiss.push_back((tf1 - tf0) / 1000.0);
                } else {
                    uint64_t tf0 = get_highres_time_ns();
                    faiss_fvec_L2sqr_ny(dists_faiss.data(), q_vec, corpus.data(), dim, count);
                    uint64_t tf1 = get_highres_time_ns();
                    lat_faiss.push_back((tf1 - tf0) / 1000.0);

                    uint64_t t0 = get_highres_time_ns();
                    run_pure_distance_avx2(corpus.data(), dim, count, q_vec, dists_alg.data());
                    uint64_t t1 = get_highres_time_ns();
                    lat_alg.push_back((t1 - t0) / 1000.0);
                }
            }

            double mean_alg = std::accumulate(lat_alg.begin(), lat_alg.end(), 0.0) / num_queries;
            double mean_faiss = std::accumulate(lat_faiss.begin(), lat_faiss.end(), 0.0) / num_queries;
            double speedup = mean_faiss / mean_alg;

            std::cout << "[D=" << std::setw(3) << dim << " | N=" << std::setw(5) << count << "] "
                      << "ALG AVX2 4x: " << std::setw(8) << std::fixed << std::setprecision(2) << mean_alg << " us | "
                      << "FAISS native fvec_L2sqr_ny: " << std::setw(8) << mean_faiss << " us | "
                      << "Speedup Real: " << std::setw(5) << std::setprecision(2) << speedup << "x | "
                      << "MaxDiff: " << std::setprecision(6) << max_diff << std::endl;
        }
    }

    return 0;
}
