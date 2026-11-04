// run_batched_comparison.cpp
// Comparación por lotes (Batch Query Benchmark): nq in {1, 8, 32, 128}
// Compara A2 frente a FAISS Flat L2 en throughput (QPS) y latencia media por batch

#include "../faiss_comparison/faiss_c_shim.h"
#include "../../include/alg_flat_types.h"
#include "../../include/simd_kernels.h"
#include "../../include/ieee754_bounds.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <windows.h>

using namespace alg_flat;

static inline uint64_t get_highres_time_ns() {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    _mm_lfence();
    QueryPerformanceCounter(&counter);
    _mm_lfence();
    return static_cast<uint64_t>((counter.QuadPart * 1000000000ULL) / freq.QuadPart);
}

static inline void run_variant_a2_exact_scan(
    const float* compact_vectors,
    uint32_t dim,
    const uint32_t* eligible_ids,
    uint32_t eligible_count,
    const float* query_values,
    uint32_t k,
    Neighbor* out_topk,
    uint32_t* out_count) {

    TopKHeap heap;
    heap.init(k);

    uint32_t i = 0;
    for (; i + 4 <= eligible_count; i += 4) {
        const float* v0 = compact_vectors + static_cast<size_t>(i + 0) * dim;
        const float* v1 = compact_vectors + static_cast<size_t>(i + 1) * dim;
        const float* v2 = compact_vectors + static_cast<size_t>(i + 2) * dim;
        const float* v3 = compact_vectors + static_cast<size_t>(i + 3) * dim;

        __m128 d4 = l2_sq_avx2_4x_aos(query_values, v0, v1, v2, v3, dim);
        float gate = (heap.size < k) ? FLT_MAX : (heap.threshold_sq * (1.0f + SIMD_TREE_GATE_REL_EPS));
        int mask = _mm_movemask_ps(_mm_cmple_ps(d4, _mm_set1_ps(gate)));
        if (mask != 0) {
            alignas(16) float d[4];
            _mm_store_ps(d, d4);
            if (mask & 1) heap.push(eligible_ids[i + 0], d[0]);
            if (mask & 2) heap.push(eligible_ids[i + 1], d[1]);
            if (mask & 4) heap.push(eligible_ids[i + 2], d[2]);
            if (mask & 8) heap.push(eligible_ids[i + 3], d[3]);
        }
    }
    for (; i < eligible_count; ++i) {
        const float* v = compact_vectors + static_cast<size_t>(i) * dim;
        float d_sq = l2_sq_avx2_1x_aos(query_values, v, dim);
        float gate = (heap.size < k) ? FLT_MAX : (heap.threshold_sq * (1.0f + SIMD_TREE_GATE_REL_EPS));
        if (d_sq <= gate) {
            heap.push(eligible_ids[i], d_sq);
        }
    }

    *out_count = heap.size;
    heap.sort_ascending();
    for (uint32_t j = 0; j < heap.size; ++j) out_topk[j] = heap.data[j];
}

int main() {
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(), 0x1);

    std::vector<uint32_t> batch_sizes = {1, 8, 32, 128};
    uint32_t dim = 64;
    uint32_t k = 10;
    const uint32_t N = 90000;

    std::vector<float> corpus(N * dim);
    std::ifstream fc("data/hito3_3/D64/corpus.f32", std::ios::binary);
    fc.read(reinterpret_cast<char*>(corpus.data()), corpus.size() * sizeof(float));
    fc.close();

    std::vector<float> queries(10000 * dim);
    std::ifstream fq("data/hito3_3/D64/queries.f32", std::ios::binary);
    fq.read(reinterpret_cast<char*>(queries.data()), queries.size() * sizeof(float));
    fq.close();

    std::vector<float> selectivities = {1.0f, 10.0f, 100.0f};

    std::cout << "================================================================================" << std::endl;
    std::cout << "BATCH QUERY BENCHMARK: A2 vs FAISS Flat (nq in {1, 8, 32, 128}, D=64, Top-10)" << std::endl;
    std::cout << "================================================================================" << std::endl;

    for (float sel : selectivities) {
        std::vector<uint32_t> eligible_ids;
        uint32_t target_cat = static_cast<uint32_t>(sel * 100.0f + 0.5f);
        if (sel >= 100.0f) target_cat = 10001U;
        for (uint32_t id = 0; id < N; ++id) {
            if ((id % 10000) < target_cat) eligible_ids.push_back(id);
        }
        uint32_t eligible_count = static_cast<uint32_t>(eligible_ids.size());

        std::vector<float> compact_vectors(static_cast<size_t>(eligible_count) * dim);
        for (uint32_t i = 0; i < eligible_count; ++i) {
            uint32_t orig_id = eligible_ids[i];
            memcpy(&compact_vectors[static_cast<size_t>(i) * dim],
                   &corpus[static_cast<size_t>(orig_id) * dim],
                   dim * sizeof(float));
        }

        FaissFlatHandle faiss_compact = faiss_shim_create_flat_l2(dim);
        faiss_shim_flat_add(faiss_compact, eligible_count, compact_vectors.data());

        std::cout << "\n--- SELECTIVIDAD " << sel << "% (" << eligible_count << " candidatos) ---" << std::endl;

        for (uint32_t nq : batch_sizes) {
            uint32_t num_batches = (nq == 1) ? 500 : (nq <= 8 ? 100 : 30);
            std::vector<double> lat_a2_batch, lat_faiss_batch;
            lat_a2_batch.reserve(num_batches);
            lat_faiss_batch.reserve(num_batches);

            std::vector<Neighbor> res_a2(nq * k);
            std::vector<float> dists_faiss(nq * k);
            std::vector<int64_t> labels_faiss(nq * k);

            for (uint32_t b = 0; b < num_batches; ++b) {
                const float* batch_queries = &queries[((b * nq) % (10000 - nq)) * dim];

                // Medición A2 (bucle de nq consultas)
                uint64_t t0 = get_highres_time_ns();
                for (uint32_t q = 0; q < nq; ++q) {
                    uint32_t cnt = 0;
                    run_variant_a2_exact_scan(compact_vectors.data(), dim, eligible_ids.data(), eligible_count,
                                              &batch_queries[q * dim], k, &res_a2[q * k], &cnt);
                }
                uint64_t t1 = get_highres_time_ns();
                lat_a2_batch.push_back((t1 - t0) / 1000.0);

                // Medición FAISS (llamada nativa con n_queries = nq)
                uint64_t tf0 = get_highres_time_ns();
                faiss_shim_search_flat(faiss_compact, nq, batch_queries, k, dists_faiss.data(), labels_faiss.data(), nullptr);
                uint64_t tf1 = get_highres_time_ns();
                lat_faiss_batch.push_back((tf1 - tf0) / 1000.0);
            }

            double mean_a2 = std::accumulate(lat_a2_batch.begin(), lat_a2_batch.end(), 0.0) / num_batches;
            double mean_faiss = std::accumulate(lat_faiss_batch.begin(), lat_faiss_batch.end(), 0.0) / num_batches;

            double lat_per_q_a2 = mean_a2 / nq;
            double lat_per_q_faiss = mean_faiss / nq;
            double speedup = mean_faiss / mean_a2;
            double qps_a2 = (1.0e6 / mean_a2) * nq;
            double qps_faiss = (1.0e6 / mean_faiss) * nq;

            std::cout << "[Batch nq=" << std::setw(3) << nq << "] "
                      << "A2: " << std::setw(7) << std::fixed << std::setprecision(2) << lat_per_q_a2 << " us/q (" << std::setw(7) << static_cast<uint64_t>(qps_a2) << " QPS) | "
                      << "FAISS: " << std::setw(7) << lat_per_q_faiss << " us/q (" << std::setw(7) << static_cast<uint64_t>(qps_faiss) << " QPS) | "
                      << "Speedup A2/FAISS: " << std::setw(5) << std::setprecision(2) << speedup << "x" << std::endl;
        }

        faiss_shim_free_flat(faiss_compact);
    }

    return 0;
}
