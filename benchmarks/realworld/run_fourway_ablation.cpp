// ==============================================================================
// run_fourway_ablation.cpp
// Hito 3.3.1: Arnés de Ablación de 4 Vías para Descomposición de Speedup
//
// Variantes Evaluadas (mismo conjunto elegible e idéntico oráculo binary64):
//   - A: ExactScan AVX2 4x unrolled sobre lista de IDs elegibles (baseline compacto)
//   - B: ALG Plan 5A sobre índice particionado (medición neta de poda geométrica)
//   - C1: FAISS Flat sobre índice compacto de candidatos con remapeo de etiquetas
//   - C2: FAISS Flat sobre corpus completo con IDSelectorBitmap
//
// Familias de Filtro:
//   - 1. Uniform_Synthetic (selectividad 1%, 10%, 100%)
//   - 2. Semantic_Correlated (MiniBatchKMeans en corpus, clusters cercanos por query)
//
// Protocolo:
//   - P-Core 0, HIGH priority, 1 hilo OpenMP, rotación balanceada A/B/C1/C2
//   - COLD (evicción 36MB LLC) y WARM
//   - Verificación estricta contra Ground Truth double (binary64)
//   - 1.000 consultas evaluadas independientemente por condición
// ==============================================================================

#include "../faiss_comparison/faiss_c_shim.h"
#include "../../include/alg_flat_types.h"
#include "../../include/dataset_loader.h"
#include "../../include/physical_layouts.h"
#include "../../include/execution_plans.h"
#include "../../include/simd_kernels.h"
#include "../../include/ieee754_bounds.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <map>
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

struct DoubleNeighbor {
    uint32_t id;
    double dist_sq;
};

struct DoubleTopKHeap {
    DoubleNeighbor data[MAX_TOPK_CAPACITY];
    uint32_t size;
    uint32_t k;
    double threshold_sq;

    inline void init(uint32_t target_k) {
        size = 0;
        k = (target_k <= MAX_TOPK_CAPACITY) ? target_k : MAX_TOPK_CAPACITY;
        threshold_sq = 1.0e300;
    }

    static inline bool less(const DoubleNeighbor& a, const DoubleNeighbor& b) {
        if (a.dist_sq != b.dist_sq) return a.dist_sq < b.dist_sq;
        return a.id < b.id;
    }

    inline void push(uint32_t id, double dist_sq) {
        if (size < k) {
            uint32_t i = size++;
            while (i > 0) {
                uint32_t p = (i - 1) >> 1;
                DoubleNeighbor cand{id, dist_sq};
                if (!less(data[p], cand)) break;
                data[i] = data[p];
                i = p;
            }
            data[i] = {id, dist_sq};
            if (size == k) threshold_sq = data[0].dist_sq;
            return;
        }
        if (dist_sq >= threshold_sq) return;

        uint32_t i = 0;
        DoubleNeighbor cand{id, dist_sq};
        while (true) {
            uint32_t left = (i << 1) + 1;
            uint32_t right = left + 1;
            if (left >= k) break;
            uint32_t best = left;
            if (right < k && less(data[left], data[right])) best = right;
            if (!less(cand, data[best])) break;
            data[i] = data[best];
            i = best;
        }
        data[i] = cand;
        threshold_sq = data[0].dist_sq;
    }

    inline void sort_ascending(DoubleNeighbor* out) {
        uint32_t n = size;
        for (uint32_t i = 0; i < n; ++i) {
            out[n - 1 - i] = data[0];
            DoubleNeighbor last = data[--size];
            if (size == 0) break;
            uint32_t cur = 0;
            while (true) {
                uint32_t left = (cur << 1) + 1;
                uint32_t right = left + 1;
                if (left >= size) break;
                uint32_t best = left;
                if (right < size && less(data[left], data[right])) best = right;
                if (!less(last, data[best])) break;
                data[cur] = data[best];
                cur = best;
            }
            data[cur] = last;
        }
    }
};

static void compute_ground_truth_double(
    const float* query_values,
    uint32_t dim,
    const float* corpus_vectors,
    const uint32_t* eligible_ids,
    uint32_t eligible_count,
    uint32_t k,
    DoubleNeighbor* out_topk,
    uint32_t* out_count) {

    DoubleTopKHeap heap;
    heap.init(k);

    for (uint32_t i = 0; i < eligible_count; ++i) {
        uint32_t id = eligible_ids[i];
        const float* v = corpus_vectors + static_cast<size_t>(id) * dim;
        double sum_sq = 0.0;
        for (uint32_t d = 0; d < dim; ++d) {
            double diff = static_cast<double>(query_values[d]) - static_cast<double>(v[d]);
            sum_sq += diff * diff;
        }
        heap.push(id, sum_sq);
    }

    *out_count = heap.size;
    heap.sort_ascending(out_topk);
}

// Variante A: ExactScan AVX2 con desenrollado 4x idéntico al kernel de Plan 5A
// pero sobre la lista contigua de candidatos (sin particionamiento ni poda)
static inline void run_variant_a_exact_scan(
    const float* corpus_vectors,
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
        uint32_t id0 = eligible_ids[i];
        uint32_t id1 = eligible_ids[i + 1];
        uint32_t id2 = eligible_ids[i + 2];
        uint32_t id3 = eligible_ids[i + 3];

        const float* v0 = corpus_vectors + static_cast<size_t>(id0) * dim;
        const float* v1 = corpus_vectors + static_cast<size_t>(id1) * dim;
        const float* v2 = corpus_vectors + static_cast<size_t>(id2) * dim;
        const float* v3 = corpus_vectors + static_cast<size_t>(id3) * dim;

        __m128 d4 = l2_sq_avx2_4x_aos(query_values, v0, v1, v2, v3, dim);
        float gate = (heap.size < k) ? FLT_MAX : (heap.threshold_sq * (1.0f + SIMD_TREE_GATE_REL_EPS));
        int mask = _mm_movemask_ps(_mm_cmple_ps(d4, _mm_set1_ps(gate)));
        if (mask != 0) {
            alignas(16) float d[4];
            _mm_store_ps(d, d4);
            if (mask & 1) heap.push(id0, d[0]);
            if (mask & 2) heap.push(id1, d[1]);
            if (mask & 4) heap.push(id2, d[2]);
            if (mask & 8) heap.push(id3, d[3]);
        }
    }
    for (; i < eligible_count; ++i) {
        uint32_t id = eligible_ids[i];
        const float* v = corpus_vectors + static_cast<size_t>(id) * dim;
        float d_sq = l2_sq_avx2_1x_aos(query_values, v, dim);
        float gate = (heap.size < k) ? FLT_MAX : (heap.threshold_sq * (1.0f + SIMD_TREE_GATE_REL_EPS));
        if (d_sq <= gate) {
            heap.push(id, d_sq);
        }
    }

    *out_count = heap.size;
    heap.sort_ascending();
    for (uint32_t j = 0; j < heap.size; ++j) {
        out_topk[j] = heap.data[j];
    }
}

static double compute_type7_quantile(const std::vector<double>& sorted_v, double p) {
    if (sorted_v.empty()) return 0.0;
    if (sorted_v.size() == 1) return sorted_v[0];
    if (p <= 0.0) return sorted_v.front();
    if (p >= 1.0) return sorted_v.back();
    double h = static_cast<double>(sorted_v.size() - 1) * p;
    size_t i = static_cast<size_t>(h);
    double f = h - static_cast<double>(i);
    return (1.0 - f) * sorted_v[i] + f * sorted_v[i + 1];
}

struct SemanticRecord {
    uint32_t query_row;
    float target_sel;
    uint32_t eligible_count;
    std::vector<uint32_t> eligible_clusters;
};

// Carga rápida del fichero JSONL de filtros semánticos
static bool load_semantic_filters_jsonl(
    const std::string& path,
    std::map<std::pair<uint32_t, int>, SemanticRecord>& out_records) {

    std::ifstream f(path);
    if (!f) return false;

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        // Parse simple de los campos
        size_t p_qr = line.find("\"query_row\":");
        size_t p_ts = line.find("\"target_selectivity\":");
        size_t p_ec = line.find("\"eligible_count\":");
        size_t p_cl = line.find("\"eligible_cluster_ids\":[");

        if (p_qr == std::string::npos || p_ts == std::string::npos ||
            p_ec == std::string::npos || p_cl == std::string::npos) continue;

        SemanticRecord rec;
        rec.query_row = std::stoul(line.substr(p_qr + 12));
        rec.target_sel = std::stof(line.substr(p_ts + 21));
        rec.eligible_count = std::stoul(line.substr(p_ec + 17));

        int sel_key = static_cast<int>(std::round(rec.target_sel * 100.0f));

        size_t cl_start = p_cl + 24;
        size_t cl_end = line.find(']', cl_start);
        if (cl_end != std::string::npos) {
            std::string sub = line.substr(cl_start, cl_end - cl_start);
            std::stringstream ss(sub);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) {
                    rec.eligible_clusters.push_back(std::stoul(item));
                }
            }
        }

        out_records[{rec.query_row, sel_key}] = rec;
    }
    return true;
}

int main(int argc, char** argv) {
    std::cout << "================================================================================" << std::endl;
    std::cout << "HITO 3.3.1: ARNES DE ABLACION DE 4 VIAS (BENCHMARK INTEGRITY)" << std::endl;
    std::cout << "  - Variante A:  ExactScan AVX2 4x sobre lista de IDs elegibles" << std::endl;
    std::cout << "  - Variante B:  ALG Plan 5A sobre índice particionado (poda esférica)" << std::endl;
    std::cout << "  - Variante C1: FAISS Flat sobre índice compacto de candidatos (con remapeo)" << std::endl;
    std::cout << "  - Variante C2: FAISS Flat sobre corpus completo + IDSelectorBitmap" << std::endl;
    std::cout << "  - Protocolo:   P-Core 0, HIGH priority, 1 hilo OpenMP, balanceo de orden" << std::endl;
    std::cout << "================================================================================" << std::endl;

    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(), 0x1);

    const std::string corpus_file = "data/hito3_3/D64/corpus.f32";
    const std::string queries_file = "data/hito3_3/D64/queries.f32";
    const std::string cluster_file = "data/hito3_3/D64/semantic_filters/corpus_cluster_ids.u32";
    const std::string jsonl_file = "data/hito3_3/D64/semantic_filters/query_cluster_filters.jsonl";

    const uint32_t N = 90000;
    const uint32_t dim = 64;
    const uint32_t num_queries_eval = 1000; // 1.000 consultas por condición según protocolo

    // Cargar vectores de corpus
    std::vector<float> corpus(static_cast<size_t>(N) * dim);
    std::ifstream f_corp(corpus_file, std::ios::binary);
    if (!f_corp) {
        std::cerr << "Error abriendo " << corpus_file << std::endl;
        return 1;
    }
    f_corp.read(reinterpret_cast<char*>(corpus.data()), corpus.size() * sizeof(float));
    f_corp.close();

    // Cargar clusters semánticos
    std::vector<uint32_t> corpus_clusters(N);
    std::ifstream f_ccl(cluster_file, std::ios::binary);
    if (!f_ccl) {
        std::cerr << "Error abriendo " << cluster_file << std::endl;
        return 1;
    }
    f_ccl.read(reinterpret_cast<char*>(corpus_clusters.data()), N * sizeof(uint32_t));
    f_ccl.close();

    // Cargar consultas
    std::vector<float> queries(10000 * dim);
    std::ifstream f_q(queries_file, std::ios::binary);
    if (!f_q) {
        std::cerr << "Error abriendo " << queries_file << std::endl;
        return 1;
    }
    f_q.read(reinterpret_cast<char*>(queries.data()), queries.size() * sizeof(float));
    f_q.close();

    std::cout << ">>> Cargando registros de query_cluster_filters.jsonl..." << std::flush;
    std::map<std::pair<uint32_t, int>, SemanticRecord> semantic_records;
    if (!load_semantic_filters_jsonl(jsonl_file, semantic_records)) {
        std::cerr << " Error abriendo " << jsonl_file << std::endl;
        return 1;
    }
    std::cout << " Listo (" << semantic_records.size() << " registros)." << std::endl;

    // Preparar dataset para ALG
    BenchmarkDataset ds;
    ds.header.num_vectors = N;
    ds.header.dimension = dim;
    ds.header.num_queries = num_queries_eval * 2;
    ds.vectors = static_cast<float*>(_aligned_malloc(static_cast<size_t>(N) * dim * sizeof(float), 64));
    ds.categories = static_cast<uint32_t*>(_aligned_malloc(static_cast<size_t>(N) * sizeof(uint32_t), 64));
    memcpy(ds.vectors, corpus.data(), static_cast<size_t>(N) * dim * sizeof(float));
    for (uint32_t i = 0; i < N; ++i) ds.categories[i] = i % 10000;

    ds.queries.resize(num_queries_eval * 2);
    for (uint32_t q = 0; q < num_queries_eval * 2; ++q) {
        memcpy(ds.queries[q].values, &queries[q * dim], dim * sizeof(float));
    }

    std::cout << ">>> Construyendo indice ALG Plan 5A (IVFPartitionedIndex, P=128)..." << std::flush;
    IVFPartitionedIndex idx_alg;
    auto t_a0 = get_highres_time_ns();
    if (!build_ivf_partitioned_index(ds, 128U, 15U, idx_alg)) {
        std::cerr << " Error en build_ivf_partitioned_index." << std::endl;
        return 1;
    }
    auto t_a1 = get_highres_time_ns();
    std::cout << " Listo (" << (t_a1 - t_a0) / 1e6 << " ms)." << std::endl;

    std::cout << ">>> Construyendo FAISS Flat de corpus completo (Variante C2)..." << std::flush;
    FaissFlatHandle faiss_full = faiss_shim_create_flat_l2(dim);
    faiss_shim_flat_add(faiss_full, N, ds.vectors);
    std::cout << " Listo." << std::endl;

    ColdCacheEvictor evictor(36);

    CreateDirectoryA("results/hito3_3_1", NULL);
    std::ofstream out_summary("results/hito3_3_1/fourway_ablation_summary.csv");
    out_summary << "FilterFamily,TargetSelectivity,AvgEligibleCount,ActualSelectivityPct,TopK,CacheMode,"
                << "Mean_A_us,Median_A_us,P95_A_us,P99_A_us,QPS_A,"
                << "Mean_B_us,Median_B_us,P95_B_us,P99_B_us,QPS_B,"
                << "Mean_C1_us,Median_C1_us,P95_C1_us,P99_C1_us,QPS_C1,"
                << "Mean_C2_us,Median_C2_us,P95_C2_us,P99_C2_us,QPS_C2,"
                << "Speedup_B_vs_A,Speedup_B_vs_C1,Speedup_B_vs_C2,Speedup_A_vs_C1,Speedup_C1_vs_C2,"
                << "AvgDists_Plan5A,PartitionsVisited_Plan5A,PartitionsPruned_Plan5A,Pruning_Ratio_Pct,"
                << "Recall_A,RankMatch_A,Recall_B,RankMatch_B,Recall_C1,RankMatch_C1,Recall_C2,RankMatch_C2\n";

    std::vector<std::string> families = {"Uniform_Synthetic", "Semantic_Correlated"};
    std::vector<float> selectivities = {1.0f, 10.0f, 100.0f};
    std::vector<uint32_t> top_ks = {10U, 100U};
    std::vector<std::string> cache_modes = {"WARM_CACHE", "COLD_LLC_FLUSH"};

    DoubleNeighbor gt[512];
    Neighbor res_a[512];
    Neighbor res_b[512];
    float dists_c1[512], dists_c2[512];
    int64_t labels_c1[512], labels_c2[512];
    uint32_t cnt_a = 0, cnt_b = 0;
    QueryExecutionMetrics met_b;

    size_t bmp_bytes = (N + 7) / 8;
    std::vector<uint8_t> bitmap(bmp_bytes, 0);

    // Buffers reutilizables
    std::vector<float> compact_vectors_buf(static_cast<size_t>(N) * dim);
    std::vector<uint32_t> eligible_ids_buf;
    eligible_ids_buf.reserve(N);

    // Pre-mapear vectores por cluster para lookup O(1)
    std::vector<std::vector<uint32_t>> cluster_to_corpus_ids(128);
    for (uint32_t id = 0; id < N; ++id) {
        cluster_to_corpus_ids[corpus_clusters[id]].push_back(id);
    }

    for (const auto& family : families) {
        std::cout << "\n================================================================================" << std::endl;
        std::cout << ">>> Evaluando Familia de Filtro: " << family << std::endl;
        std::cout << "================================================================================" << std::endl;

        for (float sel : selectivities) {
            int sel_key = static_cast<int>(std::round(sel));

            // Si es uniforme, la población elegible es idéntica para todas las consultas
            std::vector<uint32_t> static_uniform_eligible_ids;
            FaissFlatHandle static_faiss_compact = nullptr;
            FaissSelectorHandle static_sel_handle = nullptr;
            FaissParamsHandle static_params_c2 = nullptr;

            if (family == "Uniform_Synthetic") {
                uint32_t target_cat = static_cast<uint32_t>(sel * 100.0f + 0.5f);
                if (sel >= 100.0f) target_cat = 10001U;
                memset(bitmap.data(), 0, bmp_bytes);
                for (uint32_t id = 0; id < N; ++id) {
                    if (ds.categories[id] < target_cat) {
                        static_uniform_eligible_ids.push_back(id);
                        bitmap[id >> 3] |= (1U << (id & 7));
                    }
                }
                uint32_t el_cnt = static_cast<uint32_t>(static_uniform_eligible_ids.size());

                // Configurar ALG Plan 5A una vez para el batch uniforme
                idx_alg.eligible_ivf_ids_per_sel.resize(1);
                idx_alg.eligible_part_offsets_per_sel.resize(1);
                auto& el_ids = idx_alg.eligible_ivf_ids_per_sel[0];
                auto& part_offs = idx_alg.eligible_part_offsets_per_sel[0];
                el_ids.clear();
                el_ids.reserve(el_cnt);
                part_offs.resize(idx_alg.num_partitions + 1);

                std::vector<uint8_t> is_el(N, 0);
                for (uint32_t id : static_uniform_eligible_ids) is_el[id] = 1;

                for (uint32_t c = 0; c < idx_alg.num_partitions; ++c) {
                    part_offs[c] = static_cast<uint32_t>(el_ids.size());
                    uint32_t p_start = idx_alg.partitions[c].aos_offset;
                    uint32_t p_end = p_start + idx_alg.partitions[c].count;
                    for (uint32_t idx_v = p_start; idx_v < p_end; ++idx_v) {
                        uint32_t orig_id = idx_alg.part_aos_ids[idx_v];
                        if (is_el[orig_id]) {
                            el_ids.push_back(idx_v);
                        }
                    }
                }
                part_offs[idx_alg.num_partitions] = static_cast<uint32_t>(el_ids.size());

                // Construir C1 FAISS Flat compacto una vez fuera del cronómetro
                for (uint32_t i = 0; i < el_cnt; ++i) {
                    uint32_t id = static_uniform_eligible_ids[i];
                    memcpy(&compact_vectors_buf[static_cast<size_t>(i) * dim], &corpus[static_cast<size_t>(id) * dim], dim * sizeof(float));
                }
                static_faiss_compact = faiss_shim_create_flat_l2(dim);
                faiss_shim_flat_add(static_faiss_compact, el_cnt, compact_vectors_buf.data());

                // Selector C2 de FAISS
                static_sel_handle = faiss_shim_create_bitmap_selector(N, bitmap.data());
                static_params_c2 = faiss_shim_create_flat_params(static_sel_handle);
            }

            for (uint32_t k : top_ks) {
                for (const auto& cache_mode : cache_modes) {
                    bool is_cold = (cache_mode == "COLD_LLC_FLUSH");
                    uint32_t q_offset = is_cold ? num_queries_eval : 0;

                    std::vector<double> lat_a, lat_b, lat_c1, lat_c2;
                    lat_a.reserve(num_queries_eval);
                    lat_b.reserve(num_queries_eval);
                    lat_c1.reserve(num_queries_eval);
                    lat_c2.reserve(num_queries_eval);

                    uint32_t match_rec_a = 0, match_rec_b = 0, match_rec_c1 = 0, match_rec_c2 = 0;
                    uint32_t match_rnk_a = 0, match_rnk_b = 0, match_rnk_c1 = 0, match_rnk_c2 = 0;
                    uint64_t total_dists_b = 0;
                    uint64_t total_parts_visited_b = 0;
                    uint64_t total_eligible_all_q = 0;

                    for (uint32_t q = 0; q < num_queries_eval; ++q) {
                        uint32_t q_idx = q_offset + q;
                        const float* q_vec = &queries[q_idx * dim];

                        const uint32_t* cur_eligible_ids = nullptr;
                        uint32_t cur_eligible_count = 0;

                        FaissFlatHandle cur_faiss_c1 = nullptr;
                        FaissParamsHandle cur_params_c2 = nullptr;
                        FaissSelectorHandle cur_sel_handle = nullptr;

                        // Variables para semantic
                        std::vector<uint32_t> dyn_eligible_ids;
                        std::vector<uint8_t> dyn_bitmap;

                        if (family == "Uniform_Synthetic") {
                            cur_eligible_ids = static_uniform_eligible_ids.data();
                            cur_eligible_count = static_cast<uint32_t>(static_uniform_eligible_ids.size());
                            cur_faiss_c1 = static_faiss_compact;
                            cur_params_c2 = static_params_c2;
                        } else {
                            // Cargar filtro semántico específico de esta consulta
                            const auto& rec = semantic_records[{q_idx, sel_key}];
                            bool active_clusters[128] = {false};
                            for (uint32_t cid : rec.eligible_clusters) {
                                if (cid < 128) active_clusters[cid] = true;
                            }

                            dyn_bitmap.assign(bmp_bytes, 0);
                            dyn_eligible_ids.clear();
                            dyn_eligible_ids.reserve(rec.eligible_count + 128);

                            for (uint32_t id = 0; id < N; ++id) {
                                if (active_clusters[corpus_clusters[id]]) {
                                    dyn_eligible_ids.push_back(id);
                                    dyn_bitmap[id >> 3] |= (1U << (id & 7));
                                }
                            }
                            cur_eligible_ids = dyn_eligible_ids.data();
                            cur_eligible_count = static_cast<uint32_t>(dyn_eligible_ids.size());

                            // Configurar ALG Plan 5A fuera del cronómetro
                            idx_alg.eligible_ivf_ids_per_sel.resize(1);
                            idx_alg.eligible_part_offsets_per_sel.resize(1);
                            auto& el_ids = idx_alg.eligible_ivf_ids_per_sel[0];
                            auto& part_offs = idx_alg.eligible_part_offsets_per_sel[0];
                            el_ids.clear();
                            el_ids.reserve(cur_eligible_count);
                            part_offs.resize(idx_alg.num_partitions + 1);

                            std::vector<uint8_t> is_el(N, 0);
                            for (uint32_t id : dyn_eligible_ids) is_el[id] = 1;

                            for (uint32_t c = 0; c < idx_alg.num_partitions; ++c) {
                                part_offs[c] = static_cast<uint32_t>(el_ids.size());
                                uint32_t p_start = idx_alg.partitions[c].aos_offset;
                                uint32_t p_end = p_start + idx_alg.partitions[c].count;
                                for (uint32_t idx_v = p_start; idx_v < p_end; ++idx_v) {
                                    uint32_t orig_id = idx_alg.part_aos_ids[idx_v];
                                    if (is_el[orig_id]) {
                                        el_ids.push_back(idx_v);
                                    }
                                }
                            }
                            part_offs[idx_alg.num_partitions] = static_cast<uint32_t>(el_ids.size());

                            // Construir C1 compacto fuera del cronómetro
                            for (uint32_t i = 0; i < cur_eligible_count; ++i) {
                                uint32_t id = dyn_eligible_ids[i];
                                memcpy(&compact_vectors_buf[static_cast<size_t>(i) * dim], &corpus[static_cast<size_t>(id) * dim], dim * sizeof(float));
                            }
                            cur_faiss_c1 = faiss_shim_create_flat_l2(dim);
                            faiss_shim_flat_add(cur_faiss_c1, cur_eligible_count, compact_vectors_buf.data());

                            // Construir selector C2 fuera del cronómetro
                            cur_sel_handle = faiss_shim_create_bitmap_selector(N, dyn_bitmap.data());
                            cur_params_c2 = faiss_shim_create_flat_params(cur_sel_handle);
                        }

                        total_eligible_all_q += cur_eligible_count;

                        // 1. Oráculo Ground Truth binary64 sobre EXACTAMENTE los mismos IDs
                        uint32_t gt_cnt = 0;
                        compute_ground_truth_double(q_vec, dim, corpus.data(),
                                                    cur_eligible_ids, cur_eligible_count,
                                                    k, gt, &gt_cnt);

                        QueryPredicate pred;
                        pred.target_category = 0;
                        pred.eligible_ids = cur_eligible_ids;
                        pred.eligible_count = cur_eligible_count;
                        pred.sel_index = 0;

                        // Esquema de rotación balanceada para evitar sesgo térmico y de caché:
                        // Round q%4 == 0: A -> B -> C1 -> C2
                        // Round q%4 == 1: B -> C1 -> C2 -> A
                        // Round q%4 == 2: C1 -> C2 -> A -> B
                        // Round q%4 == 3: C2 -> A -> B -> C1

                        auto run_A = [&]() {
                            if (is_cold) evictor.flush_llc();
                            uint64_t t0 = get_highres_time_ns();
                            run_variant_a_exact_scan(corpus.data(), dim, cur_eligible_ids, cur_eligible_count, q_vec, k, res_a, &cnt_a);
                            uint64_t t1 = get_highres_time_ns();
                            lat_a.push_back((t1 - t0) / 1000.0);
                        };

                        auto run_B = [&]() {
                            if (is_cold) evictor.flush_llc();
                            uint64_t t0 = get_highres_time_ns();
                            execute_plan5a_ivf_part_prune_avx2(idx_alg, q_vec, pred, k, res_b, &cnt_b, met_b);
                            uint64_t t1 = get_highres_time_ns();
                            lat_b.push_back((t1 - t0) / 1000.0);
                            total_dists_b += met_b.dists_evaluated;
                            total_parts_visited_b += met_b.partitions_visited;
                        };

                        auto run_C1 = [&]() {
                            if (is_cold) evictor.flush_llc();
                            uint64_t t0 = get_highres_time_ns();
                            faiss_shim_search_flat(cur_faiss_c1, 1, q_vec, k, dists_c1, labels_c1, nullptr);
                            uint64_t t1 = get_highres_time_ns();
                            lat_c1.push_back((t1 - t0) / 1000.0);
                        };

                        auto run_C2 = [&]() {
                            if (is_cold) evictor.flush_llc();
                            uint64_t t0 = get_highres_time_ns();
                            faiss_shim_search_flat(faiss_full, 1, q_vec, k, dists_c2, labels_c2, cur_params_c2);
                            uint64_t t1 = get_highres_time_ns();
                            lat_c2.push_back((t1 - t0) / 1000.0);
                        };

                        uint32_t rot = q % 4;
                        if (rot == 0) { run_A(); run_B(); run_C1(); run_C2(); }
                        else if (rot == 1) { run_B(); run_C1(); run_C2(); run_A(); }
                        else if (rot == 2) { run_C1(); run_C2(); run_A(); run_B(); }
                        else { run_C2(); run_A(); run_B(); run_C1(); }

                        // Validar recall y rank match estricto de cada variante
                        std::vector<uint32_t> ids_a(cnt_a), ids_b(cnt_b);
                        for (uint32_t i = 0; i < cnt_a; ++i) ids_a[i] = res_a[i].id;
                        for (uint32_t i = 0; i < cnt_b; ++i) ids_b[i] = res_b[i].id;

                        std::vector<uint32_t> ids_c1(k), ids_c2(k);
                        for (uint32_t i = 0; i < k; ++i) {
                            if (labels_c1[i] >= 0 && labels_c1[i] < static_cast<int64_t>(cur_eligible_count)) {
                                ids_c1[i] = cur_eligible_ids[labels_c1[i]]; // Remapeo estricto
                            }
                            ids_c2[i] = static_cast<uint32_t>(labels_c2[i]);
                        }

                        auto eval_accuracy = [&](const std::vector<uint32_t>& res_ids, uint32_t cnt, uint32_t& rec_ok, uint32_t& rnk_ok) {
                            uint32_t m_set = 0;
                            for (uint32_t i = 0; i < cnt; ++i) {
                                for (uint32_t g = 0; g < gt_cnt; ++g) {
                                    if (res_ids[i] == gt[g].id) { m_set++; break; }
                                }
                            }
                            if (m_set == gt_cnt) rec_ok++;

                            bool rnk_match = (cnt == gt_cnt);
                            if (rnk_match) {
                                for (uint32_t i = 0; i < gt_cnt; ++i) {
                                    if (res_ids[i] != gt[i].id) { rnk_match = false; break; }
                                }
                            }
                            if (rnk_match) rnk_ok++;
                        };

                        eval_accuracy(ids_a, cnt_a, match_rec_a, match_rnk_a);
                        eval_accuracy(ids_b, cnt_b, match_rec_b, match_rnk_b);
                        eval_accuracy(ids_c1, k, match_rec_c1, match_rnk_c1);
                        eval_accuracy(ids_c2, k, match_rec_c2, match_rnk_c2);

                        // Liberar objetos dinámicos de semantic
                        if (family != "Uniform_Synthetic") {
                            faiss_shim_free_params(cur_params_c2);
                            faiss_shim_free_selector(cur_sel_handle);
                            faiss_shim_free_flat(cur_faiss_c1);
                        }
                    }

                    // Métricas estadísticas ordenadas
                    std::vector<double> s_a = lat_a, s_b = lat_b, s_c1 = lat_c1, s_c2 = lat_c2;
                    std::sort(s_a.begin(), s_a.end());
                    std::sort(s_b.begin(), s_b.end());
                    std::sort(s_c1.begin(), s_c1.end());
                    std::sort(s_c2.begin(), s_c2.end());

                    double mean_a = std::accumulate(s_a.begin(), s_a.end(), 0.0) / num_queries_eval;
                    double mean_b = std::accumulate(s_b.begin(), s_b.end(), 0.0) / num_queries_eval;
                    double mean_c1 = std::accumulate(s_c1.begin(), s_c1.end(), 0.0) / num_queries_eval;
                    double mean_c2 = std::accumulate(s_c2.begin(), s_c2.end(), 0.0) / num_queries_eval;

                    double med_a = compute_type7_quantile(s_a, 0.50);
                    double med_b = compute_type7_quantile(s_b, 0.50);
                    double med_c1 = compute_type7_quantile(s_c1, 0.50);
                    double med_c2 = compute_type7_quantile(s_c2, 0.50);

                    double p95_a = compute_type7_quantile(s_a, 0.95);
                    double p95_b = compute_type7_quantile(s_b, 0.95);
                    double p95_c1 = compute_type7_quantile(s_c1, 0.95);
                    double p95_c2 = compute_type7_quantile(s_c2, 0.95);

                    double p99_a = compute_type7_quantile(s_a, 0.99);
                    double p99_b = compute_type7_quantile(s_b, 0.99);
                    double p99_c1 = compute_type7_quantile(s_c1, 0.99);
                    double p99_c2 = compute_type7_quantile(s_c2, 0.99);

                    double qps_a = 1.0e6 / mean_a;
                    double qps_b = 1.0e6 / mean_b;
                    double qps_c1 = 1.0e6 / mean_c1;
                    double qps_c2 = 1.0e6 / mean_c2;

                    double speedup_b_vs_a = mean_a / mean_b;
                    double speedup_b_vs_c1 = mean_c1 / mean_b;
                    double speedup_b_vs_c2 = mean_c2 / mean_b;
                    double speedup_a_vs_c1 = mean_c1 / mean_a;
                    double speedup_c1_vs_c2 = mean_c2 / mean_c1;

                    double avg_eligible = static_cast<double>(total_eligible_all_q) / num_queries_eval;
                    double actual_sel_pct = (avg_eligible / N) * 100.0;

                    double avg_dists_b = static_cast<double>(total_dists_b) / num_queries_eval;
                    double avg_parts_visited_b = static_cast<double>(total_parts_visited_b) / num_queries_eval;
                    double avg_parts_pruned_b = static_cast<double>(idx_alg.num_partitions) - avg_parts_visited_b;
                    double pruning_b = 100.0 * (1.0 - (avg_dists_b / avg_eligible));
                    if (pruning_b < 0.0) pruning_b = 0.0;

                    double rec_a = 100.0 * match_rec_a / num_queries_eval;
                    double rnk_a = 100.0 * match_rnk_a / num_queries_eval;
                    double rec_b = 100.0 * match_rec_b / num_queries_eval;
                    double rnk_b = 100.0 * match_rnk_b / num_queries_eval;
                    double rec_c1 = 100.0 * match_rec_c1 / num_queries_eval;
                    double rnk_c1 = 100.0 * match_rnk_c1 / num_queries_eval;
                    double rec_c2 = 100.0 * match_rec_c2 / num_queries_eval;
                    double rnk_c2 = 100.0 * match_rnk_c2 / num_queries_eval;

                    out_summary << std::fixed << std::setprecision(4)
                                << family << "," << sel << "," << avg_eligible << "," << actual_sel_pct << ","
                                << k << "," << cache_mode << ","
                                << mean_a << "," << med_a << "," << p95_a << "," << p99_a << "," << qps_a << ","
                                << mean_b << "," << med_b << "," << p95_b << "," << p99_b << "," << qps_b << ","
                                << mean_c1 << "," << med_c1 << "," << p95_c1 << "," << p99_c1 << "," << qps_c1 << ","
                                << mean_c2 << "," << med_c2 << "," << p95_c2 << "," << p99_c2 << "," << qps_c2 << ","
                                << speedup_b_vs_a << "," << speedup_b_vs_c1 << "," << speedup_b_vs_c2 << ","
                                << speedup_a_vs_c1 << "," << speedup_c1_vs_c2 << ","
                                << avg_dists_b << "," << avg_parts_visited_b << "," << avg_parts_pruned_b << "," << pruning_b << ","
                                << rec_a << "," << rnk_a << ","
                                << rec_b << "," << rnk_b << ","
                                << rec_c1 << "," << rnk_c1 << ","
                                << rec_c2 << "," << rnk_c2 << "\n";
                    out_summary.flush();

                    std::cout << "[" << family.substr(0, 7) << " | Sel=" << std::setw(5) << sel << "% | "
                              << std::setw(15) << cache_mode << " | Top-" << std::setw(3) << k << "] "
                              << "A=" << std::setw(7) << mean_a << " us | B=" << std::setw(7) << mean_b << " us | "
                              << "C1=" << std::setw(7) << mean_c1 << " us | C2=" << std::setw(7) << mean_c2 << " us | "
                              << "B/A=" << std::setw(5) << speedup_b_vs_a << "x | "
                              << "B/C1=" << std::setw(5) << speedup_b_vs_c1 << "x | "
                              << "B/C2=" << std::setw(5) << speedup_b_vs_c2 << "x | "
                              << "Poda=" << std::setw(5) << pruning_b << "% | "
                              << "RecB=" << rec_b << "%" << std::endl;
                }
            }

            if (family == "Uniform_Synthetic") {
                faiss_shim_free_params(static_params_c2);
                faiss_shim_free_selector(static_sel_handle);
                faiss_shim_free_flat(static_faiss_compact);
            }
        }
    }

    faiss_shim_free_flat(faiss_full);
    out_summary.close();

    std::cout << "\n================================================================================" << std::endl;
    std::cout << "ARNES DE ABLACION DE 4 VIAS COMPLETADO EXITOSAMENTE" << std::endl;
    std::cout << "Resumen: results/hito3_3_1/fourway_ablation_summary.csv" << std::endl;
    std::cout << "================================================================================" << std::endl;

    return 0;
}
