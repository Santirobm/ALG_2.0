// ==============================================================================
// run_sdk_vs_faiss_native.cpp
// Native C++ Benchmark: ALG SDK Public DLL (include/alg_engine.h) vs FAISS FlatL2
//
// Evaluates 3 strictly separated scenarios on identical vectors, queries,
// candidate sets, metric (L2 squared), and tie-breaking oracle (binary64):
//   1. Full Corpus (100%):    alg_search         vs FAISS IndexFlatL2
//   2. Candidates by ID:      alg_search_compact vs FAISS IndexFlatL2 + IDSelectorBitmap
//   3. Compact Buffer:        alg_search_buffer  vs FAISS IndexFlatL2 (on compact buffer)
//
// Measures:
//   - Index build time & process memory delta (PrivateUsage / WorkingSetSize)
//   - Search latency (mean, p50, p95, QPS) in WARM_CACHE and COLD_LLC_FLUSH
//   - IDSetRecall (%) and StrictRankMatch (%) vs binary64 oracle for both engines
// ==============================================================================

#define ALG_USE_DLL
#include "../../include/alg_engine.h"
#include "../faiss_comparison/faiss_c_shim.h"

#include <windows.h>
#include <psapi.h>
#include <immintrin.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

static inline uint64_t now_ns() {
    static LARGE_INTEGER freq = {};
    if (freq.QuadPart == 0) {
        QueryPerformanceFrequency(&freq);
    }
    LARGE_INTEGER counter;
    _mm_lfence();
    QueryPerformanceCounter(&counter);
    _mm_lfence();
    return static_cast<uint64_t>((static_cast<double>(counter.QuadPart) * 1e9) /
                                 static_cast<double>(freq.QuadPart));
}

struct LLCEvictor {
    std::vector<uint8_t> buf;
    explicit LLCEvictor(size_t bytes = 36ULL * 1024ULL * 1024ULL) : buf(bytes, 1) {}
    void flush() {
        volatile uint8_t acc = 0;
        const size_t n = buf.size();
        for (size_t i = 0; i < n; i += 64) {
            buf[i] = static_cast<uint8_t>(buf[i] + 1);
            acc ^= buf[i];
        }
        (void)acc;
        _mm_mfence();
    }
};

struct MemSnapshot {
    size_t working_set_bytes = 0;
    size_t private_bytes = 0;
};

static MemSnapshot get_process_memory() {
    PROCESS_MEMORY_COUNTERS_EX pmc;
    std::memset(&pmc, 0, sizeof(pmc));
    pmc.cb = sizeof(pmc);
    if (GetProcessMemoryInfo(GetCurrentProcess(),
                             reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc),
                             sizeof(pmc))) {
        return { static_cast<size_t>(pmc.WorkingSetSize),
                 static_cast<size_t>(pmc.PrivateUsage) };
    }
    return {};
}

static bool load_raw_f32(const std::string& path, std::vector<float>& out, size_t expected_floats = 0) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        std::cerr << "[ERROR] Cannot open file: " << path << "\n";
        return false;
    }
    std::streamsize sz = f.tellg();
    f.seekg(0, std::ios::beg);
    if (sz <= 0 || (sz % sizeof(float)) != 0) {
        std::cerr << "[ERROR] Invalid file size for f32: " << path << "\n";
        return false;
    }
    size_t count = static_cast<size_t>(sz) / sizeof(float);
    if (expected_floats > 0 && count < expected_floats) {
        std::cerr << "[ERROR] File smaller than expected: " << path << "\n";
        return false;
    }
    out.resize(count);
    if (!f.read(reinterpret_cast<char*>(out.data()), sz)) {
        std::cerr << "[ERROR] Failed reading file: " << path << "\n";
        return false;
    }
    return true;
}

struct OracleItem {
    uint32_t id;
    double dist_sq;
};

static std::vector<OracleItem> compute_oracle_topk(
    const float* corpus,
    uint32_t dim,
    const uint32_t* eligible_ids, // nullptr => 0..N-1
    uint32_t count,
    const float* query,
    uint32_t k) {
    std::vector<OracleItem> all(count);
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t id = eligible_ids ? eligible_ids[i] : i;
        const float* v = corpus + static_cast<size_t>(id) * dim;
        double sum = 0.0;
        for (uint32_t d = 0; d < dim; ++d) {
            double diff = static_cast<double>(query[d]) - static_cast<double>(v[d]);
            sum += diff * diff;
        }
        all[i] = { id, sum };
    }
    uint32_t actual_k = std::min(k, count);
    std::partial_sort(all.begin(), all.begin() + actual_k, all.end(),
        [](const OracleItem& a, const OracleItem& b) {
            if (a.dist_sq != b.dist_sq) return a.dist_sq < b.dist_sq;
            return a.id < b.id;
        });
    all.resize(actual_k);
    return all;
}

struct FloatHeapItem {
    uint32_t id;
    float dist;
};

static inline bool float_item_less(const FloatHeapItem& a, const FloatHeapItem& b) {
    if (a.dist != b.dist) return a.dist < b.dist;
    return a.id < b.id;
}

static void select_topk_from_distances(
    const float* dists,
    const uint32_t* ids,
    uint32_t count,
    uint32_t k,
    std::vector<FloatHeapItem>& heap_buf,
    uint32_t* out_ids) {
    uint32_t size = 0;
    float threshold = FLT_MAX;
    for (uint32_t idx = 0; idx < count; ++idx) {
        float d = dists[idx];
        uint32_t id = ids[idx];
        if (size < k) {
            uint32_t i = size++;
            FloatHeapItem cand{ id, d };
            while (i > 0) {
                uint32_t p = (i - 1) >> 1;
                if (!float_item_less(heap_buf[p], cand)) break;
                heap_buf[i] = heap_buf[p];
                i = p;
            }
            heap_buf[i] = cand;
            if (size == k) threshold = heap_buf[0].dist;
        } else if (d <= threshold) {
            FloatHeapItem cand{ id, d };
            if (!float_item_less(cand, heap_buf[0])) continue;
            uint32_t i = 0;
            while (true) {
                uint32_t left = (i << 1) + 1;
                uint32_t right = left + 1;
                if (left >= k) break;
                uint32_t best = left;
                if (right < k && float_item_less(heap_buf[left], heap_buf[right])) best = right;
                if (!float_item_less(cand, heap_buf[best])) break;
                heap_buf[i] = heap_buf[best];
                i = best;
            }
            heap_buf[i] = cand;
            threshold = heap_buf[0].dist;
        }
    }
    std::sort(heap_buf.begin(), heap_buf.begin() + size, float_item_less);
    for (uint32_t i = 0; i < size; ++i) out_ids[i] = heap_buf[i].id;
}

static bool ref_is_finite_buffer(const float* data, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        uint32_t u;
        std::memcpy(&u, &data[i], sizeof(float));
        if ((u & 0x7F800000u) == 0x7F800000u) {
            return false;
        }
    }
    return true;
}

static void eval_accuracy(
    const std::vector<OracleItem>& oracle,
    const uint32_t* got_ids,
    uint32_t got_count,
    double& out_recall_pct,
    double& out_rank_match_pct) {
    uint32_t k = static_cast<uint32_t>(oracle.size());
    if (k == 0 || got_count != k) {
        out_recall_pct = 0.0;
        out_rank_match_pct = 0.0;
        return;
    }
    std::unordered_set<uint32_t> oracle_set;
    oracle_set.reserve(k * 2);
    for (const auto& item : oracle) oracle_set.insert(item.id);

    uint32_t hits = 0;
    bool strict_match = true;
    for (uint32_t i = 0; i < k; ++i) {
        if (oracle_set.count(got_ids[i])) hits++;
        if (got_ids[i] != oracle[i].id) strict_match = false;
    }
    out_recall_pct = (100.0 * hits) / static_cast<double>(k);
    out_rank_match_pct = strict_match ? 100.0 : 0.0;
}

struct LatencyStats {
    double mean_us = 0.0;
    double p50_us = 0.0;
    double p95_us = 0.0;
    double qps = 0.0;
};

static LatencyStats compute_stats(std::vector<double> samples_us) {
    if (samples_us.empty()) return {};
    std::sort(samples_us.begin(), samples_us.end());
    double sum = std::accumulate(samples_us.begin(), samples_us.end(), 0.0);
    double mean = sum / static_cast<double>(samples_us.size());
    size_t n = samples_us.size();
    double p50 = samples_us[n / 2];
    size_t p95_idx = static_cast<size_t>(std::min<size_t>(n - 1, (n * 95) / 100));
    double p95 = samples_us[p95_idx];
    double qps = (mean > 0.0) ? (1e6 / mean) : 0.0;
    return { mean, p50, p95, qps };
}

static std::vector<uint32_t> sample_unique_ids(uint32_t N, uint32_t M, std::mt19937& rng) {
    std::vector<uint32_t> ids;
    ids.reserve(M);
    // Reservoir/Floyd sampling for exact unique sorted IDs without allocating N
    std::unordered_set<uint32_t> chosen;
    chosen.reserve(M * 2);
    for (uint32_t i = N - M; i < N; ++i) {
        std::uniform_int_distribution<uint32_t> dist(0, i);
        uint32_t r = dist(rng);
        if (!chosen.insert(r).second) {
            chosen.insert(i);
        }
    }
    ids.assign(chosen.begin(), chosen.end());
    std::sort(ids.begin(), ids.end());
    return ids;
}

int main(int argc, char** argv) {
    uint32_t dim = 64;
    uint32_t num_corpus = 90000;
    uint32_t num_queries = 500;
    uint32_t reps = 3;
    bool skip_buffer = false;
    std::string out_csv = "results/sdk_comparison/native_n90k_d64.csv";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--dim" && i + 1 < argc) dim = static_cast<uint32_t>(std::stoul(argv[++i]));
        else if (arg == "--n" && i + 1 < argc) num_corpus = static_cast<uint32_t>(std::stoul(argv[++i]));
        else if (arg == "--queries" && i + 1 < argc) num_queries = static_cast<uint32_t>(std::stoul(argv[++i]));
        else if (arg == "--reps" && i + 1 < argc) reps = static_cast<uint32_t>(std::stoul(argv[++i]));
        else if (arg == "--skip-buffer") skip_buffer = true;
        else if (arg == "--out" && i + 1 < argc) out_csv = argv[++i];
    }

    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(), 0x1);

    std::cout << "====================================================================\n";
    std::cout << "ALG SDK (alg.dll) vs FAISS FlatL2 - Native Benchmark\n";
    std::cout << "====================================================================\n";
    std::cout << "ALG Version : " << alg_get_version() << "\n";
    std::cout << "ALG Active  : " << alg_get_active_isa_name() << "\n";
    std::cout << "Config      : N=" << num_corpus << ", D=" << dim
              << ", Queries=" << num_queries << ", Reps=" << reps << "\n\n";

    std::string data_dir = "data/hito3_3/D" + std::to_string(dim);
    std::vector<float> corpus;
    std::vector<float> queries;
    if (!load_raw_f32(data_dir + "/corpus.f32", corpus, 0)) return 1;
    if (!load_raw_f32(data_dir + "/queries.f32", queries, static_cast<size_t>(num_queries) * dim)) return 1;

    uint32_t file_vectors = static_cast<uint32_t>(corpus.size() / dim);
    if (num_corpus <= file_vectors) {
        corpus.resize(static_cast<size_t>(num_corpus) * dim);
    } else {
        const uint32_t seed = 20261112u;
        std::mt19937 gen_rng(seed);
        std::uniform_int_distribution<uint32_t> idx_dist(0, file_vectors - 1);
        std::uniform_real_distribution<float> w_dist(0.15f, 0.85f);

        corpus.resize(static_cast<size_t>(num_corpus) * dim);
        for (uint32_t i = file_vectors; i < num_corpus; ++i) {
            uint32_t a = idx_dist(gen_rng);
            uint32_t b = idx_dist(gen_rng);
            while (b == a) b = idx_dist(gen_rng);
            float w = w_dist(gen_rng);
            const float* va = corpus.data() + static_cast<size_t>(a) * dim;
            const float* vb = corpus.data() + static_cast<size_t>(b) * dim;
            float* dst = corpus.data() + static_cast<size_t>(i) * dim;
            double norm_sq = 0.0;
            for (uint32_t d = 0; d < dim; ++d) {
                float val = w * va[d] + (1.0f - w) * vb[d];
                dst[d] = val;
                norm_sq += static_cast<double>(val) * static_cast<double>(val);
            }
            double norm = std::sqrt(norm_sq);
            if (norm > 1e-6) {
                float inv = static_cast<float>(1.0 / norm);
                for (uint32_t d = 0; d < dim; ++d) dst[d] *= inv;
            } else {
                std::memcpy(dst, va, dim * sizeof(float));
            }
        }

        // Compute sample L2^2 distance stats (queries vs real vs synthetic)
        auto sample_dist_stats = [&](uint32_t start_idx, uint32_t end_idx) {
            double sum = 0.0, sum_sq = 0.0, min_d = 1e30, max_d = 0.0;
            uint32_t cnt = 0;
            std::uniform_int_distribution<uint32_t> d_idx(start_idx, end_idx - 1);
            for (uint32_t s = 0; s < 5000; ++s) {
                const float* qv = queries.data() + static_cast<size_t>(s % num_queries) * dim;
                const float* cv = corpus.data() + static_cast<size_t>(d_idx(gen_rng)) * dim;
                double d2 = 0.0;
                for (uint32_t d = 0; d < dim; ++d) {
                    double diff = static_cast<double>(qv[d]) - static_cast<double>(cv[d]);
                    d2 += diff * diff;
                }
                sum += d2; sum_sq += d2 * d2;
                if (d2 < min_d) min_d = d2;
                if (d2 > max_d) max_d = d2;
                cnt++;
            }
            double mean = sum / cnt;
            double var = std::max(0.0, (sum_sq / cnt) - mean * mean);
            std::cout << "mean=" << mean << ", std=" << std::sqrt(var)
                      << ", min=" << min_d << ", max=" << max_d;
        };

        uint32_t syn_count = num_corpus - file_vectors;
        double real_pct = (100.0 * file_vectors) / num_corpus;
        double syn_pct = (100.0 * syn_count) / num_corpus;
        std::cout << "[SYNTHETIC SCALABILITY EXPANSION]\n"
                  << "  Original real corpus : " << file_vectors << " vectors (" << real_pct << "%)\n"
                  << "  Synthetic derived    : " << syn_count << " vectors (" << syn_pct << "%)\n"
                  << "  Method               : L2-normalized convex mix w*v_a + (1-w)*v_b, w~U(0.15,0.85), seed=" << seed << "\n"
                  << "  L2^2 dist (q vs real): ";
        sample_dist_stats(0, file_vectors);
        std::cout << "\n  L2^2 dist (q vs syn) : ";
        sample_dist_stats(file_vectors, num_corpus);
        std::cout << "\n\n";
    }
    queries.resize(static_cast<size_t>(num_queries) * dim);

    LLCEvictor evictor(36ULL * 1024ULL * 1024ULL);

    // -------------------------------------------------------------------------
    // 1. Measure Index Construction & Memory Footprint Separately
    // -------------------------------------------------------------------------
    MemSnapshot mem_base = get_process_memory();

    uint64_t t0 = now_ns();
    AlgIndexConfig cfg{};
    cfg.dimension = dim;
    cfg.initial_capacity = num_corpus;
    cfg.metric = ALG_METRIC_L2_SQR;
    AlgIndex* alg_idx = nullptr;
    if (alg_create_index(&cfg, &alg_idx) != ALG_OK) {
        std::cerr << "[ERROR] alg_create_index failed\n";
        return 1;
    }
    if (alg_set_vectors(alg_idx, num_corpus, corpus.data()) != ALG_OK) {
        std::cerr << "[ERROR] alg_set_vectors failed\n";
        return 1;
    }
    uint64_t t1 = now_ns();
    double alg_build_ms = static_cast<double>(t1 - t0) / 1e6;
    MemSnapshot mem_after_alg = get_process_memory();

    // Initialize FAISS (note: first call also initializes embedded Python/NumPy runtime)
    FaissFlatHandle faiss_warmup = faiss_shim_create_flat_l2(dim);
    faiss_shim_free_flat(faiss_warmup);
    MemSnapshot mem_before_faiss = get_process_memory();

    uint64_t t2 = now_ns();
    FaissFlatHandle faiss_idx = faiss_shim_create_flat_l2(dim);
    if (!faiss_idx || !faiss_shim_flat_add(faiss_idx, num_corpus, corpus.data())) {
        std::cerr << "[ERROR] faiss_shim_create_flat_l2 / flat_add failed\n";
        return 1;
    }
    uint64_t t3 = now_ns();
    double faiss_build_ms = static_cast<double>(t3 - t2) / 1e6;
    MemSnapshot mem_after_faiss = get_process_memory();

    double raw_corpus_mb = static_cast<double>(static_cast<size_t>(num_corpus) * dim * sizeof(float)) / (1024.0 * 1024.0);
    double alg_priv_mb = static_cast<double>(mem_after_alg.private_bytes > mem_base.private_bytes
        ? mem_after_alg.private_bytes - mem_base.private_bytes : 0) / (1024.0 * 1024.0);
    double faiss_priv_mb = static_cast<double>(mem_after_faiss.private_bytes > mem_before_faiss.private_bytes
        ? mem_after_faiss.private_bytes - mem_before_faiss.private_bytes : 0) / (1024.0 * 1024.0);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "[Index Build & Memory (N=" << num_corpus << ", D=" << dim << ", Raw=" << raw_corpus_mb << " MB)]\n";
    std::cout << "  ALG   : build = " << alg_build_ms << " ms, PrivateDelta = " << alg_priv_mb << " MB\n";
    std::cout << "  FAISS : build = " << faiss_build_ms << " ms, PrivateDelta = " << faiss_priv_mb << " MB\n\n";

    std::ofstream csv(out_csv);
    if (!csv.is_open()) {
        std::cerr << "[ERROR] Cannot open output CSV: " << out_csv << "\n";
        return 1;
    }
    csv << "scenario,alg_fn,faiss_fn,N,D,selectivity_pct,candidates,top_k,cache_mode,"
        << "alg_mean_us,alg_p50_us,alg_p95_us,alg_qps,"
        << "faiss_mean_us,faiss_p50_us,faiss_p95_us,faiss_qps,"
        << "speedup_mean,advantage_pct,alg_recall_pct,alg_rank_pct,faiss_recall_pct,faiss_rank_pct\n";

    std::vector<uint32_t> topk_list = { 10, 100 };
    std::vector<std::string> cache_modes = { "WARM_CACHE", "COLD_LLC_FLUSH" };

    // Reusable result buffers
    std::vector<AlgSearchResult> alg_res(100);
    std::vector<float> faiss_dists(100);
    std::vector<int64_t> faiss_labels(100);
    std::vector<uint32_t> alg_ids_tmp(100);
    std::vector<uint32_t> faiss_ids_tmp(100);

    auto report_row = [&](const std::string& scenario, const std::string& alg_fn, const std::string& faiss_fn,
                          double sel_pct, uint32_t cands, uint32_t k, const std::string& cache_mode,
                          const LatencyStats& alg_st, const LatencyStats& faiss_st,
                          double alg_rec, double alg_rm, double faiss_rec, double faiss_rm) {
        double speedup = (alg_st.mean_us > 0.0) ? (faiss_st.mean_us / alg_st.mean_us) : 0.0;
        double adv_pct = (speedup - 1.0) * 100.0;

        std::cout << std::left << std::setw(18) << scenario
                  << " | sel=" << std::setw(5) << sel_pct << "% k=" << std::setw(3) << k
                  << " | " << std::setw(14) << cache_mode
                  << " | ALG=" << std::right << std::setw(8) << alg_st.mean_us << " us"
                  << " | FAISS=" << std::setw(8) << faiss_st.mean_us << " us"
                  << " | Speedup=" << std::setw(5) << speedup << "x (" << (adv_pct >= 0 ? "+" : "") << adv_pct << "%)"
                  << " | Recall(A/F)=" << alg_rec << "%/" << faiss_rec << "%\n";

        csv << scenario << "," << alg_fn << "," << faiss_fn << ","
            << num_corpus << "," << dim << "," << sel_pct << "," << cands << "," << k << "," << cache_mode << ","
            << alg_st.mean_us << "," << alg_st.p50_us << "," << alg_st.p95_us << "," << alg_st.qps << ","
            << faiss_st.mean_us << "," << faiss_st.p50_us << "," << faiss_st.p95_us << "," << faiss_st.qps << ","
            << speedup << "," << adv_pct << "," << alg_rec << "," << alg_rm << "," << faiss_rec << "," << faiss_rm << "\n";
    };

    // =========================================================================
    // SCENARIO 1: FULL CORPUS (100%) - alg_search vs FAISS IndexFlatL2
    // =========================================================================
    std::cout << "--- Scenario 1: Full Corpus (alg_search vs FAISS IndexFlatL2) ---\n";
    for (uint32_t k : topk_list) {
        // Precompute oracle for each query once
        std::vector<std::vector<OracleItem>> oracles(num_queries);
        for (uint32_t q = 0; q < num_queries; ++q) {
            oracles[q] = compute_oracle_topk(corpus.data(), dim, nullptr, num_corpus,
                                             queries.data() + static_cast<size_t>(q) * dim, k);
        }

        for (const auto& cm : cache_modes) {
            bool is_cold = (cm == "COLD_LLC_FLUSH");
            std::vector<double> alg_times, faiss_times;
            alg_times.reserve(num_queries * reps);
            faiss_times.reserve(num_queries * reps);
            double alg_rec_sum = 0, alg_rm_sum = 0, faiss_rec_sum = 0, faiss_rm_sum = 0;
            uint32_t acc_count = 0;

            for (uint32_t r = 0; r < reps; ++r) {
                for (uint32_t q = 0; q < num_queries; ++q) {
                    const float* qptr = queries.data() + static_cast<size_t>(q) * dim;
                    bool alg_first = ((q + r) % 2 == 0);

                    auto run_alg = [&]() {
                        if (is_cold) evictor.flush();
                        else {
                            uint32_t dummy = 0;
                            alg_search(alg_idx, qptr, k, ALG_SEARCH_FAST_FLOAT32, alg_res.data(), &dummy);
                        }
                        uint32_t out_cnt = 0;
                        uint64_t s0 = now_ns();
                        alg_search(alg_idx, qptr, k, ALG_SEARCH_FAST_FLOAT32, alg_res.data(), &out_cnt);
                        uint64_t s1 = now_ns();
                        alg_times.push_back(static_cast<double>(s1 - s0) / 1e3);
                        if (r == 0) {
                            for (uint32_t i = 0; i < out_cnt; ++i) alg_ids_tmp[i] = alg_res[i].id;
                        }
                    };

                    auto run_faiss = [&]() {
                        if (is_cold) evictor.flush();
                        else {
                            faiss_shim_search_flat(faiss_idx, 1, qptr, k, faiss_dists.data(), faiss_labels.data(), nullptr);
                        }
                        uint64_t s0 = now_ns();
                        faiss_shim_search_flat(faiss_idx, 1, qptr, k, faiss_dists.data(), faiss_labels.data(), nullptr);
                        uint64_t s1 = now_ns();
                        faiss_times.push_back(static_cast<double>(s1 - s0) / 1e3);
                        if (r == 0) {
                            for (uint32_t i = 0; i < k; ++i) faiss_ids_tmp[i] = static_cast<uint32_t>(faiss_labels[i]);
                        }
                    };

                    if (alg_first) { run_alg(); run_faiss(); }
                    else           { run_faiss(); run_alg(); }

                    if (r == 0) {
                        double ar, arm, fr, frm;
                        eval_accuracy(oracles[q], alg_ids_tmp.data(), k, ar, arm);
                        eval_accuracy(oracles[q], faiss_ids_tmp.data(), k, fr, frm);
                        alg_rec_sum += ar; alg_rm_sum += arm;
                        faiss_rec_sum += fr; faiss_rm_sum += frm;
                        acc_count++;
                    }
                }
            }
            report_row("1_FullCorpus", "alg_search", "IndexFlatL2", 100.0, num_corpus, k, cm,
                       compute_stats(alg_times), compute_stats(faiss_times),
                       alg_rec_sum / acc_count, alg_rm_sum / acc_count,
                       faiss_rec_sum / acc_count, faiss_rm_sum / acc_count);
        }
    }

    // =========================================================================
    // SCENARIOS 2 & 3: FILTERED (1% and 10%)
    // =========================================================================
    std::vector<double> selectivities = { 1.0, 10.0 };
    std::mt19937 rng(42);
    size_t bitmap_bytes = (num_corpus + 7) / 8;

    for (double sel : selectivities) {
        uint32_t M = static_cast<uint32_t>(std::round((sel / 100.0) * num_corpus));
        std::cout << "\n--- Generating " << num_queries << " candidate sets (sel=" << sel << "%, M=" << M << ") ---\n";

        std::vector<std::vector<uint32_t>> cand_ids_all(num_queries);
        std::vector<std::vector<uint8_t>> bitmaps_all(num_queries, std::vector<uint8_t>(bitmap_bytes, 0));
        for (uint32_t q = 0; q < num_queries; ++q) {
            cand_ids_all[q] = sample_unique_ids(num_corpus, M, rng);
            for (uint32_t id : cand_ids_all[q]) {
                bitmaps_all[q][id >> 3] |= static_cast<uint8_t>(1u << (id & 7u));
            }
        }

        for (uint32_t k : topk_list) {
            std::vector<std::vector<OracleItem>> oracles(num_queries);
            for (uint32_t q = 0; q < num_queries; ++q) {
                oracles[q] = compute_oracle_topk(corpus.data(), dim, cand_ids_all[q].data(), M,
                                                 queries.data() + static_cast<size_t>(q) * dim, k);
            }

            // -----------------------------------------------------------------
            // SCENARIO 2: CANDIDATES BY ID (alg_search_compact vs FAISS Bitmap)
            // -----------------------------------------------------------------
            for (const auto& cm : cache_modes) {
                bool is_cold = (cm == "COLD_LLC_FLUSH");
                std::vector<double> alg_times, faiss_times;
                alg_times.reserve(num_queries * reps);
                faiss_times.reserve(num_queries * reps);
                double alg_rec_sum = 0, alg_rm_sum = 0, faiss_rec_sum = 0, faiss_rm_sum = 0;
                uint32_t acc_count = 0;

                for (uint32_t q = 0; q < num_queries; ++q) {
                    const float* qptr = queries.data() + static_cast<size_t>(q) * dim;
                    FaissSelectorHandle sel_h = faiss_shim_create_bitmap_selector(num_corpus, bitmaps_all[q].data());
                    FaissParamsHandle params_h = faiss_shim_create_flat_params(sel_h);

                    for (uint32_t r = 0; r < reps; ++r) {
                        bool alg_first = ((q + r) % 2 == 0);

                        auto run_alg = [&]() {
                            if (is_cold) evictor.flush();
                            else {
                                uint32_t dummy = 0;
                                alg_search_compact(alg_idx, qptr, cand_ids_all[q].data(), M, k,
                                                   ALG_SEARCH_FAST_FLOAT32, alg_res.data(), &dummy);
                            }
                            uint32_t out_cnt = 0;
                            uint64_t s0 = now_ns();
                            alg_search_compact(alg_idx, qptr, cand_ids_all[q].data(), M, k,
                                               ALG_SEARCH_FAST_FLOAT32, alg_res.data(), &out_cnt);
                            uint64_t s1 = now_ns();
                            alg_times.push_back(static_cast<double>(s1 - s0) / 1e3);
                            if (r == 0) {
                                for (uint32_t i = 0; i < out_cnt; ++i) alg_ids_tmp[i] = alg_res[i].id;
                            }
                        };

                        auto run_faiss = [&]() {
                            if (is_cold) evictor.flush();
                            else {
                                faiss_shim_search_flat(faiss_idx, 1, qptr, k,
                                                       faiss_dists.data(), faiss_labels.data(), params_h);
                            }
                            uint64_t s0 = now_ns();
                            faiss_shim_search_flat(faiss_idx, 1, qptr, k,
                                                   faiss_dists.data(), faiss_labels.data(), params_h);
                            uint64_t s1 = now_ns();
                            faiss_times.push_back(static_cast<double>(s1 - s0) / 1e3);
                            if (r == 0) {
                                for (uint32_t i = 0; i < k; ++i) faiss_ids_tmp[i] = static_cast<uint32_t>(faiss_labels[i]);
                            }
                        };

                        if (alg_first) { run_alg(); run_faiss(); }
                        else           { run_faiss(); run_alg(); }

                        if (r == 0) {
                            double ar, arm, fr, frm;
                            eval_accuracy(oracles[q], alg_ids_tmp.data(), k, ar, arm);
                            eval_accuracy(oracles[q], faiss_ids_tmp.data(), k, fr, frm);
                            alg_rec_sum += ar; alg_rm_sum += arm;
                            faiss_rec_sum += fr; faiss_rm_sum += frm;
                            acc_count++;
                        }
                    }

                    faiss_shim_free_params(params_h);
                    faiss_shim_free_selector(sel_h);
                }

                report_row("2_ByID_Compact", "alg_search_compact", "FlatL2+Bitmap", sel, M, k, cm,
                           compute_stats(alg_times), compute_stats(faiss_times),
                           alg_rec_sum / acc_count, alg_rm_sum / acc_count,
                           faiss_rec_sum / acc_count, faiss_rm_sum / acc_count);
            }

            // -----------------------------------------------------------------
            // SCENARIO 2b: FAIR BY-ID (alg_search_compact vs FAISS compute_distance_subset + TopK)
            // -----------------------------------------------------------------
            for (const auto& cm : cache_modes) {
                bool is_cold = (cm == "COLD_LLC_FLUSH");
                std::vector<double> alg_times, faiss_times, faiss_dist_only_times;
                alg_times.reserve(num_queries * reps);
                faiss_times.reserve(num_queries * reps);
                faiss_dist_only_times.reserve(num_queries * reps);
                double alg_rec_sum = 0, alg_rm_sum = 0, faiss_rec_sum = 0, faiss_rm_sum = 0;
                uint32_t acc_count = 0;

                std::vector<int64_t> subset_labels_i64(M);
                std::vector<float> subset_dists(M);
                std::vector<FloatHeapItem> heap_buf(k);

                for (uint32_t q = 0; q < num_queries; ++q) {
                    const float* qptr = queries.data() + static_cast<size_t>(q) * dim;
                    for (uint32_t i = 0; i < M; ++i) {
                        subset_labels_i64[i] = static_cast<int64_t>(cand_ids_all[q][i]);
                    }

                    for (uint32_t r = 0; r < reps; ++r) {
                        bool alg_first = ((q + r) % 2 == 0);

                        auto run_alg = [&]() {
                            if (is_cold) evictor.flush();
                            else {
                                uint32_t dummy = 0;
                                alg_search_compact(alg_idx, qptr, cand_ids_all[q].data(), M, k,
                                                   ALG_SEARCH_FAST_FLOAT32, alg_res.data(), &dummy);
                            }
                            uint32_t out_cnt = 0;
                            uint64_t s0 = now_ns();
                            alg_search_compact(alg_idx, qptr, cand_ids_all[q].data(), M, k,
                                               ALG_SEARCH_FAST_FLOAT32, alg_res.data(), &out_cnt);
                            uint64_t s1 = now_ns();
                            alg_times.push_back(static_cast<double>(s1 - s0) / 1e3);
                            if (r == 0) {
                                for (uint32_t i = 0; i < out_cnt; ++i) alg_ids_tmp[i] = alg_res[i].id;
                            }
                        };

                        auto run_faiss = [&]() {
                            if (is_cold) evictor.flush();
                            else {
                                faiss_shim_compute_distance_subset(faiss_idx, 1, qptr, M,
                                                                   subset_dists.data(), subset_labels_i64.data());
                                select_topk_from_distances(subset_dists.data(), cand_ids_all[q].data(),
                                                           M, k, heap_buf, faiss_ids_tmp.data());
                            }
                            uint64_t s0 = now_ns();
                            faiss_shim_compute_distance_subset(faiss_idx, 1, qptr, M,
                                                               subset_dists.data(), subset_labels_i64.data());
                            uint64_t s1 = now_ns();
                            select_topk_from_distances(subset_dists.data(), cand_ids_all[q].data(),
                                                       M, k, heap_buf, faiss_ids_tmp.data());
                            uint64_t s2 = now_ns();
                            faiss_dist_only_times.push_back(static_cast<double>(s1 - s0) / 1e3);
                            faiss_times.push_back(static_cast<double>(s2 - s0) / 1e3);
                        };

                        if (alg_first) { run_alg(); run_faiss(); }
                        else           { run_faiss(); run_alg(); }

                        if (r == 0) {
                            double ar, arm, fr, frm;
                            eval_accuracy(oracles[q], alg_ids_tmp.data(), k, ar, arm);
                            eval_accuracy(oracles[q], faiss_ids_tmp.data(), k, fr, frm);
                            alg_rec_sum += ar; alg_rm_sum += arm;
                            faiss_rec_sum += fr; faiss_rm_sum += frm;
                            acc_count++;
                        }
                    }
                }

                auto faiss_dist_st = compute_stats(faiss_dist_only_times);
                report_row("2b_ByID_Subset", "alg_search_compact", "FlatL2_Subset+TopK", sel, M, k, cm,
                           compute_stats(alg_times), compute_stats(faiss_times),
                           alg_rec_sum / acc_count, alg_rm_sum / acc_count,
                           faiss_rec_sum / acc_count, faiss_rm_sum / acc_count);
                std::cout << "    [Detail 2b] FAISS compute_distance_subset only mean = "
                          << faiss_dist_st.mean_us << " us\n";
            }

            // -----------------------------------------------------------------
            // SCENARIO 3: COMPACT BUFFER (alg_search_buffer vs FAISS Sub-Index)
            // -----------------------------------------------------------------
            if (!skip_buffer) for (const auto& cm : cache_modes) {
                bool is_cold = (cm == "COLD_LLC_FLUSH");
                std::vector<double> alg_times, faiss_times;
                alg_times.reserve(num_queries * reps);
                faiss_times.reserve(num_queries * reps);
                double alg_rec_sum = 0, alg_rm_sum = 0, faiss_rec_sum = 0, faiss_rm_sum = 0;
                uint32_t acc_count = 0;

                std::vector<float> compact_buf(static_cast<size_t>(M) * dim);

                for (uint32_t q = 0; q < num_queries; ++q) {
                    const float* qptr = queries.data() + static_cast<size_t>(q) * dim;
                    for (uint32_t i = 0; i < M; ++i) {
                        std::memcpy(compact_buf.data() + static_cast<size_t>(i) * dim,
                                    corpus.data() + static_cast<size_t>(cand_ids_all[q][i]) * dim,
                                    dim * sizeof(float));
                    }

                    FaissFlatHandle faiss_sub = faiss_shim_create_flat_l2(dim);
                    faiss_shim_flat_add(faiss_sub, M, compact_buf.data());

                    for (uint32_t r = 0; r < reps; ++r) {
                        bool alg_first = ((q + r) % 2 == 0);

                        auto run_alg = [&]() {
                            if (is_cold) evictor.flush();
                            else {
                                uint32_t dummy = 0;
                                alg_search_buffer(qptr, compact_buf.data(), cand_ids_all[q].data(), M, dim, k,
                                                  ALG_SEARCH_FAST_FLOAT32, alg_res.data(), &dummy);
                            }
                            uint32_t out_cnt = 0;
                            uint64_t s0 = now_ns();
                            alg_search_buffer(qptr, compact_buf.data(), cand_ids_all[q].data(), M, dim, k,
                                              ALG_SEARCH_FAST_FLOAT32, alg_res.data(), &out_cnt);
                            uint64_t s1 = now_ns();
                            alg_times.push_back(static_cast<double>(s1 - s0) / 1e3);
                            if (r == 0) {
                                for (uint32_t i = 0; i < out_cnt; ++i) alg_ids_tmp[i] = alg_res[i].id;
                            }
                        };

                        auto run_faiss = [&]() {
                            if (is_cold) evictor.flush();
                            else {
                                faiss_shim_search_flat(faiss_sub, 1, qptr, k,
                                                       faiss_dists.data(), faiss_labels.data(), nullptr);
                            }
                            uint64_t s0 = now_ns();
                            faiss_shim_search_flat(faiss_sub, 1, qptr, k,
                                                   faiss_dists.data(), faiss_labels.data(), nullptr);
                            uint64_t s1 = now_ns();
                            faiss_times.push_back(static_cast<double>(s1 - s0) / 1e3);
                            if (r == 0) {
                                for (uint32_t i = 0; i < k; ++i) {
                                    int64_t lid = faiss_labels[i];
                                    faiss_ids_tmp[i] = (lid >= 0 && lid < static_cast<int64_t>(M))
                                        ? cand_ids_all[q][static_cast<size_t>(lid)] : UINT32_MAX;
                                }
                            }
                        };

                        if (alg_first) { run_alg(); run_faiss(); }
                        else           { run_faiss(); run_alg(); }

                        if (r == 0) {
                            double ar, arm, fr, frm;
                            eval_accuracy(oracles[q], alg_ids_tmp.data(), k, ar, arm);
                            eval_accuracy(oracles[q], faiss_ids_tmp.data(), k, fr, frm);
                            alg_rec_sum += ar; alg_rm_sum += arm;
                            faiss_rec_sum += fr; faiss_rm_sum += frm;
                            acc_count++;
                        }
                    }

                    faiss_shim_free_flat(faiss_sub);
                }

                report_row("3_CompactBuffer", "alg_search_buffer", "IndexFlatL2(Sub)", sel, M, k, cm,
                           compute_stats(alg_times), compute_stats(faiss_times),
                           alg_rec_sum / acc_count, alg_rm_sum / acc_count,
                           faiss_rec_sum / acc_count, faiss_rm_sum / acc_count);
            }
        }
    }

    // =========================================================================
    // SCENARIO 4: ISOLATE alg_search_buffer VALIDATION COST (No SDK edits)
    // =========================================================================
    if (!skip_buffer) {
    std::cout << "\n--- Scenario 4: Isolate alg_search_buffer Validation Cost (D=" << dim << ", k=10) ---\n";
    std::vector<uint32_t> val_sizes = { 100, 900, 3000, 9000, 30000 };
    std::ofstream val_csv("results/sdk_comparison/buffer_validation_isolation.csv");
    val_csv << "M,D,total_floats,cache_mode,alg_search_buffer_us,validation_only_us,search_excl_val_us,validation_pct\n";

    volatile bool val_sink = false;
    for (uint32_t M : val_sizes) {
        if (M > num_corpus) continue;
        std::vector<float> buf(static_cast<size_t>(M) * dim);
        std::memcpy(buf.data(), corpus.data(), buf.size() * sizeof(float));
        std::vector<uint32_t> ids(M);
        std::iota(ids.begin(), ids.end(), 0);
        const float* qptr = queries.data();
        const uint32_t val_iters = 100;

        for (const auto& cm : cache_modes) {
            bool is_cold = (cm == "COLD_LLC_FLUSH");
            std::vector<double> tot_us, val_us;
            tot_us.reserve(val_iters);
            val_us.reserve(val_iters);

            for (uint32_t it = 0; it < val_iters; ++it) {
                if (is_cold) evictor.flush();
                else {
                    uint32_t dummy = 0;
                    alg_search_buffer(qptr, buf.data(), ids.data(), M, dim, 10, ALG_SEARCH_FAST_FLOAT32, alg_res.data(), &dummy);
                }
                uint32_t out_cnt = 0;
                uint64_t t_a = now_ns();
                alg_search_buffer(qptr, buf.data(), ids.data(), M, dim, 10, ALG_SEARCH_FAST_FLOAT32, alg_res.data(), &out_cnt);
                uint64_t t_b = now_ns();
                tot_us.push_back(static_cast<double>(t_b - t_a) / 1e3);

                if (is_cold) evictor.flush();
                else {
                    val_sink = ref_is_finite_buffer(buf.data(), buf.size());
                }
                uint64_t v_a = now_ns();
                bool ok = ref_is_finite_buffer(buf.data(), buf.size());
                uint64_t v_b = now_ns();
                val_sink = ok;
                val_us.push_back(static_cast<double>(v_b - v_a) / 1e3);
            }

            auto st_tot = compute_stats(tot_us);
            auto st_val = compute_stats(val_us);
            double excl = st_tot.mean_us - st_val.mean_us;
            double pct = (st_tot.mean_us > 0.0) ? (100.0 * st_val.mean_us / st_tot.mean_us) : 0.0;

            std::cout << "  M=" << std::setw(5) << M << " | " << std::setw(14) << cm
                      << " | Total=" << std::setw(8) << st_tot.mean_us << " us"
                      << " | Validation=" << std::setw(8) << st_val.mean_us << " us (" << pct << "%)"
                      << " | NetSearch~=" << std::setw(8) << excl << " us\n";

            val_csv << M << "," << dim << "," << buf.size() << "," << cm << ","
                    << st_tot.mean_us << "," << st_val.mean_us << "," << excl << "," << pct << "\n";
        }
    }
    (void)val_sink;
    }

    faiss_shim_free_flat(faiss_idx);
    alg_free_index(alg_idx);

    std::cout << "\nSaved CSV to: " << out_csv << "\n";
    return 0;
}
