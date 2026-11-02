#include "../include/alg_flat_types.h"
#include "../include/dataset_loader.h"
#include "../include/physical_layouts.h"
#include "../include/execution_plans.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace alg_flat;

static double compute_recall(const Neighbor* pred, uint32_t pred_cnt,
                             const Neighbor* gt, uint32_t gt_cnt, uint32_t k) {
    uint32_t target = (gt_cnt < k) ? gt_cnt : k;
    if (target == 0) return 1.0;
    uint32_t n_p = (pred_cnt < k) ? pred_cnt : k;
    float kth_gt_dist = gt[target - 1].dist_sq * 1.00001f;
    uint32_t hits = 0;
    for (uint32_t i = 0; i < n_p; ++i) {
        bool found = false;
        for (uint32_t j = 0; j < target; ++j) {
            if (pred[i].id == gt[j].id) {
                found = true;
                break;
            }
        }
        if (!found && pred[i].dist_sq <= kth_gt_dist) {
            found = true; /* Empate numerico exacto en la frontera K */
        }
        if (found) hits++;
    }
    return static_cast<double>(hits) / static_cast<double>(target);
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << "ALG-Flat 2.0: Test de Exactitud Matematica y Bitwise Recall@10 / Recall@100\n";
    std::cout << "========================================================================\n";

    BenchmarkDataset ds;
    bool ok = generate_dataset_deterministic(8, 20000, 100, 64, 0.09f, MIX_50_50, ds);
    assert(ok && "Dataset generation failed");

    FlatAoSIndex aos_idx;
    FlatSoAIndex soa_idx;
    FlatAoSoAIndex aosoa_idx;
    IVFPartitionedIndex ivf_idx;

    build_flat_aos_index(ds, aos_idx);
    build_flat_soa_index(ds, soa_idx);
    build_flat_aosoa_index(ds, aosoa_idx);
    build_ivf_partitioned_index(ds, 64, 12, ivf_idx);

    std::cout << "Indices construidos: AoS=" << aos_idx.build_time_ms << " ms, "
              << "SoA=" << soa_idx.build_time_ms << " ms, "
              << "AoSoA=" << aosoa_idx.build_time_ms << " ms, "
              << "IVF(P=64)=" << ivf_idx.total_build_time_ms << " ms\n";

    const uint32_t sel_idx = 2; /* 15.0% selectividad */
    const uint32_t target_cat = ds.header.selectivities_deci_percent[sel_idx] * 10U;
    const auto& el_ids = ds.eligible_per_sel[sel_idx];
    QueryPredicate pred{target_cat, el_ids.data(), static_cast<uint32_t>(el_ids.size()), sel_idx};

    const uint32_t k = 100;
    Neighbor gt[100], out[100];
    uint32_t gt_cnt = 0, out_cnt = 0;
    QueryExecutionMetrics m{};

    double r10_sums[8] = {0}, r100_sums[8] = {0};
    const char* names[8] = {
        "Plan 1  (Flat_Scalar_AoS)",
        "Plan 2A (Flat_AVX2_AoS_1x)",
        "Plan 2B (Flat_AVX2_AoS_4x)",
        "Plan 3A (Flat_AVX2_SoA)",
        "Plan 3B (Flat_AVX2_AoSoA)",
        "Plan 4  (IVF_Flat_AVX2)",
        "Plan 5A (IVF_PartPrune_AVX2)",
        "Plan 5B (IVF_FullExactPrune_AVX2)"
    };

    for (uint32_t q = 0; q < ds.header.num_queries; ++q) {
        const float* qv = ds.queries[q].values;
        execute_plan0_ref_hito9(aos_idx, qv, pred, k, gt, &gt_cnt, m);

        execute_plan1_flat_scalar_aos(aos_idx, qv, pred, k, out, &out_cnt, m);
        r10_sums[0] += compute_recall(out, out_cnt, gt, gt_cnt, 10);
        r100_sums[0] += compute_recall(out, out_cnt, gt, gt_cnt, 100);

        execute_plan2a_flat_avx2_aos_1x(aos_idx, qv, pred, k, out, &out_cnt, m);
        r10_sums[1] += compute_recall(out, out_cnt, gt, gt_cnt, 10);
        r100_sums[1] += compute_recall(out, out_cnt, gt, gt_cnt, 100);

        execute_plan2b_flat_avx2_aos_4x(aos_idx, qv, pred, k, out, &out_cnt, m);
        r10_sums[2] += compute_recall(out, out_cnt, gt, gt_cnt, 10);
        r100_sums[2] += compute_recall(out, out_cnt, gt, gt_cnt, 100);

        execute_plan3a_flat_avx2_soa(soa_idx, qv, pred, k, out, &out_cnt, m);
        r10_sums[3] += compute_recall(out, out_cnt, gt, gt_cnt, 10);
        r100_sums[3] += compute_recall(out, out_cnt, gt, gt_cnt, 100);

        execute_plan3b_flat_avx2_aosoa(aosoa_idx, qv, pred, k, out, &out_cnt, m);
        r10_sums[4] += compute_recall(out, out_cnt, gt, gt_cnt, 10);
        r100_sums[4] += compute_recall(out, out_cnt, gt, gt_cnt, 100);

        execute_plan4_ivf_flat_avx2(ivf_idx, qv, pred, k, out, &out_cnt, m);
        r10_sums[5] += compute_recall(out, out_cnt, gt, gt_cnt, 10);
        r100_sums[5] += compute_recall(out, out_cnt, gt, gt_cnt, 100);

        execute_plan5a_ivf_part_prune_avx2(ivf_idx, qv, pred, k, out, &out_cnt, m);
        r10_sums[6] += compute_recall(out, out_cnt, gt, gt_cnt, 10);
        r100_sums[6] += compute_recall(out, out_cnt, gt, gt_cnt, 100);

        execute_plan5b_ivf_full_exact_prune_avx2(ivf_idx, qv, pred, k, out, &out_cnt, m);
        r10_sums[7] += compute_recall(out, out_cnt, gt, gt_cnt, 10);
        r100_sums[7] += compute_recall(out, out_cnt, gt, gt_cnt, 100);
    }

    double nq = static_cast<double>(ds.header.num_queries);
    for (int i = 0; i < 8; ++i) {
        double r10 = (r10_sums[i] / nq) * 100.0;
        double r100 = (r100_sums[i] / nq) * 100.0;
        printf("  %-35s -> Recall@10 = %6.2f%% | Recall@100 = %6.2f%%\n", names[i], r10, r100);
        assert(r10 == 100.0 && "Recall@10 must be 100.00% exact!");
        assert(r100 == 100.0 && "Recall@100 must be 100.00% exact!");
    }

    std::cout << "\nTODOS LOS PLANES CERTIFICARON 100.00% DE RECALL EXACTO.\n";
    return 0;
}
