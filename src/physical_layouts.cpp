#include "../include/physical_layouts.h"
#include "../include/simd_kernels.h"
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace alg_flat {

static inline double elapsed_ms(std::chrono::high_resolution_clock::time_point t0,
                                std::chrono::high_resolution_clock::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static inline uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/*
 * Calcula una base ortonormal exacta Q^T (dim x dim) ordenada por varianza decreciente
 * mediante rotaciones de Jacobi en doble precision + Gram-Schmidt modificado.
 */
static void compute_pca_ortho_basis(const float* sample_vecs, uint32_t sample_cnt, uint32_t dim, float* out_qt) {
    std::vector<double> mean(dim, 0.0);
    for (uint32_t i = 0; i < sample_cnt; ++i) {
        const float* v = sample_vecs + static_cast<size_t>(i) * dim;
        for (uint32_t d = 0; d < dim; ++d) mean[d] += static_cast<double>(v[d]);
    }
    for (uint32_t d = 0; d < dim; ++d) mean[d] /= static_cast<double>(sample_cnt);

    std::vector<double> cov(static_cast<size_t>(dim) * dim, 0.0);
    for (uint32_t i = 0; i < sample_cnt; ++i) {
        const float* v = sample_vecs + static_cast<size_t>(i) * dim;
        for (uint32_t r = 0; r < dim; ++r) {
            double dr = static_cast<double>(v[r]) - mean[r];
            for (uint32_t c = r; c < dim; ++c) {
                cov[r * dim + c] += dr * (static_cast<double>(v[c]) - mean[c]);
            }
        }
    }
    for (uint32_t r = 0; r < dim; ++r) {
        for (uint32_t c = 0; c < r; ++c) cov[r * dim + c] = cov[c * dim + r];
    }

    std::vector<double> eigvecs(static_cast<size_t>(dim) * dim, 0.0);
    for (uint32_t d = 0; d < dim; ++d) eigvecs[d * dim + d] = 1.0;

    /* 8 barridos de Jacobi ciclico (suficiente para ordenar varianza y diagonalizar 64x64 en < 0.4 ms) */
    for (int sweep = 0; sweep < 8; ++sweep) {
        for (uint32_t p = 0; p < dim - 1; ++p) {
            for (uint32_t q = p + 1; q < dim; ++q) {
                double apq = cov[p * dim + q];
                if (std::fabs(apq) < 1e-12) continue;
                double app = cov[p * dim + p];
                double aqq = cov[q * dim + q];
                double tau = (aqq - app) / (2.0 * apq);
                double t = (tau >= 0.0) ? (1.0 / (tau + std::sqrt(1.0 + tau * tau)))
                                        : (-1.0 / (-tau + std::sqrt(1.0 + tau * tau)));
                double c = 1.0 / std::sqrt(1.0 + t * t);
                double s = t * c;

                cov[p * dim + p] = app - t * apq;
                cov[q * dim + q] = aqq + t * apq;
                cov[p * dim + q] = 0.0;
                cov[q * dim + p] = 0.0;

                for (uint32_t k = 0; k < dim; ++k) {
                    if (k != p && k != q) {
                        double akp = cov[k * dim + p];
                        double akq = cov[k * dim + q];
                        cov[k * dim + p] = cov[p * dim + k] = c * akp - s * akq;
                        cov[k * dim + q] = cov[q * dim + k] = s * akp + c * akq;
                    }
                    double vkp = eigvecs[k * dim + p];
                    double vkq = eigvecs[k * dim + q];
                    eigvecs[k * dim + p] = c * vkp - s * vkq;
                    eigvecs[k * dim + q] = s * vkp + c * vkq;
                }
            }
        }
    }

    /* Ordenar autovectores por autovalor (varianza) descendente */
    std::vector<uint32_t> order(dim);
    std::iota(order.begin(), order.end(), 0U);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return cov[a * dim + a] > cov[b * dim + b];
    });

    /* Construir filas de Q^T y aplicar Gram-Schmidt modificado en doble precision para ortonormalidad exacta */
    std::vector<double> rows(static_cast<size_t>(dim) * dim, 0.0);
    for (uint32_t r = 0; r < dim; ++r) {
        uint32_t col = order[r];
        for (uint32_t d = 0; d < dim; ++d) {
            rows[r * dim + d] = eigvecs[d * dim + col];
        }
        for (uint32_t prev = 0; prev < r; ++prev) {
            double dot = 0.0;
            for (uint32_t d = 0; d < dim; ++d) dot += rows[r * dim + d] * rows[prev * dim + d];
            for (uint32_t d = 0; d < dim; ++d) rows[r * dim + d] -= dot * rows[prev * dim + d];
        }
        double norm2 = 0.0;
        for (uint32_t d = 0; d < dim; ++d) norm2 += rows[r * dim + d] * rows[r * dim + d];
        double inv = 1.0 / std::sqrt(norm2 > 1e-24 ? norm2 : 1.0);
        for (uint32_t d = 0; d < dim; ++d) {
            rows[r * dim + d] *= inv;
            out_qt[r * dim + d] = static_cast<float>(rows[r * dim + d]);
        }
    }
}

static inline void rotate_vector_64(const float* qt, const float* in_v, float* out_v, uint32_t dim) {
    for (uint32_t r = 0; r < dim; ++r) {
        const float* row = qt + static_cast<size_t>(r) * dim;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        for (uint32_t d = 0; d < dim; d += 16) {
            acc0 = _mm256_fmadd_ps(_mm256_load_ps(row + d), _mm256_loadu_ps(in_v + d), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_load_ps(row + d + 8), _mm256_loadu_ps(in_v + d + 8), acc1);
        }
        out_v[r] = hsum256_ps_fast(_mm256_add_ps(acc0, acc1));
    }
}

void build_flat_aos_index(const BenchmarkDataset& ds, FlatAoSIndex& out_idx) {
    auto t0 = std::chrono::high_resolution_clock::now();
    out_idx.num_vectors = ds.header.num_vectors;
    out_idx.dim = ds.header.dimension;
    out_idx.vectors = ds.vectors;
    out_idx.categories = ds.categories;
    auto t1 = std::chrono::high_resolution_clock::now();
    out_idx.build_time_ms = elapsed_ms(t0, t1);
}

void build_flat_soa_index(const BenchmarkDataset& ds, FlatSoAIndex& out_idx) {
    auto t0 = std::chrono::high_resolution_clock::now();
    uint32_t nv = ds.header.num_vectors;
    uint32_t dim = ds.header.dimension;
    uint32_t num_blocks = (nv + 7U) / 8U;
    uint32_t stride = num_blocks * 8U;

    aligned_free_64(out_idx.soa_data);
    aligned_free_64(out_idx.categories);
    aligned_free_64(out_idx.ids);

    out_idx.num_vectors = nv;
    out_idx.num_blocks = num_blocks;
    out_idx.stride = stride;
    out_idx.dim = dim;

    out_idx.soa_data = static_cast<float*>(aligned_alloc_64(static_cast<size_t>(dim) * stride * sizeof(float)));
    out_idx.categories = static_cast<uint32_t*>(aligned_alloc_64(static_cast<size_t>(stride) * sizeof(uint32_t)));
    out_idx.ids = static_cast<uint32_t*>(aligned_alloc_64(static_cast<size_t>(stride) * sizeof(uint32_t)));

    for (uint32_t i = 0; i < stride; ++i) {
        if (i < nv) {
            out_idx.categories[i] = ds.categories[i];
            out_idx.ids[i] = i;
            const float* src = ds.vectors + static_cast<size_t>(i) * dim;
            for (uint32_t d = 0; d < dim; ++d) {
                out_idx.soa_data[static_cast<size_t>(d) * stride + i] = src[d];
            }
        } else {
            out_idx.categories[i] = UINT32_MAX;
            out_idx.ids[i] = UINT32_MAX;
            for (uint32_t d = 0; d < dim; ++d) {
                out_idx.soa_data[static_cast<size_t>(d) * stride + i] = 1e10f;
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    out_idx.build_time_ms = elapsed_ms(t0, t1);
}

void build_flat_aosoa_index(const BenchmarkDataset& ds, FlatAoSoAIndex& out_idx) {
    auto t0 = std::chrono::high_resolution_clock::now();
    uint32_t nv = ds.header.num_vectors;
    uint32_t dim = ds.header.dimension;
    uint32_t num_blocks = (nv + 7U) / 8U;
    uint32_t padded_nv = num_blocks * 8U;

    aligned_free_64(out_idx.block_data);
    aligned_free_64(out_idx.categories);
    aligned_free_64(out_idx.ids);

    out_idx.num_vectors = nv;
    out_idx.num_blocks = num_blocks;
    out_idx.dim = dim;

    out_idx.block_data = static_cast<float*>(aligned_alloc_64(static_cast<size_t>(num_blocks) * dim * 8U * sizeof(float)));
    out_idx.categories = static_cast<uint32_t*>(aligned_alloc_64(static_cast<size_t>(padded_nv) * sizeof(uint32_t)));
    out_idx.ids = static_cast<uint32_t*>(aligned_alloc_64(static_cast<size_t>(padded_nv) * sizeof(uint32_t)));

    for (uint32_t b = 0; b < num_blocks; ++b) {
        float* blk = out_idx.block_data + static_cast<size_t>(b) * dim * 8U;
        for (uint32_t lane = 0; lane < 8U; ++lane) {
            uint32_t idx = b * 8U + lane;
            if (idx < nv) {
                out_idx.categories[idx] = ds.categories[idx];
                out_idx.ids[idx] = idx;
                const float* src = ds.vectors + static_cast<size_t>(idx) * dim;
                for (uint32_t d = 0; d < dim; ++d) {
                    blk[d * 8U + lane] = src[d];
                }
            } else {
                out_idx.categories[idx] = UINT32_MAX;
                out_idx.ids[idx] = UINT32_MAX;
                for (uint32_t d = 0; d < dim; ++d) {
                    blk[d * 8U + lane] = 1e10f;
                }
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    out_idx.build_time_ms = elapsed_ms(t0, t1);
}

void build_ivf_partitioned_index(const BenchmarkDataset& ds,
                                 uint32_t num_partitions,
                                 uint32_t kmeans_iters,
                                 IVFPartitionedIndex& out_idx) {
    auto t_start = std::chrono::high_resolution_clock::now();
    const uint32_t nv = ds.header.num_vectors;
    const uint32_t dim = ds.header.dimension;

    aligned_free_64(out_idx.ortho_basis);
    aligned_free_64(out_idx.centroids);
    aligned_free_64(out_idx.rotated_centroids);
    aligned_free_64(out_idx.part_aos_vectors);
    aligned_free_64(out_idx.part_aos_ids);
    aligned_free_64(out_idx.part_aos_categories);
    aligned_free_64(out_idx.part_aos_radii);
    aligned_free_64(out_idx.part_aos_tail48_norm);
    aligned_free_64(out_idx.part_aos_tail32_norm);

    out_idx.num_vectors = nv;
    out_idx.dim = dim;
    out_idx.num_partitions = num_partitions;

    out_idx.ortho_basis = static_cast<float*>(aligned_alloc_64(static_cast<size_t>(dim) * dim * sizeof(float)));
    out_idx.centroids = static_cast<float*>(aligned_alloc_64(static_cast<size_t>(num_partitions) * dim * sizeof(float)));
    out_idx.rotated_centroids = static_cast<float*>(aligned_alloc_64(static_cast<size_t>(num_partitions) * dim * sizeof(float)));

    /* 1. Submuestreo determinista para entrenamiento K-Means++ y PCA */
    const uint32_t sample_size = (nv < 8192U) ? nv : 8192U;
    std::vector<uint32_t> sample_ids(sample_size);
    std::vector<float> sample_vecs(static_cast<size_t>(sample_size) * dim);
    uint64_t rng = 0x2026F1A7ULL;
    uint32_t stride_s = nv / sample_size;
    for (uint32_t i = 0; i < sample_size; ++i) {
        sample_ids[i] = (i * stride_s + static_cast<uint32_t>(splitmix64(&rng) % stride_s)) % nv;
        memcpy(sample_vecs.data() + static_cast<size_t>(i) * dim,
               ds.vectors + static_cast<size_t>(sample_ids[i]) * dim,
               dim * sizeof(float));
    }

    /* Calcular base ortonormal Q^T (64x64) que concentra varianza en las primeras 16 dimensiones */
    compute_pca_ortho_basis(sample_vecs.data(), sample_size, dim, out_idx.ortho_basis);

    /* Inicializacion K-Means++ */
    uint32_t first_id = sample_ids[splitmix64(&rng) % sample_size];
    memcpy(out_idx.centroids, ds.vectors + static_cast<size_t>(first_id) * dim, dim * sizeof(float));

    std::vector<float> min_dist_sq(sample_size, FLT_MAX);
    for (uint32_t c = 1; c < num_partitions; ++c) {
        const float* prev_c = out_idx.centroids + static_cast<size_t>(c - 1) * dim;
        double sum_w = 0.0;
        for (uint32_t s = 0; s < sample_size; ++s) {
            const float* v = sample_vecs.data() + static_cast<size_t>(s) * dim;
            float d2 = l2_sq_avx2_1x_aos(prev_c, v, dim);
            if (d2 < min_dist_sq[s]) min_dist_sq[s] = d2;
            sum_w += static_cast<double>(min_dist_sq[s]);
        }

        double target = (static_cast<double>(splitmix64(&rng) >> 11) / 9007199254740992.0) * sum_w;
        double accum = 0.0;
        uint32_t chosen_s = sample_size - 1;
        for (uint32_t s = 0; s < sample_size; ++s) {
            accum += static_cast<double>(min_dist_sq[s]);
            if (accum >= target) {
                chosen_s = s;
                break;
            }
        }
        memcpy(out_idx.centroids + static_cast<size_t>(c) * dim,
               sample_vecs.data() + static_cast<size_t>(chosen_s) * dim,
               dim * sizeof(float));
    }

    /* Iteraciones de Lloyd */
    std::vector<double> accum_centroids(static_cast<size_t>(num_partitions) * dim, 0.0);
    std::vector<uint32_t> counts(num_partitions, 0);

    for (uint32_t iter = 0; iter < kmeans_iters; ++iter) {
        std::fill(accum_centroids.begin(), accum_centroids.end(), 0.0);
        std::fill(counts.begin(), counts.end(), 0U);

        for (uint32_t s = 0; s < sample_size; ++s) {
            const float* v = sample_vecs.data() + static_cast<size_t>(s) * dim;
            uint32_t best_c = 0;
            float best_d2 = FLT_MAX;
            for (uint32_t c = 0; c < num_partitions; ++c) {
                float d2 = l2_sq_avx2_1x_aos(v, out_idx.centroids + static_cast<size_t>(c) * dim, dim);
                if (d2 < best_d2) {
                    best_d2 = d2;
                    best_c = c;
                }
            }
            counts[best_c]++;
            double* acc = accum_centroids.data() + static_cast<size_t>(best_c) * dim;
            for (uint32_t d = 0; d < dim; ++d) acc[d] += static_cast<double>(v[d]);
        }

        for (uint32_t c = 0; c < num_partitions; ++c) {
            if (counts[c] > 0) {
                float inv = static_cast<float>(1.0 / static_cast<double>(counts[c]));
                float* cent = out_idx.centroids + static_cast<size_t>(c) * dim;
                const double* acc = accum_centroids.data() + static_cast<size_t>(c) * dim;
                for (uint32_t d = 0; d < dim; ++d) cent[d] = static_cast<float>(acc[d]) * inv;
            }
        }
    }

    /* Rotar centroides por Q^T */
    for (uint32_t c = 0; c < num_partitions; ++c) {
        rotate_vector_64(out_idx.ortho_basis,
                         out_idx.centroids + static_cast<size_t>(c) * dim,
                         out_idx.rotated_centroids + static_cast<size_t>(c) * dim,
                         dim);
    }

    auto t_kmeans = std::chrono::high_resolution_clock::now();
    out_idx.kmeans_time_ms = elapsed_ms(t_start, t_kmeans);

    /* 2. Asignacion exacta de los N vectores a las particiones y calculo de radios exactos */
    struct AssignedVec {
        uint32_t id;
        float radius;
    };
    std::vector<std::vector<AssignedVec>> part_buckets(num_partitions);
    for (uint32_t c = 0; c < num_partitions; ++c) {
        part_buckets[c].reserve((nv / num_partitions) * 2);
    }

    for (uint32_t i = 0; i < nv; ++i) {
        const float* v = ds.vectors + static_cast<size_t>(i) * dim;
        uint32_t best_c = 0;
        float best_d2 = FLT_MAX;
        for (uint32_t c = 0; c < num_partitions; ++c) {
            float d2 = l2_sq_avx2_1x_aos(v, out_idx.centroids + static_cast<size_t>(c) * dim, dim);
            if (d2 < best_d2) {
                best_d2 = d2;
                best_c = c;
            }
        }
        part_buckets[best_c].push_back({i, sqrtf(best_d2)});
    }

    /* Ordenar cada particion por radio ascendente */
    for (uint32_t c = 0; c < num_partitions; ++c) {
        auto& bucket = part_buckets[c];
        std::sort(bucket.begin(), bucket.end(), [](const AssignedVec& a, const AssignedVec& b) {
            if (a.radius != b.radius) return a.radius < b.radius;
            return a.id < b.id;
        });
    }

    out_idx.partitions.resize(num_partitions);
    out_idx.part_aos_vectors = static_cast<float*>(aligned_alloc_64(static_cast<size_t>(nv) * dim * sizeof(float)));
    out_idx.part_aos_ids = static_cast<uint32_t*>(aligned_alloc_64(static_cast<size_t>(nv) * sizeof(uint32_t)));
    out_idx.part_aos_categories = static_cast<uint32_t*>(aligned_alloc_64(static_cast<size_t>(nv) * sizeof(uint32_t)));
    out_idx.part_aos_radii = static_cast<float*>(aligned_alloc_64(static_cast<size_t>(nv) * sizeof(float)));
    out_idx.part_aos_tail48_norm = static_cast<float*>(aligned_alloc_64(static_cast<size_t>(nv) * sizeof(float)));
    out_idx.part_aos_tail32_norm = static_cast<float*>(aligned_alloc_64(static_cast<size_t>(nv) * sizeof(float)));

    uint32_t cur_aos_offset = 0;
    for (uint32_t c = 0; c < num_partitions; ++c) {
        const auto& bucket = part_buckets[c];
        uint32_t cnt = static_cast<uint32_t>(bucket.size());

        GeometricPartition& gp = out_idx.partitions[c];
        gp.partition_id = c;
        gp.count = cnt;
        gp.aosoa_blocks = (cnt + 7U) / 8U;
        gp.aos_offset = cur_aos_offset;
        gp.aosoa_offset = 0;
        gp.min_radius = (cnt > 0) ? bucket.front().radius : 0.0f;
        gp.max_radius = (cnt > 0) ? bucket.back().radius : 0.0f;

        for (uint32_t k = 0; k < cnt; ++k) {
            uint32_t orig_id = bucket[k].id;
            uint32_t dst_idx = cur_aos_offset + k;
            const float* src_v = ds.vectors + static_cast<size_t>(orig_id) * dim;
            float* dst_v = out_idx.part_aos_vectors + static_cast<size_t>(dst_idx) * dim;

            /* Guardar el vector rotado por Q^T para concentrar varianza en dims 0..15 */
            rotate_vector_64(out_idx.ortho_basis, src_v, dst_v, dim);

            out_idx.part_aos_ids[dst_idx] = orig_id;
            out_idx.part_aos_categories[dst_idx] = ds.categories[orig_id];
            out_idx.part_aos_radii[dst_idx] = bucket[k].radius;

            double tail32_sq = 0.0;
            for (uint32_t d = 32; d < dim; ++d) {
                tail32_sq += static_cast<double>(dst_v[d]) * dst_v[d];
            }
            double tail16_31_sq = 0.0;
            for (uint32_t d = 16; d < 32; ++d) {
                tail16_31_sq += static_cast<double>(dst_v[d]) * dst_v[d];
            }
            out_idx.part_aos_tail32_norm[dst_idx] = static_cast<float>(std::sqrt(tail32_sq));
            out_idx.part_aos_tail48_norm[dst_idx] = static_cast<float>(std::sqrt(tail32_sq + tail16_31_sq));
        }

        cur_aos_offset += cnt;
    }

    /* Pre-indexar las posiciones elegibles por particion (equivalente exacto a ds.eligible_per_sel) */
    uint32_t num_sels = ds.header.num_selectivities;
    out_idx.eligible_ivf_ids_per_sel.resize(num_sels);
    out_idx.eligible_part_offsets_per_sel.resize(num_sels);

    for (uint32_t s = 0; s < num_sels; ++s) {
        uint32_t target_cat = ds.header.selectivities_deci_percent[s] * 10U;
        auto& el_ids = out_idx.eligible_ivf_ids_per_sel[s];
        auto& part_offs = out_idx.eligible_part_offsets_per_sel[s];
        el_ids.clear();
        el_ids.reserve(nv / 3);
        part_offs.resize(num_partitions + 1);

        for (uint32_t c = 0; c < num_partitions; ++c) {
            part_offs[c] = static_cast<uint32_t>(el_ids.size());
            uint32_t p_start = out_idx.partitions[c].aos_offset;
            uint32_t p_end = p_start + out_idx.partitions[c].count;
            for (uint32_t idx_v = p_start; idx_v < p_end; ++idx_v) {
                if (out_idx.part_aos_categories[idx_v] < target_cat) {
                    el_ids.push_back(idx_v);
                }
            }
        }
        part_offs[num_partitions] = static_cast<uint32_t>(el_ids.size());
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    out_idx.layout_time_ms = elapsed_ms(t_kmeans, t_end);
    out_idx.total_build_time_ms = elapsed_ms(t_start, t_end);
}

} // namespace alg_flat
