/*
 * Copyright (c) 2026 Santiago Robledo.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "../include/dataset_loader.h"
#include <fstream>
#include <cmath>
#include <cstring>

namespace alg_flat {

static inline uint64_t rng_next(uint64_t *s) {
    uint64_t x = *s;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static inline float uniform01(uint64_t *s) {
    return static_cast<float>(rng_next(s) >> 40) / 16777216.0f;
}

static inline float gaussian01(uint64_t *s) {
    float u1 = uniform01(s);
    float u2 = uniform01(s);
    if (u1 < 1.0e-7f) u1 = 1.0e-7f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853071795864769f * u2);
}

static inline void normalize_vec(float *v, uint32_t dim) {
    double n2 = 0.0;
    for (uint32_t d = 0; d < dim; ++d) n2 += static_cast<double>(v[d]) * v[d];
    float inv = static_cast<float>(1.0 / sqrt(n2 > 1.0e-20 ? n2 : 1.0));
    for (uint32_t d = 0; d < dim; ++d) v[d] *= inv;
}

static void populate_eligible_lists(BenchmarkDataset& ds) {
    const uint32_t nv = ds.header.num_vectors;
    const uint32_t num_sels = ds.header.num_selectivities;
    ds.eligible_per_sel.resize(num_sels);
    for (uint32_t s = 0; s < num_sels; ++s) {
        uint32_t target_cat = ds.header.selectivities_deci_percent[s] * 10U;
        auto& list = ds.eligible_per_sel[s];
        list.clear();
        list.reserve(nv / 3);
        for (uint32_t id = 0; id < nv; ++id) {
            if (ds.categories[id] < target_cat) {
                list.push_back(id);
            }
        }
    }
}

bool load_dataset_from_file(const std::string& path, BenchmarkDataset& out_ds) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    f.read(reinterpret_cast<char*>(&out_ds.header), sizeof(DatasetBinaryHeader));
    if (!f || out_ds.header.magic != OOD_MAGIC || out_ds.header.dimension != DEFAULT_DIM) {
        return false;
    }

    uint32_t nv = out_ds.header.num_vectors;
    uint32_t dim = out_ds.header.dimension;
    uint32_t nq = out_ds.header.num_queries;

    aligned_free_64(out_ds.vectors);
    aligned_free_64(out_ds.categories);

    out_ds.vectors = static_cast<float*>(aligned_alloc_64(static_cast<size_t>(nv) * dim * sizeof(float)));
    out_ds.categories = static_cast<uint32_t*>(aligned_alloc_64(static_cast<size_t>(nv) * sizeof(uint32_t)));
    out_ds.queries.resize(nq);

    f.read(reinterpret_cast<char*>(out_ds.vectors), static_cast<size_t>(nv) * dim * sizeof(float));
    f.read(reinterpret_cast<char*>(out_ds.categories), static_cast<size_t>(nv) * sizeof(uint32_t));
    f.read(reinterpret_cast<char*>(out_ds.queries.data()), static_cast<size_t>(nq) * sizeof(QueryRecord));
    if (!f) return false;

    populate_eligible_lists(out_ds);
    return true;
}

bool generate_dataset_deterministic(uint64_t seed,
                                    uint32_t nv,
                                    uint32_t nq,
                                    uint32_t basins,
                                    float sigma,
                                    MixMode mix,
                                    BenchmarkDataset& out_ds) {
    if (nv < 128 || nq == 0 || basins == 0) return false;
    const uint32_t dim = DEFAULT_DIM;
    const float local_query_sigma = 0.05f;

    aligned_free_64(out_ds.vectors);
    aligned_free_64(out_ds.categories);

    out_ds.vectors = static_cast<float*>(aligned_alloc_64(static_cast<size_t>(nv) * dim * sizeof(float)));
    out_ds.categories = static_cast<uint32_t*>(aligned_alloc_64(static_cast<size_t>(nv) * sizeof(uint32_t)));
    out_ds.queries.resize(nq);

    std::vector<float> centroids(static_cast<size_t>(basins) * dim);
    std::vector<uint32_t> basin_nodes(basins, UINT32_MAX);

    uint64_t centroid_rng = seed ^ 0x9988776655443322ULL;
    for (uint32_t c = 0; c < basins; ++c) {
        float *center = centroids.data() + static_cast<size_t>(c) * dim;
        for (uint32_t d = 0; d < dim; ++d) center[d] = 2.0f * uniform01(&centroid_rng) - 1.0f;
        normalize_vec(center, dim);
    }

    uint64_t point_rng = seed ^ 0x3344556677889900ULL;
    for (uint32_t i = 0; i < nv; ++i) {
        uint32_t c = static_cast<uint32_t>(rng_next(&point_rng) % basins);
        const float *center = centroids.data() + static_cast<size_t>(c) * dim;
        if (basin_nodes[c] == UINT32_MAX) basin_nodes[c] = i;
        float *v = out_ds.vectors + static_cast<size_t>(i) * dim;
        for (uint32_t d = 0; d < dim; ++d) v[d] = center[d] + sigma * gaussian01(&point_rng);
        normalize_vec(v, dim);

        uint64_t z = seed + static_cast<uint64_t>(i) * 0x9E3779B97F4A7C15ULL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z ^= z >> 31;
        out_ds.categories[i] = static_cast<uint32_t>(z % 10000ULL);
    }

    uint64_t query_rng = seed ^ 0xFEEDFACECAFEBEEFULL;
    const uint32_t local_count = (mix == MIX_R) ? nq : (mix == MIX_D) ? 0U : (nq / 2U);

    for (uint32_t q = 0; q < nq; ++q) {
        bool is_local = (q < local_count);
        if (mix == MIX_R) is_local = true;
        if (mix == MIX_D) is_local = false;

        QueryRecord *rec = &out_ds.queries[q];
        rec->label = is_local ? REGIME_LOCAL : REGIME_ADVERSARIAL;
        rec->reserved[0] = 0;
        rec->reserved[1] = 0;

        bool concentrated = is_local && ((rng_next(&query_rng) % 100U) < 85U);
        if (!is_local) rec->component = COMP_ADVERSARIAL_UNIFORM;
        else if (concentrated) rec->component = COMP_LOCAL_BASIN;
        else rec->component = COMP_LOCAL_UNIFORM;

        if (concentrated) {
            uint32_t b = static_cast<uint32_t>(rng_next(&query_rng) % basins);
            const float *base = out_ds.vectors + static_cast<size_t>(basin_nodes[b]) * dim;
            for (uint32_t d = 0; d < dim; ++d)
                rec->values[d] = base[d] + local_query_sigma * gaussian01(&query_rng);
        } else {
            for (uint32_t d = 0; d < dim; ++d)
                rec->values[d] = 2.0f * uniform01(&query_rng) - 1.0f;
        }
        normalize_vec(rec->values, dim);
    }

    for (uint32_t i = nq - 1U; i > 0U; --i) {
        uint32_t j = static_cast<uint32_t>(rng_next(&query_rng) % static_cast<uint64_t>(i + 1U));
        QueryRecord tmp = out_ds.queries[i];
        out_ds.queries[i] = out_ds.queries[j];
        out_ds.queries[j] = tmp;
    }

    memset(&out_ds.header, 0, sizeof(DatasetBinaryHeader));
    out_ds.header.magic = OOD_MAGIC;
    out_ds.header.version = OOD_VERSION;
    out_ds.header.seed = seed;
    out_ds.header.num_vectors = nv;
    out_ds.header.dimension = dim;
    out_ds.header.num_queries = nq;
    out_ds.header.num_basins = basins;
    out_ds.header.vector_sigma = sigma;
    out_ds.header.local_query_sigma = local_query_sigma;
    out_ds.header.mix_mode = static_cast<uint32_t>(mix);
    out_ds.header.local_fraction_percent = (mix == MIX_R) ? 100U : (mix == MIX_D) ? 0U : 50U;
    out_ds.header.local_basin_fraction_percent = 85U;
    out_ds.header.num_selectivities = 7U;
    const uint32_t sels[7] = {100, 125, 150, 175, 200, 225, 250};
    for (uint32_t s = 0; s < 7U; ++s) out_ds.header.selectivities_deci_percent[s] = sels[s];

    populate_eligible_lists(out_ds);
    return true;
}

} // namespace alg_flat
