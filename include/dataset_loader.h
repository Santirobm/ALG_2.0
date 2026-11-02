#ifndef ALG_FLAT_DATASET_LOADER_H
#define ALG_FLAT_DATASET_LOADER_H

#include "alg_flat_types.h"
#include <vector>
#include <string>

namespace alg_flat {

static constexpr uint32_t OOD_MAGIC = 0x39444F41U; /* "AOD9" */
static constexpr uint32_t OOD_VERSION = 1U;
static constexpr uint32_t DEFAULT_DIM = 64U;

enum QueryRegimeLabel : uint8_t {
    REGIME_LOCAL = 0,
    REGIME_ADVERSARIAL = 1
};

enum QueryComponentType : uint8_t {
    COMP_LOCAL_BASIN = 0,
    COMP_LOCAL_UNIFORM = 1,
    COMP_ADVERSARIAL_UNIFORM = 2
};

enum MixMode : uint32_t {
    MIX_R = 0,
    MIX_D = 1,
    MIX_50_50 = 2
};

#pragma pack(push, 1)
struct DatasetBinaryHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t seed;
    uint32_t num_vectors;
    uint32_t dimension;
    uint32_t num_queries;
    uint32_t num_basins;
    float vector_sigma;
    float local_query_sigma;
    uint32_t mix_mode;
    uint32_t local_fraction_percent;
    uint32_t local_basin_fraction_percent;
    uint32_t num_selectivities;
    uint32_t selectivities_deci_percent[7];
};

struct QueryRecord {
    float values[DEFAULT_DIM];
    uint8_t label;
    uint8_t component;
    uint8_t reserved[2];
};
#pragma pack(pop)

struct BenchmarkDataset {
    DatasetBinaryHeader header;
    float* vectors;                  /* Aligned 64B: num_vectors * dim */
    uint32_t* categories;            /* Aligned 64B: num_vectors */
    std::vector<QueryRecord> queries;
    std::vector<std::vector<uint32_t>> eligible_per_sel;

    BenchmarkDataset() : vectors(nullptr), categories(nullptr) {
        memset(&header, 0, sizeof(header));
    }

    ~BenchmarkDataset() {
        aligned_free_64(vectors);
        aligned_free_64(categories);
    }

    BenchmarkDataset(const BenchmarkDataset&) = delete;
    BenchmarkDataset& operator=(const BenchmarkDataset&) = delete;
};

/* Carga un archivo .bin compatible con Hito 8/9 */
bool load_dataset_from_file(const std::string& path, BenchmarkDataset& out_ds);

/* Genera en memoria exactamente el mismo dataset determinista de Hito 9 */
bool generate_dataset_deterministic(uint64_t seed,
                                    uint32_t nv,
                                    uint32_t nq,
                                    uint32_t basins,
                                    float sigma,
                                    MixMode mix,
                                    BenchmarkDataset& out_ds);

} // namespace alg_flat

#endif // ALG_FLAT_DATASET_LOADER_H
