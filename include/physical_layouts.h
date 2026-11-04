/*
 * Copyright (c) 2026 Santiago Robledo.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef ALG_FLAT_PHYSICAL_LAYOUTS_H
#define ALG_FLAT_PHYSICAL_LAYOUTS_H

#include "alg_flat_types.h"
#include "dataset_loader.h"
#include <vector>

namespace alg_flat {

/*
 * Layout 1: Array of Structures (AoS) tradicional
 * Vector i almacenado contiguamente en [i * 64 .. i * 64 + 63]
 */
struct FlatAoSIndex {
    uint32_t num_vectors;
    uint32_t dim;
    const float* vectors;       /* Referencia alineada al corpus base */
    const uint32_t* categories;
    double build_time_ms;
};

/*
 * Layout 2: Structure of Arrays (SoA) global
 * Dimension d para todos los vectores almacenada en [d * stride + i]
 */
struct FlatSoAIndex {
    uint32_t num_vectors;
    uint32_t num_blocks;        /* ceil(num_vectors / 8) */
    uint32_t stride;            /* num_blocks * 8 */
    uint32_t dim;
    const float* orig_vectors;  /* Referencia al corpus original para distancia exacta canonica */
    float* soa_data;            /* Aligned 64B: dim * stride floats */
    uint32_t* categories;       /* Aligned 64B: stride uint32_t (padded con UINT32_MAX) */
    uint32_t* ids;              /* Aligned 64B: stride uint32_t */
    double build_time_ms;

    FlatSoAIndex() : num_vectors(0), num_blocks(0), stride(0), dim(0),
                     orig_vectors(nullptr), soa_data(nullptr), categories(nullptr), ids(nullptr), build_time_ms(0.0) {}
    ~FlatSoAIndex() {
        aligned_free_64(soa_data);
        aligned_free_64(categories);
        aligned_free_64(ids);
    }
};

/*
 * Layout 3: Array of Structures of Arrays (AoSoA - Bloques transpuestos de 8 vectores x 64 dims)
 * Cada bloque ocupa exactamente 2048 bytes (32 cache lines contiguas).
 */
struct FlatAoSoAIndex {
    uint32_t num_vectors;
    uint32_t num_blocks;        /* ceil(num_vectors / 8) */
    uint32_t dim;
    const float* orig_vectors;  /* Referencia al corpus original para distancia exacta canonica */
    float* block_data;          /* Aligned 64B: num_blocks * 64 * 8 floats */
    uint32_t* categories;       /* Aligned 64B: num_blocks * 8 uint32_t */
    uint32_t* ids;              /* Aligned 64B: num_blocks * 8 uint32_t */
    double build_time_ms;

    FlatAoSoAIndex() : num_vectors(0), num_blocks(0), dim(0),
                       orig_vectors(nullptr), block_data(nullptr), categories(nullptr), ids(nullptr), build_time_ms(0.0) {}
    ~FlatAoSoAIndex() {
        aligned_free_64(block_data);
        aligned_free_64(categories);
        aligned_free_64(ids);
    }
};

/*
 * Metadatos de una particion geometrica individual en ALG-Flat IVF
 */
struct GeometricPartition {
    uint32_t partition_id;
    uint32_t count;             /* Numero de vectores en esta particion */
    uint32_t aosoa_blocks;      /* ceil(count / 8) */
    uint32_t aos_offset;        /* Offset en el buffer contiguo AoS de particiones */
    uint32_t aosoa_offset;      /* Offset en bloques (x 512 floats) en el buffer AoSoA */
    float min_radius;           /* Cota inferior conservadora (redondeada hacia 0) de min_{x in P} ||x - c||_2 */
    float max_radius;           /* Cota superior conservadora (redondeada hacia +inf) de max_{x in P} ||x - c||_2 */
};

/*
 * Layout 4: Indice Fisicamente Particionado (IVF-Flat Exacto con Rotacion Ortogonal PCA por Particion
 * y Cotas Conservadoras Certificadas en IEEE-754 float32)
 */
struct IVFPartitionedIndex {
    uint32_t num_vectors;
    uint32_t dim;
    uint32_t num_partitions;

    float* ortho_basis;                 /* Aligned 64B: 64 x 64 matriz ortogonal Q^T (filas = ejes principales) */
    float ortho_defect;                 /* Norma de Frobenius exacta ||Q * Q^T - I||_F medida en double */
    float* global_mean;                 /* Aligned 64B: dim floats (media global mu para centrado libre de cancelacion) */
    float* centroids;                   /* Aligned 64B: num_partitions * dim (centroides c_p en espacio original) */
    float* rotated_centroids;           /* Aligned 64B: num_partitions * dim (Q^T * (c_p - mu) en espacio rotado) */
    float* centroid_mean_norms;         /* Aligned 64B: num_partitions (||c_p - mu||_2 para cota IEEE-754) */
    std::vector<GeometricPartition> partitions;

    /* Almacenamiento fisico contiguo AoS ordenado por (particion, radio ascendente) */
    float* part_orig_vectors;           /* Aligned 64B: num_vectors * dim (vectores x en espacio original float32) */
    float* part_aos_vectors;            /* Aligned 64B: num_vectors * dim (residuos rotados Q^T * (x - c_p)) */
    uint32_t* part_aos_ids;             /* Aligned 64B: num_vectors (ID original 0..N-1) */
    uint32_t* part_aos_categories;      /* Aligned 64B: num_vectors */
    float* part_aos_radii;              /* Aligned 64B: num_vectors (||x - c_p||_2 en espacio original) */
    float* part_aos_tail48_norm;        /* Aligned 64B: num_vectors (||Q^T(x - c_p)[16..63]||_2 respecto del centroide c_p) */
    float* part_aos_tail32_norm;        /* Aligned 64B: num_vectors (||Q^T(x - c_p)[32..63]||_2 respecto del centroide c_p) */

    /* Listas de posiciones elegibles pre-indexadas por selectividad y particion (mismo contrato que eligible_ids) */
    std::vector<std::vector<uint32_t>> eligible_ivf_ids_per_sel;
    std::vector<std::vector<uint32_t>> eligible_part_offsets_per_sel; /* [sel][part_id] -> offset en eligible_ivf_ids */

    double kmeans_time_ms;
    double layout_time_ms;
    double total_build_time_ms;
    bool certified_Contract_valid;      /* true si MXCSR y todo el corpus cumplieron el contrato IEEE-754 en build-time */

    IVFPartitionedIndex()
        : num_vectors(0), dim(0), num_partitions(0),
          ortho_basis(nullptr), ortho_defect(0.0f), global_mean(nullptr),
          centroids(nullptr), rotated_centroids(nullptr), centroid_mean_norms(nullptr),
          part_orig_vectors(nullptr), part_aos_vectors(nullptr),
          part_aos_ids(nullptr), part_aos_categories(nullptr),
          part_aos_radii(nullptr), part_aos_tail48_norm(nullptr), part_aos_tail32_norm(nullptr),
          kmeans_time_ms(0.0), layout_time_ms(0.0), total_build_time_ms(0.0),
          certified_Contract_valid(false) {}

    ~IVFPartitionedIndex() {
        aligned_free_64(ortho_basis);
        aligned_free_64(global_mean);
        aligned_free_64(centroids);
        aligned_free_64(rotated_centroids);
        aligned_free_64(centroid_mean_norms);
        aligned_free_64(part_orig_vectors);
        aligned_free_64(part_aos_vectors);
        aligned_free_64(part_aos_ids);
        aligned_free_64(part_aos_categories);
        aligned_free_64(part_aos_radii);
        aligned_free_64(part_aos_tail48_norm);
        aligned_free_64(part_aos_tail32_norm);
    }
};

void build_flat_aos_index(const BenchmarkDataset& ds, FlatAoSIndex& out_idx);
void build_flat_soa_index(const BenchmarkDataset& ds, FlatSoAIndex& out_idx);
void build_flat_aosoa_index(const BenchmarkDataset& ds, FlatAoSoAIndex& out_idx);
/*
 * Construye el indice particionado certificado IVF.
 * Si el corpus contiene coordenadas no finitas o fuera de [-CERTIFIED_MAX_COORD_ABS, CERTIFIED_MAX_COORD_ABS],
 * si dim != 64, o si el estado MXCSR de construccion es invalido, RECHAZA la construccion del indice
 * (`out_idx.certified_Contract_valid = false`, `out_idx.num_vectors = 0`) y devuelve `false`.
 */
bool build_ivf_partitioned_index(const BenchmarkDataset& ds,
                                 uint32_t num_partitions,
                                 uint32_t kmeans_iters,
                                 IVFPartitionedIndex& out_idx);

} // namespace alg_flat

#endif // ALG_FLAT_PHYSICAL_LAYOUTS_H
