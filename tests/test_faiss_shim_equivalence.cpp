#include "../benchmarks/faiss_comparison/faiss_c_shim.h"
#include "../include/alg_flat_types.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    std::cout << "================================================================================" << std::endl;
    std::cout << "TEST DE EQUIVALENCIA Y CONTRATO DEL SHIM C++ PARA FAISS (Hito 3.2)" << std::endl;
    std::cout << "================================================================================" << std::endl;

    const uint32_t dim = 64;
    const uint32_t num_vectors = 5000;
    std::vector<float> xb(num_vectors * dim);
    for (size_t i = 0; i < xb.size(); ++i) {
        xb[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    std::cout << "1. Creando IndexFlatL2 a traves del Shim C..." << std::endl;
    FaissFlatHandle flat = faiss_shim_create_flat_l2(dim);
    assert(flat != nullptr);

    bool add_ok = faiss_shim_flat_add(flat, num_vectors, xb.data());
    assert(add_ok);
    std::cout << "   - IndexFlatL2 creado y cargado con " << num_vectors << " vectores." << std::endl;

    std::cout << "2. Creando IndexIVFFlat a traves del Shim C (nlist=64)..." << std::endl;
    FaissIVFHandle ivf = faiss_shim_create_ivf_flat(dim, 64);
    assert(ivf != nullptr);

    bool train_add_ok = faiss_shim_ivf_train_and_add(ivf, num_vectors, xb.data());
    assert(train_add_ok);
    std::cout << "   - IndexIVFFlat entrenado y cargado exitosamente." << std::endl;

    std::cout << "3. Creando Bitmap Selector (primeros 500 vectores elegibles, 10%)..." << std::endl;
    size_t bmp_bytes = (num_vectors + 7) / 8;
    std::vector<uint8_t> bmp(bmp_bytes, 0);
    for (uint32_t i = 0; i < 500; ++i) {
        bmp[i >> 3] |= (1U << (i & 7));
    }
    FaissSelectorHandle sel = faiss_shim_create_bitmap_selector(num_vectors, bmp.data());
    assert(sel != nullptr);

    FaissParamsHandle params_flat = faiss_shim_create_flat_params(sel);
    assert(params_flat != nullptr);

    FaissParamsHandle params_ivf = faiss_shim_create_ivf_params(sel, 16);
    assert(params_ivf != nullptr);
    std::cout << "   - Selector y parametros instanciados correctamente." << std::endl;

    std::cout << "4. Ejecutando busquedas nativas a traves de la interfaz C tipada..." << std::endl;
    std::vector<float> q(dim, 0.5f);
    const uint32_t k = 10;
    std::vector<float> dist_flat(k);
    std::vector<int64_t> labels_flat(k);
    std::vector<float> dist_ivf(k);
    std::vector<int64_t> labels_ivf(k);

    int ret_flat = faiss_shim_search_flat(flat, 1, q.data(), k, dist_flat.data(), labels_flat.data(), params_flat);
    assert(ret_flat == 0);

    int ret_ivf = faiss_shim_search_ivf(ivf, 1, q.data(), k, dist_ivf.data(), labels_ivf.data(), params_ivf);
    assert(ret_ivf == 0);

    std::cout << "   - Flat Top-1: ID=" << labels_flat[0] << ", Dist=" << dist_flat[0] << std::endl;
    std::cout << "   - IVF  Top-1: ID=" << labels_ivf[0] << ", Dist=" << dist_ivf[0] << std::endl;

    // Verificar que los IDs retornados respetan estrictamente el filtro de bits (< 500)
    for (uint32_t i = 0; i < k; ++i) {
        assert(labels_flat[i] >= 0 && labels_flat[i] < 500);
        assert(labels_ivf[i] >= 0 && labels_ivf[i] < 500);
    }
    std::cout << "   - Todos los vecinos cumplen estrictamente el filtro de 10% (IDs en [0..499])." << std::endl;

    std::cout << "5. Liberando recursos a traves del Shim C..." << std::endl;
    faiss_shim_free_params(params_ivf);
    faiss_shim_free_params(params_flat);
    faiss_shim_free_selector(sel);
    faiss_shim_free_ivf(ivf);
    faiss_shim_free_flat(flat);

    std::cout << "\n>>> TEST DE EQUIVALENCIA Y CONTRATO DEL SHIM: APROBADO EXITOSAMENTE <<<\n" << std::endl;
    return 0;
}
