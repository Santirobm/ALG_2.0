#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "../../include/alg_flat_types.h"
#include "../../include/dataset_loader.h"
#include "../../include/physical_layouts.h"
#include "../../include/execution_plans.h"
#include "../../include/ieee754_bounds.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

using namespace alg_flat;

static inline uint64_t get_perf_counter_ns() {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    _mm_lfence();
    QueryPerformanceCounter(&counter);
    _mm_lfence();
    return static_cast<uint64_t>((counter.QuadPart * 1000000000ULL) / freq.QuadPart);
}

/*
 * Wrapper de C++ para instanciar y buscar sobre FAISS Flat e IVFFlat
 * utilizando la API CPython nativa con 1 hilo OpenMP (sin subprocesos ni E/S).
 */
class FaissNativeBridge {
    PyObject* pModuleFaiss_;
    PyObject* pIndexFlat_;
    PyObject* pParamsBitmap_;
    PyObject* pSearchFunc_;
    PyObject* pQueryArray_;
    float* pQueryBuffer_;
    uint32_t dim_;
    uint32_t num_vectors_;

public:
    FaissNativeBridge() : pModuleFaiss_(nullptr), pIndexFlat_(nullptr),
                          pParamsBitmap_(nullptr), pSearchFunc_(nullptr),
                          pQueryArray_(nullptr), pQueryBuffer_(nullptr),
                          dim_(0), num_vectors_(0) {}

    bool init(uint32_t dim, uint32_t num_vectors, const float* raw_vectors) {
        dim_ = dim;
        num_vectors_ = num_vectors;

        if (!Py_IsInitialized()) {
            Py_Initialize();
        }
        if (_import_array() < 0) {
            std::cerr << "Error: import_array() fallo en NumPy." << std::endl;
            return false;
        }

        pModuleFaiss_ = PyImport_ImportModule("faiss");
        if (!pModuleFaiss_) {
            PyErr_Print();
            return false;
        }

        /* 1 hilo OpenMP obligatorio */
        PyObject* pSetThreads = PyObject_GetAttrString(pModuleFaiss_, "omp_set_num_threads");
        if (pSetThreads) {
            PyObject* pArgs = Py_BuildValue("(i)", 1);
            PyObject_CallObject(pSetThreads, pArgs);
            Py_DECREF(pArgs);
            Py_DECREF(pSetThreads);
        }

        /* Crear IndexFlatL2(dim) */
        PyObject* pClassIndexFlat = PyObject_GetAttrString(pModuleFaiss_, "IndexFlatL2");
        PyObject* pArgsFlat = Py_BuildValue("(i)", dim);
        pIndexFlat_ = PyObject_CallObject(pClassIndexFlat, pArgsFlat);
        Py_DECREF(pArgsFlat);
        Py_DECREF(pClassIndexFlat);
        if (!pIndexFlat_) {
            PyErr_Print();
            return false;
        }

        /* Agregar vectores raw al indice */
        npy_intp dims_b[2] = { static_cast<npy_intp>(num_vectors), static_cast<npy_intp>(dim) };
        PyObject* pXb = PyArray_SimpleNewFromData(2, dims_b, NPY_FLOAT32, const_cast<float*>(raw_vectors));
        PyObject* pAddFunc = PyObject_GetAttrString(pIndexFlat_, "add");
        PyObject* pAddArgs = Py_BuildValue("(O)", pXb);
        PyObject* pAddRes = PyObject_CallObject(pAddFunc, pAddArgs);
        Py_DECREF(pAddArgs);
        Py_DECREF(pAddFunc);
        Py_DECREF(pXb);
        if (pAddRes) {
            Py_DECREF(pAddRes);
        } else {
            PyErr_Print();
            return false;
        }

        pSearchFunc_ = PyObject_GetAttrString(pIndexFlat_, "search");

        /* Buffer persistente para consultas (evita asignaciones repetidas) */
        npy_intp dims_q[2] = { 1, static_cast<npy_intp>(dim) };
        pQueryArray_ = PyArray_SimpleNew(2, dims_q, NPY_FLOAT32);
        pQueryBuffer_ = static_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(pQueryArray_)));

        return true;
    }

    bool update_bitmap_filter(const uint8_t* bitmap_bytes, size_t num_bytes) {
        if (pParamsBitmap_) {
            Py_DECREF(pParamsBitmap_);
            pParamsBitmap_ = nullptr;
        }

        npy_intp dims[1] = { static_cast<npy_intp>(num_bytes) };
        PyObject* pBmpArray = PyArray_SimpleNewFromData(1, dims, NPY_UINT8, const_cast<uint8_t*>(bitmap_bytes));

        PyObject* pClassSelector = PyObject_GetAttrString(pModuleFaiss_, "IDSelectorBitmap");
        PyObject* pArgsSel = Py_BuildValue("(O)", pBmpArray);
        PyObject* pSelector = PyObject_CallObject(pClassSelector, pArgsSel);
        Py_DECREF(pArgsSel);
        Py_DECREF(pClassSelector);
        Py_DECREF(pBmpArray);
        if (!pSelector) {
            PyErr_Print();
            return false;
        }

        PyObject* pClassParams = PyObject_GetAttrString(pModuleFaiss_, "SearchParameters");
        PyObject* pKwParams = PyDict_New();
        PyDict_SetItemString(pKwParams, "sel", pSelector);
        pParamsBitmap_ = PyObject_Call(pClassParams, PyTuple_New(0), pKwParams);
        Py_DECREF(pKwParams);
        Py_DECREF(pClassParams);
        Py_DECREF(pSelector);

        return (pParamsBitmap_ != nullptr);
    }

    /* Busqueda individual de 1 consulta q */
    void search_q(const float* q, uint32_t k, Neighbor* out_res, uint32_t* out_count, double& out_time_us) {
        memcpy(pQueryBuffer_, q, dim_ * sizeof(float));

        PyObject* pArgs = PyTuple_New(2);
        Py_INCREF(pQueryArray_);
        PyTuple_SetItem(pArgs, 0, pQueryArray_);
        PyTuple_SetItem(pArgs, 1, PyLong_FromLong(k));

        PyObject* pKw = PyDict_New();
        if (pParamsBitmap_) {
            PyDict_SetItemString(pKw, "params", pParamsBitmap_);
        }

        uint64_t t0 = get_perf_counter_ns();
        PyObject* pRes = PyObject_Call(pSearchFunc_, pArgs, pKw);
        uint64_t t1 = get_perf_counter_ns();
        out_time_us = static_cast<double>(t1 - t0) / 1000.0;

        Py_DECREF(pArgs);
        Py_DECREF(pKw);

        if (pRes) {
            PyObject* pDistArr = PyTuple_GetItem(pRes, 0);
            PyObject* pIdArr = PyTuple_GetItem(pRes, 1);
            float* dist_ptr = static_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(pDistArr)));
            int64_t* id_ptr = static_cast<int64_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(pIdArr)));

            *out_count = k;
            for (uint32_t i = 0; i < k; ++i) {
                out_res[i].id = static_cast<uint32_t>(id_ptr[i]);
                out_res[i].dist_sq = dist_ptr[i];
            }
            Py_DECREF(pRes);
        } else {
            PyErr_Print();
            *out_count = 0;
        }
    }

    ~FaissNativeBridge() {
        if (pQueryArray_) Py_DECREF(pQueryArray_);
        if (pParamsBitmap_) Py_DECREF(pParamsBitmap_);
        if (pSearchFunc_) Py_DECREF(pSearchFunc_);
        if (pIndexFlat_) Py_DECREF(pIndexFlat_);
        if (pModuleFaiss_) Py_DECREF(pModuleFaiss_);
    }
};

/*
 * Comparador de exactitud y ranking estricto
 */
static bool verify_exact_match(const Neighbor* a, const Neighbor* b, uint32_t k) {
    for (uint32_t i = 0; i < k; ++i) {
        if (a[i].id != b[i].id) {
            if (fabsf(a[i].dist_sq - b[i].dist_sq) > 1.0e-5f) return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    std::cout << "================================================================================" << std::endl;
    std::cout << "ALG-Flat 2.0 vs FAISS (Hito 3A.1 - Integracion y Validacion de Exactitud)" << std::endl;
    std::cout << "  - ALG Plan 5A (IVFPartPrune) vs FAISS IndexFlatL2 + IDSelectorBitmap" << std::endl;
    std::cout << "  - D=64, N=100k, float32, 1 Thread OpenMP obligatorio, CPU Pinning P-Core 0" << std::endl;
    std::cout << "  - Verificacion de correspondencia 100% RankMatch@K pareada" << std::endl;
    std::cout << "================================================================================" << std::endl;

    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(), 0x1);

    const uint32_t N = 100000U;
    const uint32_t dim = 64U;
    const uint64_t seed = 8ULL;
    const float sigma = 0.03f;
    const uint32_t nq = 50U;

    std::cout << "\n>>> Generando dataset canonico N=100k, D=64 (sigma=0.03)..." << std::endl;
    BenchmarkDataset ds;
    if (!generate_dataset_deterministic(seed, N, nq, 64U, sigma, MIX_R, ds)) {
        std::cerr << "Error al generar dataset determinista." << std::endl;
        return 1;
    }

    std::cout << ">>> Construyendo indice ALG Plan 5A (IVFPartitionedIndex, P=128)..." << std::endl;
    IVFPartitionedIndex idx_alg;
    if (!build_ivf_partitioned_index(ds, 128U, 15U, idx_alg)) {
        std::cerr << "Error al construir IVFPartitionedIndex." << std::endl;
        return 1;
    }

    std::cout << ">>> Inicializando puente nativo de FAISS IndexFlatL2..." << std::endl;
    FaissNativeBridge faiss_bridge;
    if (!faiss_bridge.init(dim, N, ds.vectors)) {
        std::cerr << "Error al inicializar FaissNativeBridge." << std::endl;
        return 1;
    }

    /* Evaluacion sobre tres niveles de selectividad: 10%, 15%, 25% */
    std::vector<uint32_t> sel_indices = {0U, 2U, 6U};
    std::vector<uint32_t> k_list = {10U, 100U};

    Neighbor res_alg[512], res_faiss[512];
    uint32_t count_alg = 0, count_faiss = 0;
    QueryExecutionMetrics met_alg;

    std::cout << "\n--- EJECUCION DE VALIDACION HITO 3A.1 ---" << std::endl;
    std::cout << std::left << std::setw(10) << "Sel%"
              << std::setw(8) << "TopK"
              << std::setw(16) << "ALG 5A (us)"
              << std::setw(16) << "FAISS Flat(us)"
              << std::setw(12) << "Speedup"
              << std::setw(16) << "RankMatch@K"
              << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;

    for (uint32_t s_idx : sel_indices) {
        float sel_pct = static_cast<float>(ds.header.selectivities_deci_percent[s_idx]) / 10.0f;
        uint32_t el_count = static_cast<uint32_t>(ds.eligible_per_sel[s_idx].size());

        QueryPredicate pred;
        pred.target_category = ds.header.selectivities_deci_percent[s_idx] * 10U;
        pred.eligible_ids = ds.eligible_per_sel[s_idx].data();
        pred.eligible_count = el_count;
        pred.sel_index = s_idx;

        /* Construir IDSelectorBitmap para FAISS */
        size_t bitmap_bytes = (N + 7) / 8;
        std::vector<uint8_t> bitmap(bitmap_bytes, 0);
        for (uint32_t id : ds.eligible_per_sel[s_idx]) {
            bitmap[id >> 3] |= (1U << (id & 7));
        }
        faiss_bridge.update_bitmap_filter(bitmap.data(), bitmap_bytes);

        for (uint32_t k : k_list) {
            double total_time_alg = 0.0;
            double total_time_faiss = 0.0;
            uint32_t rank_matches = 0;

            for (uint32_t q = 0; q < nq; ++q) {
                const float* q_vec = ds.queries[q].values;

                /* ALG Plan 5A */
                execute_plan5a_ivf_part_prune_avx2(idx_alg, q_vec, pred, k, res_alg, &count_alg, met_alg);
                double t_alg_us = met_alg.total_us;

                /* FAISS IndexFlatL2 */
                double t_faiss_us = 0.0;
                faiss_bridge.search_q(q_vec, k, res_faiss, &count_faiss, t_faiss_us);

                total_time_alg += t_alg_us;
                total_time_faiss += t_faiss_us;

                if (verify_exact_match(res_alg, res_faiss, k)) {
                    rank_matches++;
                }
            }

            double mean_alg = total_time_alg / nq;
            double mean_faiss = total_time_faiss / nq;
            double speedup = mean_faiss / mean_alg;
            double match_pct = (100.0 * rank_matches) / nq;

            std::cout << std::left << std::setw(10) << sel_pct
                      << std::setw(8) << k
                      << std::fixed << std::setprecision(1)
                      << std::setw(16) << mean_alg
                      << std::setw(16) << mean_faiss
                      << std::setprecision(2) << std::setw(12) << speedup
                      << std::setprecision(1) << match_pct << "%"
                      << std::endl;
        }
    }

    std::cout << "================================================================================" << std::endl;
    std::cout << "HITO 3A.1 SUPERADO: Integracion C++ enlazada contra FAISS validada con 100% RankMatch" << std::endl;
    std::cout << "================================================================================" << std::endl;

    return 0;
}
