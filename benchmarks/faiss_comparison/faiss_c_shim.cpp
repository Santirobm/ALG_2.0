#include "faiss_c_shim.h"

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <iostream>
#include <vector>
#include <cstring>

/* Enlace directo a los simbolos exportados de faiss.dll (MSVC ABI) */
extern "C" {
    void faiss_IndexFlat_search(
        const void* this_ptr,
        int64_t n,
        const float* x,
        int64_t k,
        float* distances,
        int64_t* labels,
        const void* params) asm("?search@IndexFlat@faiss@@UEBAX_JPEBM0PEAMPEA_JPEBUSearchParameters@2@@Z");

    void faiss_IndexFlat_compute_distance_subset(
        const void* this_ptr,
        int64_t n,
        const float* x,
        int64_t k,
        float* distances,
        const int64_t* labels) asm("?compute_distance_subset@IndexFlat@faiss@@QEBAX_JPEBM0PEAMPEB_J@Z");

    void faiss_IndexIVF_search(
        const void* this_ptr,
        int64_t n,
        const float* x,
        int64_t k,
        float* distances,
        int64_t* labels,
        const void* params) asm("?search@IndexIVF@faiss@@UEBAX_JPEBM0PEAMPEA_JPEBUSearchParameters@2@@Z");
}

static inline void* extract_swig_ptr(PyObject* py_obj) {
    if (!py_obj) return nullptr;
    PyObject* pThis = PyObject_GetAttrString(py_obj, "this");
    if (!pThis) return nullptr;
    PyObject* pLong = PyNumber_Long(pThis);
    Py_DECREF(pThis);
    if (!pLong) return nullptr;
    void* ptr = reinterpret_cast<void*>(PyLong_AsLongLong(pLong));
    Py_DECREF(pLong);
    return ptr;
}

static void ensure_python_faiss_initialized() {
    if (!Py_IsInitialized()) {
        Py_Initialize();
    }
    if (_import_array() < 0) {
        std::cerr << "[faiss_c_shim] Error: _import_array() fallo en NumPy." << std::endl;
    }
    PyObject* pFaiss = PyImport_ImportModule("faiss");
    if (pFaiss) {
        PyObject* pSetThreads = PyObject_GetAttrString(pFaiss, "omp_set_num_threads");
        if (pSetThreads) {
            PyObject* pArgs = Py_BuildValue("(i)", 1);
            PyObject_CallObject(pSetThreads, pArgs);
            Py_DECREF(pArgs);
            Py_DECREF(pSetThreads);
        }
        Py_DECREF(pFaiss);
    }
}

struct FaissFlatContext {
    PyObject* py_index;
    void* raw_index;
    uint32_t dim;
    uint32_t ntotal;
};

struct FaissIVFContext {
    PyObject* py_index;
    PyObject* py_quantizer;
    void* raw_index;
    uint32_t dim;
    uint32_t nlist;
    uint32_t ntotal;
};

struct FaissSelectorContext {
    PyObject* py_selector;
    void* raw_selector;
};

struct FaissParamsContext {
    PyObject* py_params;
    void* raw_params;
    bool is_ivf;
};

FaissFlatHandle faiss_shim_create_flat_l2(uint32_t dim) {
    ensure_python_faiss_initialized();
    PyObject* pFaiss = PyImport_ImportModule("faiss");
    if (!pFaiss) return nullptr;

    PyObject* pClassFlat = PyObject_GetAttrString(pFaiss, "IndexFlatL2");
    PyObject* pArgs = Py_BuildValue("(i)", dim);
    PyObject* pIndex = PyObject_CallObject(pClassFlat, pArgs);
    Py_DECREF(pArgs);
    Py_DECREF(pClassFlat);
    Py_DECREF(pFaiss);

    if (!pIndex) return nullptr;

    void* raw = extract_swig_ptr(pIndex);
    if (!raw) {
        Py_DECREF(pIndex);
        return nullptr;
    }

    FaissFlatContext* ctx = new FaissFlatContext();
    ctx->py_index = pIndex;
    ctx->raw_index = raw;
    ctx->dim = dim;
    ctx->ntotal = 0;
    return reinterpret_cast<FaissFlatHandle>(ctx);
}

void faiss_shim_free_flat(FaissFlatHandle handle) {
    if (!handle) return;
    FaissFlatContext* ctx = reinterpret_cast<FaissFlatContext*>(handle);
    if (ctx->py_index) {
        Py_DECREF(ctx->py_index);
    }
    delete ctx;
}

bool faiss_shim_flat_add(FaissFlatHandle handle, uint32_t n, const float* x) {
    if (!handle || !x || n == 0) return false;
    FaissFlatContext* ctx = reinterpret_cast<FaissFlatContext*>(handle);

    npy_intp dims[2] = { static_cast<npy_intp>(n), static_cast<npy_intp>(ctx->dim) };
    PyObject* pXb = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, const_cast<float*>(x));
    if (!pXb) return false;

    PyObject* pAddRes = PyObject_CallMethod(ctx->py_index, "add", "O", pXb);
    Py_DECREF(pXb);

    if (!pAddRes) {
        PyErr_Print();
        return false;
    }
    Py_DECREF(pAddRes);
    ctx->ntotal += n;
    return true;
}

FaissIVFHandle faiss_shim_create_ivf_flat(uint32_t dim, uint32_t nlist) {
    ensure_python_faiss_initialized();
    PyObject* pFaiss = PyImport_ImportModule("faiss");
    if (!pFaiss) return nullptr;

    PyObject* pQuantizer = PyObject_CallMethod(pFaiss, "IndexFlatL2", "i", dim);
    if (!pQuantizer) {
        Py_DECREF(pFaiss);
        return nullptr;
    }

    // METRIC_L2 = 0
    PyObject* pIndexIVF = PyObject_CallMethod(pFaiss, "IndexIVFFlat", "Oiii", pQuantizer, dim, nlist, 0);
    Py_DECREF(pFaiss);
    if (!pIndexIVF) {
        Py_DECREF(pQuantizer);
        return nullptr;
    }

    void* raw = extract_swig_ptr(pIndexIVF);
    if (!raw) {
        Py_DECREF(pIndexIVF);
        Py_DECREF(pQuantizer);
        return nullptr;
    }

    FaissIVFContext* ctx = new FaissIVFContext();
    ctx->py_index = pIndexIVF;
    ctx->py_quantizer = pQuantizer;
    ctx->raw_index = raw;
    ctx->dim = dim;
    ctx->nlist = nlist;
    ctx->ntotal = 0;
    return reinterpret_cast<FaissIVFHandle>(ctx);
}

void faiss_shim_free_ivf(FaissIVFHandle handle) {
    if (!handle) return;
    FaissIVFContext* ctx = reinterpret_cast<FaissIVFContext*>(handle);
    if (ctx->py_index) Py_DECREF(ctx->py_index);
    if (ctx->py_quantizer) Py_DECREF(ctx->py_quantizer);
    delete ctx;
}

bool faiss_shim_ivf_train_and_add(FaissIVFHandle handle, uint32_t n, const float* x) {
    if (!handle || !x || n == 0) return false;
    FaissIVFContext* ctx = reinterpret_cast<FaissIVFContext*>(handle);

    npy_intp dims[2] = { static_cast<npy_intp>(n), static_cast<npy_intp>(ctx->dim) };
    PyObject* pXb = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, const_cast<float*>(x));
    if (!pXb) return false;

    PyObject* pTrainRes = PyObject_CallMethod(ctx->py_index, "train", "O", pXb);
    if (!pTrainRes) {
        Py_DECREF(pXb);
        PyErr_Print();
        return false;
    }
    Py_DECREF(pTrainRes);

    PyObject* pAddRes = PyObject_CallMethod(ctx->py_index, "add", "O", pXb);
    Py_DECREF(pXb);
    if (!pAddRes) {
        PyErr_Print();
        return false;
    }
    Py_DECREF(pAddRes);
    ctx->ntotal += n;
    return true;
}

FaissSelectorHandle faiss_shim_create_bitmap_selector(uint32_t n_bits, const uint8_t* bitmap) {
    ensure_python_faiss_initialized();
    PyObject* pFaiss = PyImport_ImportModule("faiss");
    if (!pFaiss) return nullptr;

    size_t num_bytes = (n_bits + 7) / 8;
    npy_intp dims[1] = { static_cast<npy_intp>(num_bytes) };
    PyObject* pBmpArray = PyArray_SimpleNewFromData(1, dims, NPY_UINT8, const_cast<uint8_t*>(bitmap));
    if (!pBmpArray) {
        Py_DECREF(pFaiss);
        return nullptr;
    }

    PyObject* pClassSelector = PyObject_GetAttrString(pFaiss, "IDSelectorBitmap");
    PyObject* pArgsSel = Py_BuildValue("(O)", pBmpArray);
    PyObject* pSelector = PyObject_CallObject(pClassSelector, pArgsSel);
    Py_DECREF(pArgsSel);
    Py_DECREF(pClassSelector);
    Py_DECREF(pBmpArray);
    Py_DECREF(pFaiss);

    if (!pSelector) {
        PyErr_Print();
        return nullptr;
    }

    void* raw = extract_swig_ptr(pSelector);
    if (!raw) {
        Py_DECREF(pSelector);
        return nullptr;
    }

    FaissSelectorContext* ctx = new FaissSelectorContext();
    ctx->py_selector = pSelector;
    ctx->raw_selector = raw;
    return reinterpret_cast<FaissSelectorHandle>(ctx);
}

void faiss_shim_free_selector(FaissSelectorHandle handle) {
    if (!handle) return;
    FaissSelectorContext* ctx = reinterpret_cast<FaissSelectorContext*>(handle);
    if (ctx->py_selector) Py_DECREF(ctx->py_selector);
    delete ctx;
}

FaissParamsHandle faiss_shim_create_flat_params(FaissSelectorHandle sel) {
    ensure_python_faiss_initialized();
    PyObject* pFaiss = PyImport_ImportModule("faiss");
    if (!pFaiss) return nullptr;

    PyObject* pParams = PyObject_CallMethod(pFaiss, "SearchParameters", NULL);
    Py_DECREF(pFaiss);
    if (!pParams) return nullptr;

    void* raw = extract_swig_ptr(pParams);
    if (!raw) {
        Py_DECREF(pParams);
        return nullptr;
    }

    FaissParamsContext* ctx = new FaissParamsContext();
    ctx->py_params = pParams;
    ctx->raw_params = raw;
    ctx->is_ivf = false;

    if (sel) {
        faiss_shim_params_set_selector(reinterpret_cast<FaissParamsHandle>(ctx), sel);
    }
    return reinterpret_cast<FaissParamsHandle>(ctx);
}

FaissParamsHandle faiss_shim_create_ivf_params(FaissSelectorHandle sel, uint32_t nprobe) {
    ensure_python_faiss_initialized();
    PyObject* pFaiss = PyImport_ImportModule("faiss");
    if (!pFaiss) return nullptr;

    PyObject* pParams = PyObject_CallMethod(pFaiss, "SearchParametersIVF", NULL);
    Py_DECREF(pFaiss);
    if (!pParams) return nullptr;

    void* raw = extract_swig_ptr(pParams);
    if (!raw) {
        Py_DECREF(pParams);
        return nullptr;
    }

    FaissParamsContext* ctx = new FaissParamsContext();
    ctx->py_params = pParams;
    ctx->raw_params = raw;
    ctx->is_ivf = true;

    faiss_shim_params_set_nprobe(reinterpret_cast<FaissParamsHandle>(ctx), nprobe);
    if (sel) {
        faiss_shim_params_set_selector(reinterpret_cast<FaissParamsHandle>(ctx), sel);
    }
    return reinterpret_cast<FaissParamsHandle>(ctx);
}

void faiss_shim_params_set_selector(FaissParamsHandle params, FaissSelectorHandle sel) {
    if (!params) return;
    FaissParamsContext* pctx = reinterpret_cast<FaissParamsContext*>(params);
    void* raw_sel = sel ? reinterpret_cast<FaissSelectorContext*>(sel)->raw_selector : nullptr;
    /* Offset 8 es IDSelector* sel */
    *reinterpret_cast<void**>(static_cast<char*>(pctx->raw_params) + 8) = raw_sel;
}

void faiss_shim_params_set_nprobe(FaissParamsHandle params, uint32_t nprobe) {
    if (!params) return;
    FaissParamsContext* pctx = reinterpret_cast<FaissParamsContext*>(params);
    if (pctx->is_ivf) {
        /* Offset 16 es size_t nprobe */
        *reinterpret_cast<uint64_t*>(static_cast<char*>(pctx->raw_params) + 16) = nprobe;
    }
}

void faiss_shim_free_params(FaissParamsHandle handle) {
    if (!handle) return;
    FaissParamsContext* ctx = reinterpret_cast<FaissParamsContext*>(handle);
    if (ctx->py_params) Py_DECREF(ctx->py_params);
    delete ctx;
}

/* Camino critico de medicion */
int faiss_shim_search_flat(
    FaissFlatHandle handle,
    uint32_t n_queries,
    const float* queries,
    uint32_t k,
    float* out_distances,
    int64_t* out_labels,
    FaissParamsHandle params) {

    if (!handle || !queries || !out_distances || !out_labels) return -1;
    FaissFlatContext* ctx = reinterpret_cast<FaissFlatContext*>(handle);
    void* raw_p = params ? reinterpret_cast<FaissParamsContext*>(params)->raw_params : nullptr;

    faiss_IndexFlat_search(ctx->raw_index, n_queries, queries, k, out_distances, out_labels, raw_p);
    return 0;
}

int faiss_shim_search_ivf(
    FaissIVFHandle handle,
    uint32_t n_queries,
    const float* queries,
    uint32_t k,
    float* out_distances,
    int64_t* out_labels,
    FaissParamsHandle params) {

    if (!handle || !queries || !out_distances || !out_labels) return -1;
    FaissIVFContext* ctx = reinterpret_cast<FaissIVFContext*>(handle);
    void* raw_p = params ? reinterpret_cast<FaissParamsContext*>(params)->raw_params : nullptr;

    faiss_IndexIVF_search(ctx->raw_index, n_queries, queries, k, out_distances, out_labels, raw_p);
    return 0;
}

void faiss_shim_compute_distance_subset(
    FaissFlatHandle handle,
    uint32_t n_queries,
    const float* queries,
    uint32_t k_subset,
    float* out_distances,
    const int64_t* subset_labels) {

    if (!handle || !queries || !out_distances || !subset_labels || k_subset == 0) return;
    FaissFlatContext* ctx = reinterpret_cast<FaissFlatContext*>(handle);
    faiss_IndexFlat_compute_distance_subset(
        ctx->raw_index,
        static_cast<int64_t>(n_queries),
        queries,
        static_cast<int64_t>(k_subset),
        out_distances,
        subset_labels);
}
