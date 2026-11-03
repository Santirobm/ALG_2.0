#pragma once

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "../../include/alg_flat_types.h"
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace alg_flat {

static inline uint64_t get_highres_time_ns() {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    _mm_lfence();
    QueryPerformanceCounter(&counter);
    _mm_lfence();
    return static_cast<uint64_t>((counter.QuadPart * 1000000000ULL) / freq.QuadPart);
}

/*
 * Enlace directo C++ contra los simbolos exportados de faiss.dll (MSVC ABI).
 * Se ejecutan sin intermediacion de CPython, sin asignacion de PyObjects,
 * sin construccion de tuplas/diccionarios ni conversion de NumPy dentro del cronometro.
 */
extern "C" {
    void faiss_IndexFlat_search(
        const void* this_ptr,
        int64_t n,
        const float* x,
        int64_t k,
        float* distances,
        int64_t* labels,
        const void* params) asm("?search@IndexFlat@faiss@@UEBAX_JPEBM0PEAMPEA_JPEBUSearchParameters@2@@Z");

    void faiss_IndexIVF_search(
        const void* this_ptr,
        int64_t n,
        const float* x,
        int64_t k,
        float* distances,
        int64_t* labels,
        const void* params) asm("?search@IndexIVF@faiss@@UEBAX_JPEBM0PEAMPEA_JPEBUSearchParameters@2@@Z");
}

/*
 * Motor C++ de FAISS 1.15.1 de latencia pura:
 * - Instancia indices mediante API CPython una unica vez durante setup.
 * - Extrae los punteros nativos directos a la estructura en memoria.
 * - Ejecuta la busqueda llamando directamente al metodo C++ de la DLL.
 * - 1 Hilo OpenMP forzado (omp_set_num_threads = 1).
 */
class FaissNativeEngine {
    PyObject* pModuleFaiss_;
    PyObject* pIndexFlat_;
    PyObject* pQuantizer_;
    PyObject* pIndexIVF_;
    PyObject* pSelectorBitmap_;
    PyObject* pParamsFlat_;
    PyObject* pParamsIVF_;

    void* raw_flat_ptr_;
    void* raw_ivf_ptr_;
    void* raw_params_flat_;
    void* raw_params_ivf_;

    uint32_t dim_;
    uint32_t num_vectors_;

    std::vector<float> dist_buf_;
    std::vector<int64_t> label_buf_;

public:
    FaissNativeEngine()
        : pModuleFaiss_(nullptr), pIndexFlat_(nullptr), pQuantizer_(nullptr),
          pIndexIVF_(nullptr), pSelectorBitmap_(nullptr),
          pParamsFlat_(nullptr), pParamsIVF_(nullptr),
          raw_flat_ptr_(nullptr), raw_ivf_ptr_(nullptr),
          raw_params_flat_(nullptr), raw_params_ivf_(nullptr),
          dim_(0), num_vectors_(0) {
        dist_buf_.resize(512, 0.0f);
        label_buf_.resize(512, 0);
    }

    ~FaissNativeEngine() {
        if (pParamsIVF_) { Py_DECREF(pParamsIVF_); pParamsIVF_ = nullptr; }
        if (pParamsFlat_) { Py_DECREF(pParamsFlat_); pParamsFlat_ = nullptr; }
        if (pSelectorBitmap_) { Py_DECREF(pSelectorBitmap_); pSelectorBitmap_ = nullptr; }
        if (pIndexIVF_) { Py_DECREF(pIndexIVF_); pIndexIVF_ = nullptr; }
        if (pQuantizer_) { Py_DECREF(pQuantizer_); pQuantizer_ = nullptr; }
        if (pIndexFlat_) { Py_DECREF(pIndexFlat_); pIndexFlat_ = nullptr; }
        if (pModuleFaiss_) { Py_DECREF(pModuleFaiss_); pModuleFaiss_ = nullptr; }
    }

    bool init(uint32_t dim, uint32_t num_vectors, const float* raw_vectors) {
        dim_ = dim;
        num_vectors_ = num_vectors;

        if (!Py_IsInitialized()) {
            Py_Initialize();
        }
        if (_import_array() < 0) {
            std::cerr << "Error: _import_array() fallo en NumPy." << std::endl;
            return false;
        }

        pModuleFaiss_ = PyImport_ImportModule("faiss");
        if (!pModuleFaiss_) {
            PyErr_Print();
            return false;
        }

        /* 1 Hilo OpenMP estrictamente obligatorio */
        PyObject* pSetThreads = PyObject_GetAttrString(pModuleFaiss_, "omp_set_num_threads");
        if (pSetThreads) {
            PyObject* pArgs = Py_BuildValue("(i)", 1);
            PyObject_CallObject(pSetThreads, pArgs);
            Py_DECREF(pArgs);
            Py_DECREF(pSetThreads);
        }

        npy_intp dims_b[2] = { static_cast<npy_intp>(num_vectors), static_cast<npy_intp>(dim) };
        PyObject* pXb = PyArray_SimpleNewFromData(2, dims_b, NPY_FLOAT32, const_cast<float*>(raw_vectors));

        /* 1. Construir IndexFlatL2 nativo */
        PyObject* pClassIndexFlat = PyObject_GetAttrString(pModuleFaiss_, "IndexFlatL2");
        PyObject* pArgsFlat = Py_BuildValue("(i)", dim);
        pIndexFlat_ = PyObject_CallObject(pClassIndexFlat, pArgsFlat);
        Py_DECREF(pArgsFlat);
        Py_DECREF(pClassIndexFlat);
        if (!pIndexFlat_) {
            PyErr_Print();
            Py_DECREF(pXb);
            return false;
        }

        PyObject* pAddFlat = PyObject_CallMethod(pIndexFlat_, "add", "O", pXb);
        if (!pAddFlat) {
            PyErr_Print();
            Py_DECREF(pXb);
            return false;
        }
        Py_DECREF(pAddFlat);

        /* 2. Construir IndexIVFFlat nativo (nlist=128, METRIC_L2=0) */
        PyObject* pClassQuantizer = PyObject_GetAttrString(pModuleFaiss_, "IndexFlatL2");
        PyObject* pArgsQuant = Py_BuildValue("(i)", dim);
        pQuantizer_ = PyObject_CallObject(pClassQuantizer, pArgsQuant);
        Py_DECREF(pArgsQuant);
        Py_DECREF(pClassQuantizer);

        PyObject* pClassIndexIVF = PyObject_GetAttrString(pModuleFaiss_, "IndexIVFFlat");
        PyObject* pArgsIVF = Py_BuildValue("(Oiii)", pQuantizer_, dim, 128, 0);
        pIndexIVF_ = PyObject_CallObject(pClassIndexIVF, pArgsIVF);
        Py_DECREF(pArgsIVF);
        Py_DECREF(pClassIndexIVF);
        if (!pIndexIVF_) {
            PyErr_Print();
            Py_DECREF(pXb);
            return false;
        }

        PyObject* pTrainIVF = PyObject_CallMethod(pIndexIVF_, "train", "O", pXb);
        if (!pTrainIVF) {
            PyErr_Print();
            Py_DECREF(pXb);
            return false;
        }
        Py_DECREF(pTrainIVF);

        PyObject* pAddIVF = PyObject_CallMethod(pIndexIVF_, "add", "O", pXb);
        if (!pAddIVF) {
            PyErr_Print();
            Py_DECREF(pXb);
            return false;
        }
        Py_DECREF(pAddIVF);
        Py_DECREF(pXb);

        /* 3. Instanciar SearchParameters y SearchParametersIVF */
        PyObject* pClassParams = PyObject_GetAttrString(pModuleFaiss_, "SearchParameters");
        pParamsFlat_ = PyObject_CallObject(pClassParams, NULL);
        Py_DECREF(pClassParams);

        PyObject* pClassParamsIVF = PyObject_GetAttrString(pModuleFaiss_, "SearchParametersIVF");
        pParamsIVF_ = PyObject_CallObject(pClassParamsIVF, NULL);
        Py_DECREF(pClassParamsIVF);

        /* 4. Extraer punteros nativos directos (desreferenciando swig.this) */
        raw_flat_ptr_ = extract_native_ptr(pIndexFlat_);
        raw_ivf_ptr_ = extract_native_ptr(pIndexIVF_);
        raw_params_flat_ = extract_native_ptr(pParamsFlat_);
        raw_params_ivf_ = extract_native_ptr(pParamsIVF_);

        if (!raw_flat_ptr_ || !raw_ivf_ptr_ || !raw_params_flat_ || !raw_params_ivf_) {
            std::cerr << "Error al extraer punteros nativos C++ de FAISS." << std::endl;
            return false;
        }

        return true;
    }

    void set_ivf_nprobe(size_t nprobe) {
        if (!raw_params_ivf_) return;
        /* SearchParametersIVF: offset 16 es nprobe */
        *reinterpret_cast<uint64_t*>(static_cast<char*>(raw_params_ivf_) + 16) = nprobe;
    }

    bool update_bitmap_filter(const uint8_t* bitmap_bytes, size_t num_bytes) {
        if (pSelectorBitmap_) {
            Py_DECREF(pSelectorBitmap_);
            pSelectorBitmap_ = nullptr;
        }

        npy_intp dims[1] = { static_cast<npy_intp>(num_bytes) };
        PyObject* pBmpArray = PyArray_SimpleNewFromData(1, dims, NPY_UINT8, const_cast<uint8_t*>(bitmap_bytes));

        PyObject* pClassSelector = PyObject_GetAttrString(pModuleFaiss_, "IDSelectorBitmap");
        PyObject* pArgsSel = Py_BuildValue("(O)", pBmpArray);
        pSelectorBitmap_ = PyObject_CallObject(pClassSelector, pArgsSel);
        Py_DECREF(pArgsSel);
        Py_DECREF(pClassSelector);
        Py_DECREF(pBmpArray);

        if (!pSelectorBitmap_) {
            PyErr_Print();
            return false;
        }

        void* raw_sel_ptr = extract_native_ptr(pSelectorBitmap_);

        /* Asignar IDSelector* sel en SearchParameters (offset 8) */
        *reinterpret_cast<void**>(static_cast<char*>(raw_params_flat_) + 8) = raw_sel_ptr;
        *reinterpret_cast<void**>(static_cast<char*>(raw_params_ivf_) + 8) = raw_sel_ptr;

        return true;
    }

    /* Busqueda nativa C++ pura IndexFlatL2 */
    inline void search_flat(const float* q, uint32_t k, Neighbor* out_res, uint32_t* out_count, double& out_time_us) {
        uint64_t t0 = get_highres_time_ns();
        faiss_IndexFlat_search(raw_flat_ptr_, 1, q, k, dist_buf_.data(), label_buf_.data(), raw_params_flat_);
        uint64_t t1 = get_highres_time_ns();
        out_time_us = static_cast<double>(t1 - t0) / 1000.0;

        *out_count = k;
        for (uint32_t i = 0; i < k; ++i) {
            out_res[i].id = static_cast<uint32_t>(label_buf_[i]);
            out_res[i].dist_sq = dist_buf_[i];
        }
    }

    /* Busqueda nativa C++ pura IndexIVFFlat */
    inline void search_ivf(const float* q, uint32_t k, Neighbor* out_res, uint32_t* out_count, double& out_time_us) {
        uint64_t t0 = get_highres_time_ns();
        faiss_IndexIVF_search(raw_ivf_ptr_, 1, q, k, dist_buf_.data(), label_buf_.data(), raw_params_ivf_);
        uint64_t t1 = get_highres_time_ns();
        out_time_us = static_cast<double>(t1 - t0) / 1000.0;

        *out_count = k;
        for (uint32_t i = 0; i < k; ++i) {
            out_res[i].id = static_cast<uint32_t>(label_buf_[i]);
            out_res[i].dist_sq = dist_buf_[i];
        }
    }

private:
    static void* extract_native_ptr(PyObject* py_obj) {
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
};



} // namespace alg_flat
