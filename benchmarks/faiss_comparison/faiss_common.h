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
 * Wrapper de alto rendimiento para interactuar con FAISS desde C++:
 * - Control estricto de hilos (omp_set_num_threads = 1).
 * - Cero E/S o invocacion de procesos externos por consulta.
 * - Reutilizacion de buffers de entrada/salida para no pagar asignaciones.
 */
class FaissEngine {
    PyObject* pModuleFaiss_;
    PyObject* pIndexFlat_;
    PyObject* pIndexIVF_;
    PyObject* pParamsFlatBitmap_;
    PyObject* pParamsIVFBitmap_;
    PyObject* pSearchFlatFunc_;
    PyObject* pSearchIVFFunc_;
    PyObject* pQueryArray_;
    float* pQueryBuffer_;
    uint32_t dim_;
    uint32_t num_vectors_;

public:
    FaissEngine() : pModuleFaiss_(nullptr), pIndexFlat_(nullptr), pIndexIVF_(nullptr),
                    pParamsFlatBitmap_(nullptr), pParamsIVFBitmap_(nullptr),
                    pSearchFlatFunc_(nullptr), pSearchIVFFunc_(nullptr),
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

        /* 1 hilo OpenMP estrictamente obligatorio */
        PyObject* pSetThreads = PyObject_GetAttrString(pModuleFaiss_, "omp_set_num_threads");
        if (pSetThreads) {
            PyObject* pArgs = Py_BuildValue("(i)", 1);
            PyObject_CallObject(pSetThreads, pArgs);
            Py_DECREF(pArgs);
            Py_DECREF(pSetThreads);
        }

        /* 1. Crear e indexar IndexFlatL2 */
        PyObject* pClassIndexFlat = PyObject_GetAttrString(pModuleFaiss_, "IndexFlatL2");
        PyObject* pArgsFlat = Py_BuildValue("(i)", dim);
        pIndexFlat_ = PyObject_CallObject(pClassIndexFlat, pArgsFlat);
        Py_DECREF(pArgsFlat);
        Py_DECREF(pClassIndexFlat);
        if (!pIndexFlat_) {
            PyErr_Print();
            return false;
        }

        npy_intp dims_b[2] = { static_cast<npy_intp>(num_vectors), static_cast<npy_intp>(dim) };
        PyObject* pXb = PyArray_SimpleNewFromData(2, dims_b, NPY_FLOAT32, const_cast<float*>(raw_vectors));
        PyObject* pAddFunc = PyObject_GetAttrString(pIndexFlat_, "add");
        PyObject* pAddArgs = Py_BuildValue("(O)", pXb);
        PyObject* pAddRes = PyObject_CallObject(pAddFunc, pAddArgs);
        Py_DECREF(pAddArgs);
        Py_DECREF(pAddFunc);
        if (pAddRes) {
            Py_DECREF(pAddRes);
        } else {
            Py_DECREF(pXb);
            PyErr_Print();
            return false;
        }

        /* 2. Crear, entrenar e indexar IndexIVFFlat (nlist = 128) */
        PyObject* pClassQuantizer = PyObject_GetAttrString(pModuleFaiss_, "IndexFlatL2");
        PyObject* pArgsQ = Py_BuildValue("(i)", dim);
        PyObject* pQuantizer = PyObject_CallObject(pClassQuantizer, pArgsQ);
        Py_DECREF(pArgsQ);
        Py_DECREF(pClassQuantizer);

        PyObject* pClassIVF = PyObject_GetAttrString(pModuleFaiss_, "IndexIVFFlat");
        /* IndexIVFFlat(quantizer, dim, nlist, METRIC_L2) */
        PyObject* pMetricL2 = PyObject_GetAttrString(pModuleFaiss_, "METRIC_L2");
        PyObject* pArgsIVF = Py_BuildValue("(Oiii)", pQuantizer, dim, 128, PyLong_AsLong(pMetricL2));
        Py_DECREF(pMetricL2);
        pIndexIVF_ = PyObject_CallObject(pClassIVF, pArgsIVF);
        Py_DECREF(pArgsIVF);
        Py_DECREF(pClassIVF);
        Py_DECREF(pQuantizer);

        if (!pIndexIVF_) {
            Py_DECREF(pXb);
            PyErr_Print();
            return false;
        }

        PyObject* pTrainFunc = PyObject_GetAttrString(pIndexIVF_, "train");
        PyObject* pTrainArgs = Py_BuildValue("(O)", pXb);
        PyObject* pTrainRes = PyObject_CallObject(pTrainFunc, pTrainArgs);
        Py_DECREF(pTrainArgs);
        Py_DECREF(pTrainFunc);
        if (pTrainRes) {
            Py_DECREF(pTrainRes);
        } else {
            Py_DECREF(pXb);
            PyErr_Print();
            return false;
        }

        PyObject* pAddIVFFunc = PyObject_GetAttrString(pIndexIVF_, "add");
        PyObject* pAddIVFArgs = Py_BuildValue("(O)", pXb);
        PyObject* pAddIVFRes = PyObject_CallObject(pAddIVFFunc, pAddIVFArgs);
        Py_DECREF(pAddIVFArgs);
        Py_DECREF(pAddIVFFunc);
        Py_DECREF(pXb);
        if (pAddIVFRes) {
            Py_DECREF(pAddIVFRes);
        } else {
            PyErr_Print();
            return false;
        }

        pSearchFlatFunc_ = PyObject_GetAttrString(pIndexFlat_, "search");
        pSearchIVFFunc_ = PyObject_GetAttrString(pIndexIVF_, "search");

        /* Buffer persistente para no pagar asignacion por consulta */
        npy_intp dims_q[2] = { 1, static_cast<npy_intp>(dim) };
        pQueryArray_ = PyArray_SimpleNew(2, dims_q, NPY_FLOAT32);
        pQueryBuffer_ = static_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(pQueryArray_)));

        return true;
    }

    void set_ivf_nprobe(int nprobe) {
        if (!pIndexIVF_) return;
        PyObject_SetAttrString(pIndexIVF_, "nprobe", PyLong_FromLong(nprobe));
    }

    bool update_bitmap_filter(const uint8_t* bitmap_bytes, size_t num_bytes) {
        if (pParamsFlatBitmap_) { Py_DECREF(pParamsFlatBitmap_); pParamsFlatBitmap_ = nullptr; }
        if (pParamsIVFBitmap_) { Py_DECREF(pParamsIVFBitmap_); pParamsIVFBitmap_ = nullptr; }

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
        PyObject* pKwFlat = PyDict_New();
        PyDict_SetItemString(pKwFlat, "sel", pSelector);
        pParamsFlatBitmap_ = PyObject_Call(pClassParams, PyTuple_New(0), pKwFlat);
        Py_DECREF(pKwFlat);

        /* SearchParametersIVF para IVFFlat */
        PyObject* pClassParamsIVF = PyObject_GetAttrString(pModuleFaiss_, "SearchParametersIVF");
        PyObject* pKwIVF = PyDict_New();
        PyDict_SetItemString(pKwIVF, "sel", pSelector);
        pParamsIVFBitmap_ = PyObject_Call(pClassParamsIVF, PyTuple_New(0), pKwIVF);
        Py_DECREF(pKwIVF);
        Py_DECREF(pClassParamsIVF);

        Py_DECREF(pClassParams);
        Py_DECREF(pSelector);

        return (pParamsFlatBitmap_ != nullptr && pParamsIVFBitmap_ != nullptr);
    }

    void search_flat(const float* q, uint32_t k, Neighbor* out_res, uint32_t* out_count, double& out_time_us) {
        memcpy(pQueryBuffer_, q, dim_ * sizeof(float));

        PyObject* pArgs = PyTuple_New(2);
        Py_INCREF(pQueryArray_);
        PyTuple_SetItem(pArgs, 0, pQueryArray_);
        PyTuple_SetItem(pArgs, 1, PyLong_FromLong(k));

        PyObject* pKw = PyDict_New();
        if (pParamsFlatBitmap_) PyDict_SetItemString(pKw, "params", pParamsFlatBitmap_);

        uint64_t t0 = get_highres_time_ns();
        PyObject* pRes = PyObject_Call(pSearchFlatFunc_, pArgs, pKw);
        uint64_t t1 = get_highres_time_ns();
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

    void search_ivf(const float* q, uint32_t k, Neighbor* out_res, uint32_t* out_count, double& out_time_us) {
        memcpy(pQueryBuffer_, q, dim_ * sizeof(float));

        PyObject* pArgs = PyTuple_New(2);
        Py_INCREF(pQueryArray_);
        PyTuple_SetItem(pArgs, 0, pQueryArray_);
        PyTuple_SetItem(pArgs, 1, PyLong_FromLong(k));

        PyObject* pKw = PyDict_New();
        if (pParamsIVFBitmap_) PyDict_SetItemString(pKw, "params", pParamsIVFBitmap_);

        uint64_t t0 = get_highres_time_ns();
        PyObject* pRes = PyObject_Call(pSearchIVFFunc_, pArgs, pKw);
        uint64_t t1 = get_highres_time_ns();
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

    ~FaissEngine() {
        if (pQueryArray_) Py_DECREF(pQueryArray_);
        if (pParamsFlatBitmap_) Py_DECREF(pParamsFlatBitmap_);
        if (pParamsIVFBitmap_) Py_DECREF(pParamsIVFBitmap_);
        if (pSearchFlatFunc_) Py_DECREF(pSearchFlatFunc_);
        if (pSearchIVFFunc_) Py_DECREF(pSearchIVFFunc_);
        if (pIndexFlat_) Py_DECREF(pIndexFlat_);
        if (pIndexIVF_) Py_DECREF(pIndexIVF_);
        if (pModuleFaiss_) Py_DECREF(pModuleFaiss_);
    }
};

} // namespace alg_flat
