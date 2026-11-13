"""
run_python_integration_bench.py
Focused Python integration benchmark: ALG (ctypes -> bin/alg.dll) vs FAISS Python.

Dataset: data/hito3_3/D64 (N=90,000 real embeddings, D=64)
Candidates: M = 900 (1%) and M = 9,000 (10%), Top-K = 10
Measures:
  - Total Python-perceived latency (mean, p50, p95, QPS)
  - Estimated Python/ctypes call overhead (via no-op / raw C call baseline)
  - Exact IDSetRecall (%) against binary64 ground-truth oracle
"""

import ctypes
import os
import sys
import time
import csv
import numpy as np

# Ensure bin/ is on PATH / DLL search path for alg.dll
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BIN_DIR = os.path.join(ROOT_DIR, "bin")
if hasattr(os, "add_dll_directory") and os.path.isdir(BIN_DIR):
    os.add_dll_directory(BIN_DIR)
os.environ["PATH"] = BIN_DIR + os.pathsep + os.environ.get("PATH", "")

import faiss

faiss.omp_set_num_threads(1)


# --- ALG Ctypes Bindings ---
class AlgIndexConfig(ctypes.Structure):
    _fields_ = [
        ("dimension", ctypes.c_uint32),
        ("initial_capacity", ctypes.c_uint32),
        ("metric", ctypes.c_int32),
    ]


class AlgSearchResult(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint32),
        ("distance", ctypes.c_float),
    ]


def load_alg_lib():
    dll_path = os.path.join(BIN_DIR, "alg.dll")
    lib = ctypes.CDLL(dll_path)

    lib.alg_get_version.restype = ctypes.c_char_p
    lib.alg_get_active_isa_name.restype = ctypes.c_char_p

    lib.alg_create_index.argtypes = [ctypes.POINTER(AlgIndexConfig), ctypes.POINTER(ctypes.c_void_p)]
    lib.alg_create_index.restype = ctypes.c_int32

    lib.alg_free_index.argtypes = [ctypes.c_void_p]
    lib.alg_free_index.restype = None

    lib.alg_set_vectors.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)]
    lib.alg_set_vectors.restype = ctypes.c_int32

    lib.alg_search_compact.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_int32,
        ctypes.POINTER(AlgSearchResult),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.alg_search_compact.restype = ctypes.c_int32
    return lib


def compute_stats(samples_us):
    arr = np.sort(np.asarray(samples_us, dtype=np.float64))
    mean = float(np.mean(arr))
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    qps = 1e6 / mean if mean > 0 else 0.0
    return mean, p50, p95, qps


def oracle_topk_f64(corpus_f64, q_f64, cand_ids, k):
    sub = corpus_f64[cand_ids]
    diffs = sub - q_f64
    dists = np.einsum("ij,ij->i", diffs, diffs)
    # Deterministic ascending ID tie-break (cand_ids is already sorted ascending)
    order = np.argsort(dists, kind="mergesort")[:k]
    return cand_ids[order]


def main():
    N = 90000
    D = 64
    K = 10
    NUM_QUERIES = 200
    REPS = 5

    data_dir = os.path.join(ROOT_DIR, "data", "hito3_3", f"D{D}")
    corpus = np.fromfile(os.path.join(data_dir, "corpus.f32"), dtype=np.float32).reshape(N, D)
    queries = np.fromfile(os.path.join(data_dir, "queries.f32"), dtype=np.float32).reshape(-1, D)[:NUM_QUERIES]
    corpus = np.ascontiguousarray(corpus)
    queries = np.ascontiguousarray(queries)
    corpus_f64 = corpus.astype(np.float64)

    alg = load_alg_lib()
    print("=" * 72)
    print("ALG (Python ctypes -> alg.dll) vs FAISS Python Integration Benchmark")
    print("=" * 72)
    print(f"ALG Version : {alg.alg_get_version().decode()}")
    print(f"ALG Active  : {alg.alg_get_active_isa_name().decode()}")
    print(f"Config      : N={N}, D={D}, Top-K={K}, Queries={NUM_QUERIES}, Reps={REPS} ({NUM_QUERIES*REPS} samples/cell)\n")

    # Build ALG Index
    cfg = AlgIndexConfig(dimension=D, initial_capacity=N, metric=0)
    alg_idx = ctypes.c_void_p()
    t0 = time.perf_counter_ns()
    assert alg.alg_create_index(ctypes.byref(cfg), ctypes.byref(alg_idx)) == 0
    assert alg.alg_set_vectors(alg_idx, N, corpus.ctypes.data_as(ctypes.POINTER(ctypes.c_float))) == 0
    t1 = time.perf_counter_ns()
    alg_build_ms = (t1 - t0) / 1e6

    # Build FAISS IndexFlatL2
    t2 = time.perf_counter_ns()
    faiss_idx = faiss.IndexFlatL2(D)
    faiss_idx.add(corpus)
    t3 = time.perf_counter_ns()
    faiss_build_ms = (t3 - t2) / 1e6
    print(f"[Index Build] ALG={alg_build_ms:.2f} ms | FAISS={faiss_build_ms:.2f} ms\n")

    # Measure bare ctypes call overhead (alg_get_version)
    n_ovh = 5000
    t_ovh0 = time.perf_counter_ns()
    for _ in range(n_ovh):
        alg.alg_get_version()
    t_ovh1 = time.perf_counter_ns()
    ctypes_noop_us = (t_ovh1 - t_ovh0) / (n_ovh * 1000.0)
    print(f"[Overhead Baseline] Bare ctypes call = {ctypes_noop_us:.3f} us/call\n")

    # Pre-allocated output buffers
    alg_out_res = (AlgSearchResult * K)()
    alg_out_cnt = ctypes.c_uint32(0)
    alg_out_cnt_ptr = ctypes.byref(alg_out_cnt)

    rng = np.random.default_rng(42)
    out_csv = os.path.join(ROOT_DIR, "results", "sdk_comparison", "python_n90k_d64.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    rows = []
    for M in [900, 9000]:
        sel_pct = (100.0 * M) / N
        print(f"--- Candidates M={M} (sel={sel_pct:.1f}%, Top-K={K}) ---")

        # Generate identical candidate sets (sorted unique uint32 and int64)
        cand_u32_list = []
        cand_i64_list = []
        oracles = []
        for q in range(NUM_QUERIES):
            ids = np.sort(rng.choice(N, size=M, replace=False)).astype(np.uint32)
            cand_u32_list.append(ids)
            cand_i64_list.append(ids.astype(np.int64))
            oracles.append(set(oracle_topk_f64(corpus_f64, queries[q].astype(np.float64), ids, K).tolist()))

        # Pre-allocated FAISS buffers
        dist_buf = np.empty((1, M), dtype=np.float32)
        dist_buf_1d = dist_buf[0]
        dist_ptr = faiss.swig_ptr(dist_buf)

        alg_times = []
        faiss_subset_times = []
        faiss_subset_swig_only_times = []
        faiss_batch_times = []

        alg_recalls = []
        faiss_sub_recalls = []
        faiss_batch_recalls = []

        for r in range(REPS):
            for q in range(NUM_QUERIES):
                q_1d = queries[q]
                q_2d = queries[q : q + 1]
                q_cptr = q_1d.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                q_swig = faiss.swig_ptr(q_2d)

                ids_u32 = cand_u32_list[q]
                ids_cptr = ids_u32.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
                ids_i64 = cand_i64_list[q]
                ids_swig = faiss.swig_ptr(ids_i64)

                # 1. ALG ctypes
                t_a0 = time.perf_counter_ns()
                rc = alg.alg_search_compact(
                    alg_idx, q_cptr, ids_cptr, M, K, 0, alg_out_res, alg_out_cnt_ptr
                )
                t_a1 = time.perf_counter_ns()
                if rc != 0:
                    raise RuntimeError(f"alg_search_compact failed: {rc}")
                alg_times.append((t_a1 - t_a0) / 1000.0)
                if r == 0:
                    got = {alg_out_res[i].id for i in range(alg_out_cnt.value)}
                    alg_recalls.append(100.0 * len(got & oracles[q]) / K)

                # 2. FAISS Python: compute_distance_subset + np.argpartition Top-K
                t_f0 = time.perf_counter_ns()
                faiss_idx.compute_distance_subset(1, q_swig, M, dist_ptr, ids_swig)
                t_f1 = time.perf_counter_ns()
                top_local = np.argpartition(dist_buf_1d, K - 1)[:K]
                top_local = top_local[np.argsort(dist_buf_1d[top_local], kind="mergesort")]
                top_ids = ids_u32[top_local]
                t_f2 = time.perf_counter_ns()
                faiss_subset_swig_only_times.append((t_f1 - t_f0) / 1000.0)
                faiss_subset_times.append((t_f2 - t_f0) / 1000.0)
                if r == 0:
                    got_f = set(top_ids.tolist())
                    faiss_sub_recalls.append(100.0 * len(got_f & oracles[q]) / K)

                # 3. FAISS Python: standard idx.search with IDSelectorBatch
                t_b0 = time.perf_counter_ns()
                sel = faiss.IDSelectorBatch(M, ids_swig)
                params = faiss.SearchParameters(sel=sel)
                D_b, I_b = faiss_idx.search(q_2d, K, params=params)
                t_b1 = time.perf_counter_ns()
                faiss_batch_times.append((t_b1 - t_b0) / 1000.0)
                if r == 0:
                    got_b = {int(x) for x in I_b[0] if x >= 0}
                    faiss_batch_recalls.append(100.0 * len(got_b & oracles[q]) / K)

        a_mean, a_p50, a_p95, a_qps = compute_stats(alg_times)
        fs_mean, fs_p50, fs_p95, fs_qps = compute_stats(faiss_subset_times)
        fso_mean, fso_p50, fso_p95, _ = compute_stats(faiss_subset_swig_only_times)
        fb_mean, fb_p50, fb_p95, fb_qps = compute_stats(faiss_batch_times)

        sp_sub_mean = fs_mean / a_mean
        sp_sub_p50 = fs_p50 / a_p50
        sp_batch_mean = fb_mean / a_mean
        sp_batch_p50 = fb_p50 / a_p50

        print(f"  ALG (ctypes)             : mean={a_mean:7.2f} us | p50={a_p50:7.2f} us | p95={a_p95:7.2f} us | QPS={a_qps:9.0f} | Recall={np.mean(alg_recalls):.2f}%")
        print(f"  FAISS (subset+argpart)   : mean={fs_mean:7.2f} us | p50={fs_p50:7.2f} us | p95={fs_p95:7.2f} us | QPS={fs_qps:9.0f} | Recall={np.mean(faiss_sub_recalls):.2f}%  (subset-only p50={fso_p50:.2f} us)")
        print(f"  FAISS (IDSelectorBatch)  : mean={fb_mean:7.2f} us | p50={fb_p50:7.2f} us | p95={fb_p95:7.2f} us | QPS={fb_qps:9.0f} | Recall={np.mean(faiss_batch_recalls):.2f}%")
        print(f"  -> Speedup vs subset+argpart : mean={sp_sub_mean:.2f}x | p50={sp_sub_p50:.2f}x")
        print(f"  -> Speedup vs IDSelectorBatch: mean={sp_batch_mean:.2f}x | p50={sp_batch_p50:.2f}x\n")

        rows.append([M, sel_pct, K, "ALG_ctypes", a_mean, a_p50, a_p95, a_qps, np.mean(alg_recalls)])
        rows.append([M, sel_pct, K, "FAISS_subset_argpartition", fs_mean, fs_p50, fs_p95, fs_qps, np.mean(faiss_sub_recalls)])
        rows.append([M, sel_pct, K, "FAISS_IDSelectorBatch", fb_mean, fb_p50, fb_p95, fb_qps, np.mean(faiss_batch_recalls)])

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["M", "selectivity_pct", "top_k", "engine", "mean_us", "p50_us", "p95_us", "qps", "recall_pct"])
        w.writerows(rows)
    print(f"Saved CSV to: {out_csv}")

    alg.alg_free_index(alg_idx)


if __name__ == "__main__":
    main()
