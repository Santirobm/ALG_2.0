#!/usr/bin/env python3
# Copyright (c) 2026 Santiago Robledo.
# SPDX-License-Identifier: Apache-2.0
"""Export real Hugging Face embedding rows into deterministic ALG Hito 3.3 files.

Dependencies:
  pip install "datasets>=3,<5" "numpy>=1.26" "scikit-learn>=1.4"

The exporter does not synthesize embeddings. It reads an existing vector column.
When source dimension > target D, PCA is fitted on corpus rows only and then
applied to held-out query rows. It never pads vectors or labels projections as
native-dimensional data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from sklearn.decomposition import IncrementalPCA


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, help="Hugging Face dataset ID")
    p.add_argument("--config", default=None, help="Dataset config/subset")
    p.add_argument("--split", default="train", help="HF split to load")
    p.add_argument("--revision", required=True, help="Immutable Hugging Face dataset revision (commit SHA or tag)")
    p.add_argument("--vector-column", required=True, help="Column containing float embeddings")
    p.add_argument("--id-column", default=None)
    p.add_argument("--tenant-column", default=None)
    p.add_argument("--category-column", default=None)
    p.add_argument("--timestamp-column", default=None)
    p.add_argument("--require-metadata", action="store_true",
                   help="Fail unless tenant, category and timestamp columns are supplied and present")
    p.add_argument("--rows", type=int, default=100_000,
                   help="Maximum rows to load before deterministic train/query split")
    p.add_argument("--query-fraction", type=float, default=0.10)
    p.add_argument("--dimensions", default="32,64,128,256")
    p.add_argument("--seed", type=int, default=3303)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--output", type=Path, default=Path("data/hito3_3"))
    p.add_argument("--no-l2-normalize", action="store_true",
                   help="Preserve embedding magnitudes instead of normalizing rows")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [json_safe(x) for x in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    return str(value)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_f32(path: Path, matrix: np.ndarray) -> None:
    # Explicit little-endian float32, row-major, no header: trivial mmap in C/C++.
    np.asarray(matrix, dtype="<f4", order="C").tofile(path)


def main() -> int:
    args = parse_args()
    if args.rows < 100 or not 0.0 < args.query_fraction < 0.5:
        raise SystemExit("--rows must be >=100 and --query-fraction must be in (0, 0.5)")
    dims = sorted(set(int(x.strip()) for x in args.dimensions.split(",") if x.strip()))
    if not dims or any(d not in (32, 64, 128, 256) for d in dims):
        raise SystemExit("--dimensions accepts only 32,64,128,256")

    print(f"Loading {args.dataset} [{args.config or 'default'}] split={args.split} ...", flush=True)
    ds = load_dataset(args.dataset, args.config, split=args.split, revision=args.revision)
    required_cols = [args.vector_column]
    for c in (args.id_column, args.tenant_column, args.category_column, args.timestamp_column):
        if c:
            required_cols.append(c)
    missing = [c for c in required_cols if c not in ds.column_names]
    if missing:
        raise SystemExit(f"Dataset is missing requested columns: {missing}; available={ds.column_names}")
    if args.require_metadata and not all((args.tenant_column, args.category_column, args.timestamp_column)):
        raise SystemExit("--require-metadata requires --tenant-column, --category-column and --timestamp-column")

    take = min(args.rows, len(ds))
    if take < args.rows:
        print(f"Dataset has only {take} rows; using all available rows.", flush=True)
    ds = ds.select(range(take))

    vectors = np.asarray(ds[args.vector_column], dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != take:
        raise SystemExit(f"Expected a dense 2D vector column; got shape={vectors.shape}")
    if not np.isfinite(vectors).all():
        raise SystemExit("Embedding column contains NaN or infinity; refusing silent cleanup.")
    source_dim = int(vectors.shape[1])
    if max(dims) > source_dim:
        raise SystemExit(f"Source D={source_dim}; cannot create D={max(dims)} without padding (forbidden).")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(take)
    n_query = max(1, int(round(take * args.query_fraction)))
    query_idx = order[:n_query]
    corpus_idx = order[n_query:]
    if len(corpus_idx) < 100:
        raise SystemExit("Too few corpus rows after query holdout.")
    raw_corpus = np.ascontiguousarray(vectors[corpus_idx], dtype=np.float32)
    raw_queries = np.ascontiguousarray(vectors[query_idx], dtype=np.float32)

    def metadata(indices: np.ndarray) -> list[dict[str, Any]]:
        result = []
        for idx in indices.tolist():
            row: dict[str, Any] = {
                "row_id": int(idx),
                "external_id": json_safe(ds[args.id_column][idx]) if args.id_column else int(idx),
            }
            for key, col in (("tenant_id", args.tenant_column),
                             ("category", args.category_column),
                             ("timestamp", args.timestamp_column)):
                row[key] = json_safe(ds[col][idx]) if col else None
            result.append(row)
        return result

    corpus_meta = metadata(corpus_idx)
    query_meta = metadata(query_idx)
    args.output.mkdir(parents=True, exist_ok=True)

    for d in dims:
        out = args.output / f"D{d}"
        out.mkdir(parents=True, exist_ok=True)
        if d == source_dim:
            corpus_d = raw_corpus.copy()
            queries_d = raw_queries.copy()
            projection = {"type": "native", "source_dimension": source_dim, "target_dimension": d}
        else:
            # Fit only on corpus. IncrementalPCA avoids a full N x N covariance matrix.
            batch = max(args.batch_size, d * 4)
            ipca = IncrementalPCA(n_components=d, batch_size=batch)
            for start in range(0, len(raw_corpus), batch):
                chunk = raw_corpus[start:start + batch]
                if len(chunk) >= d:
                    ipca.partial_fit(chunk)
            corpus_d = ipca.transform(raw_corpus).astype(np.float32, copy=False)
            queries_d = ipca.transform(raw_queries).astype(np.float32, copy=False)
            projection = {
                "type": "incremental_pca",
                "source_dimension": source_dim,
                "target_dimension": d,
                "fit_split": "corpus_only",
                "explained_variance_ratio_sum": float(np.sum(ipca.explained_variance_ratio_)),
                "components_sha256": hashlib.sha256(np.asarray(ipca.components_, dtype="<f4").tobytes()).hexdigest(),
            }

        if not args.no_l2_normalize:
            for matrix in (corpus_d, queries_d):
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                if np.any(norms <= 1e-12):
                    raise SystemExit(f"Projection D={d} produced a zero/near-zero vector.")
                matrix /= norms

        corpus_path = out / "corpus.f32"
        query_path = out / "queries.f32"
        corpus_meta_path = out / "corpus_metadata.jsonl"
        query_meta_path = out / "query_metadata.jsonl"
        write_f32(corpus_path, corpus_d)
        write_f32(query_path, queries_d)
        write_jsonl(corpus_meta_path, corpus_meta)
        write_jsonl(query_meta_path, query_meta)

        id_col_arg = f"--id-column {args.id_column} " if args.id_column else ""
        export_cmd = (
            f"python benchmarks/realworld/prepare_hf_embeddings.py "
            f"--dataset {args.dataset} "
            f"--revision {args.revision} "
            f"--vector-column {args.vector_column} "
            f"{id_col_arg}"
            f"--rows {args.rows} "
            f"--query-fraction {args.query_fraction:.2f} "
            f"--seed {args.seed} "
            f"--dimensions {','.join(str(x) for x in dims)} "
            f"--output {args.output}"
        )
        manifest = {
            "format": "ALG-HITO3.3-F32-ROWMAJOR-v1",
            "dataset": args.dataset,
            "dataset_commit": args.revision,
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "export_command": export_cmd,
            "config": args.config,
            "source_split": args.split,
            "source_vector_column": args.vector_column,
            "source_dimension": source_dim,
            "dimension": d,
            "corpus_rows": int(len(corpus_d)),
            "query_rows": int(len(queries_d)),
            "seed": args.seed,
            "query_fraction": args.query_fraction,
            "l2_normalized": not args.no_l2_normalize,
            "projection": projection,
            "metadata_columns": {
                "id": args.id_column,
                "tenant": args.tenant_column,
                "category": args.category_column,
                "timestamp": args.timestamp_column,
            },
            "metadata_completeness": {
                key: sum(row[key] is not None for row in corpus_meta) / len(corpus_meta)
                for key in ("tenant_id", "category", "timestamp")
            },
            "files": {
                "corpus.f32": {
                    "sha256": sha256_file(corpus_path),
                    "bytes": corpus_path.stat().st_size,
                    "rows": int(len(corpus_d)),
                },
                "queries.f32": {
                    "sha256": sha256_file(query_path),
                    "bytes": query_path.stat().st_size,
                    "rows": int(len(queries_d)),
                },
                "corpus_metadata.jsonl": {
                    "sha256": sha256_file(corpus_meta_path),
                    "bytes": corpus_meta_path.stat().st_size,
                    "rows": len(corpus_meta),
                },
                "query_metadata.jsonl": {
                    "sha256": sha256_file(query_meta_path),
                    "bytes": query_meta_path.stat().st_size,
                    "rows": len(query_meta),
                },
            },
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                           encoding="utf-8")
        print(f"D={d}: corpus={len(corpus_d):,}, queries={len(queries_d):,}, "
              f"projection={projection['type']}, variance={projection.get('explained_variance_ratio_sum', 1.0):.4f}",
              flush=True)

    print(f"Export complete: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
