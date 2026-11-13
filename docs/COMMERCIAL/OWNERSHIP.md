# ALG Vector Engine — Ownership, Authorship & Licensing Verification

**Repository:** `Santirobm/ALG_2.0`  
**Reference Release:** `v1.0.0` (Commit `275d417c716932bd3c8ffce90883f748a0fc1232`)  
**Primary License:** Apache License 2.0 (`LICENSE`)  
**Verification Branch:** `chore/commercial-readiness`

---

## 1. Git Technical Identity

Inspection of `git log --all --format="%h | %ai | A: %an <%ae> | C: %cn <%ce> | %s"` across all branches and tags shows **39 commits** with identical Author and Committer fields on every commit, under three Git configuration identities:

| Commits | Git Author & Committer Name | Git Email | Date Range (UTC-03:00) |
| :---: | :--- | :--- | :--- |
| 27 | `SGI LM` | `Sgi.LM@tubhier.local` | `2026-11-02 16:38` – `2026-11-04 09:58` (includes `v1.0.0` at `275d417`) |
| 4 | `Santirobm` | `santiagorobledo198@gmail.com` | `2026-11-03 15:55` – `2026-11-03 16:29` |
| 8 | `Santiago Robledo` | `santiagorobledo198@gmail.com` | `2026-11-04 10:34` – `2026-11-12 16:19` |

### Governance & Copyright Notices
- **`LICENSE`**: Standard Apache License 2.0 (unmodified).
- **`AUTHORS.md`**: Attributes original work to **Santiago Robledo** (initial public release `v1.0.0`, `2026-11-04`).
- **`CONTRIBUTING.md`**: Requires Apache 2.0 compatibility and `Signed-off-by` (Developer Certificate of Origin) for incoming pull requests (introduced in commit `aaac8c7`).
- **SPDX Headers**: All 25 core headers, source files, and build files (`include/`, `src/`, `CMakeLists.txt`, `build.ps1`) carry `Copyright (c) 2026 Santiago Robledo.` and `SPDX-License-Identifier: Apache-2.0` (added in `aaac8c7`).

---

## 2. Actual Authorship

Based on the observable Git metadata, all 39 commits can **reasonably be attributed to a single developer** (Santiago Robledo / `Santirobm`) working across two Git configurations:
- Commits alternate within minutes on the same days (`2026-11-03` and `2026-11-04`) in the same `-03:00` timezone with a continuous, sequential commit progression (e.g., `9e59436` at `15:52` by `SGI LM` followed by `d28dceb` at `15:55` by `Santirobm`; `d55be11` at `09:58` by `SGI LM` followed by `aaac8c7` at `10:34` by `Santiago Robledo`).
- `SGI LM` corresponds to the local Windows user/workstation profile (`C:\Users\sgi.lm\...`) rather than an external contributor.
- **Unverified via Git alone:** Git metadata cannot independently prove who physically typed each commit or whether any external AI/tooling or third party assisted. Commits prior to `aaac8c7` do not contain `Signed-off-by` DCO trailers.

---

## 3. Legal Ownership

- **Public Licensing:** The repository is licensed under **Apache License 2.0**. An acquirer can purchase copyright ownership (if held cleanly by the author), repository/trademark control, and future commercial/dual-licensing rights for future versions, but **cannot retroactively revoke** Apache 2.0 rights already granted to prior public recipients of `v1.0.0`.
- **No Exclusive Ownership Claim Without Diligence:** Exclusive, unencumbered ownership cannot be certified from repository files alone and remains subject to the employment/IP questions below.

---

## 4. Potential Employment / IP Considerations

Because 27 commits use the local workstation email `Sgi.LM@tubhier.local` (a `.local` domain associated with a work PC name):
- **Do not assume employer ownership**, and **do not assume absence of employer claims** without verification.
- The presence of a work-PC Git configuration does not by itself transfer copyright to an employer, but buyers conducting M&A or IP acquisition diligence routinely examine whether code was written on company hardware, during working hours, or under an employment contract with an IP assignment clause.

### Questions Only the Developer Can Answer
1. Were any commits in `ALG_2.0` created during working hours, using company-owned hardware, or using company resources associated with `tubhier.local`?
2. Is the developer bound by an employment agreement, contractor agreement, or IP assignment clause that covers software developed outside or alongside their job?
3. Can the developer provide a written authorship declaration and, if applicable, an employer IP waiver / non-assertion letter prior to closing an acquisition?

---

## 5. Third-Party Code & Dependencies

### Production SDK (`libalg.a`, `alg.dll`, `libalg.so`)
- **Zero third-party runtime dependencies.** Compiles from standard C++17/C11 using only OS/compiler standard headers and x86_64 SIMD intrinsics (`<immintrin.h>`). Does not link against FAISS, BLAS, MKL, OpenMP, or Python.

### Benchmarking & Testing Only (Excluded from Production SDK)
1. **FAISS (`third_party/faiss/`):** Upstream `facebookresearch/faiss` `v1.15.1` (commit `75c755c5a1fc854756aaf3bd1131213aeb21d8d1`), **MIT License** (Meta Platforms, Inc. and affiliates). Used solely for comparative benchmarks and equivalence tests.
2. **Evaluation Datasets (`data/`):** Derived PCA projections from Hugging Face `lsb/simplewiki2023-all-minilm-l6-v2-embedding` (revision `14270bcfcf2b7854cddce4395c0b38a199edef05`). Underlying SimpleWiki text is CC BY-SA 3.0 / GFDL; embedding model `all-MiniLM-L6-v2` is Apache 2.0. Raw `.f32`, `.u32`, and `.jsonl` files are gitignored and not distributed with the SDK.
