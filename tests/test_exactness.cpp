#include "../include/alg_flat_types.h"
#include "../include/dataset_loader.h"
#include "../include/physical_layouts.h"
#include "../include/execution_plans.h"
#include "../include/ieee754_bounds.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <cstdio>

using namespace alg_flat;

/*
 * ============================================================================
 * ORACULO DE PRECISION EXTENDIDA Y CONTRATO DE EJECUCION IEEE-754 (HITO 2A.3)
 * ============================================================================
 * 1. Verifica que las primitivas de intervalos en binary64 (add_up_f64, mul_up_f64,
 *    div_up_f64, sqrt_up_f64, sub_down_f64, mul_down_f64, div_down_f64, sqrt_down_f64)
 *    y las funciones compute_partition_certified_bounds / compute_partition_distance_bounds /
 *    compute_radial_shell_interval envuelvan de forma conservadora al oraculo en `long double`
 *    a lo largo de 12 ordenes de magnitud (1e-6 a 1e14) y casos de borde (0.0, 1 ULP, asimetria).
 * 2. Verifica en tiempo de ejecucion el contrato MXCSR y el dominio de coordenadas:
 *    - Si se activa FTZ (0x8000), DAZ (0x0040) o un modo de redondeo distinto de Nearest-Even,
 *      `check_ieee754_mxcsr_state()` lo detecta y tanto Plan 5A como Plan 5B desactivan
 *      inmediatamente las podas y conmutan a ExactScan (`metrics.fallback_triggered == 1`),
 *      preservando 100.0000% IDSetRecall y RankMatch.
 *    - Si una consulta contiene una coordenada fuera de CERTIFIED_MAX_COORD_ABS (> 1.0e15f)
 *      o no finita (`INFINITY` / `NaN`), `check_vector_domain_certified()` rechaza la poda
 *      certificada y activa el fallback a ExactScan.
 */
/*
 * ============================================================================
 * ORACULO DE PRECISION EXTENDIDA DE 106 BITS (DEKKER/KNUTH DOUBLE-DOUBLE + FMA)
 * ============================================================================
 * En la ABI de Windows x86_64 (MSVC / Clang-MSVC), `long double` tiene el mismo
 * ancho de mantisa que `double` (`LDBL_MANT_DIG == 53`). Para disponer de un
 * oraculo verdaderamente independiente y de mayor precision que `binary64` en
 * cualquier sistema operativo, implementamos aritmetica exacta `DoubleDouble`
 * (`dd106`, 106 bits de mantisa = 2 x 53 bits, u_106 = 2^-106 = 2.465e-32)
 * basada en las transformaciones libres de error (Error-Free Transformations):
 *   - `two_sum(a, b)` de Knuth (1969): s + e = a + b exacto.
 *   - `two_prod(a, b)` con FMA IEEE-754 de hardware (`std::fma`): p + e = a * b exacto.
 */
struct dd106 {
    double hi;
    double lo;

    constexpr dd106(double h = 0.0, double l = 0.0) : hi(h), lo(l) {}
};

static inline dd106 dd_two_sum(double a, double b) {
    double s = a + b;
    double v = s - a;
    double e = (a - (s - v)) + (b - v);
    return {s, e};
}

static inline dd106 dd_quick_two_sum(double a, double b) {
    double s = a + b;
    double e = b - (s - a);
    return {s, e};
}

static inline dd106 dd_two_prod(double a, double b) {
    double p = a * b;
    double e = std::fma(a, b, -p);
    return {p, e};
}

static inline dd106 dd_add(dd106 a, dd106 b) {
    dd106 s = dd_two_sum(a.hi, b.hi);
    dd106 t = dd_two_sum(a.lo, b.lo);
    s.lo += t.hi;
    s = dd_quick_two_sum(s.hi, s.lo);
    s.lo += t.lo;
    return dd_quick_two_sum(s.hi, s.lo);
}

static inline dd106 dd_sub(dd106 a, dd106 b) {
    return dd_add(a, dd106(-b.hi, -b.lo));
}

static inline dd106 dd_mul(dd106 a, dd106 b) {
    dd106 p = dd_two_prod(a.hi, b.hi);
    p.lo += a.hi * b.lo + a.lo * b.hi;
    return dd_quick_two_sum(p.hi, p.lo);
}

static inline dd106 dd_div(dd106 a, dd106 b) {
    double q1 = a.hi / b.hi;
    dd106 r = dd_sub(a, dd_mul(b, dd106(q1, 0.0)));
    double q2 = r.hi / b.hi;
    r = dd_sub(r, dd_mul(b, dd106(q2, 0.0)));
    double q3 = r.hi / b.hi;
    dd106 q = dd_quick_two_sum(q1, q2);
    return dd_add(q, dd106(q3, 0.0));
}

static inline dd106 dd_sqrt(dd106 a) {
    if (a.hi <= 0.0) return {0.0, 0.0};
    dd106 x(std::sqrt(a.hi), 0.0);
    /* Dos iteraciones de Newton-Raphson en precision de 106 bits: x = 0.5 * (x + a / x) */
    x = dd_mul(dd106(0.5, 0.0), dd_add(x, dd_div(a, x)));
    x = dd_mul(dd106(0.5, 0.0), dd_add(x, dd_div(a, x)));
    return x;
}

/* Devuelve true si a >= b en precision extendida de 106 bits */
static inline bool dd_ge(dd106 a, dd106 b) {
    dd106 diff = dd_sub(a, b);
    return (diff.hi > 0.0) || (diff.hi == 0.0 && diff.lo >= 0.0);
}

/* Devuelve true si a <= b en precision extendida de 106 bits */
static inline bool dd_le(dd106 a, dd106 b) {
    dd106 diff = dd_sub(a, b);
    return (diff.hi < 0.0) || (diff.hi == 0.0 && diff.lo <= 0.0);
}

static void verify_directed_rounding_and_interval_oracle() {
    assert(check_ieee754_mxcsr_state() &&
           "El estado MXCSR inicial del proceso debe cumplir Round-to-Nearest-Even, FTZ=0, DAZ=0!");

    const double scales[10] = {
        1.0e-6, 1.0e-4, 1.0e-2, 0.25, 1.0, 15.0, 500.0, 25000.0, 1.0e9, 1.0e13
    };
    const double ratios[4][4] = {
        {1.37, 0.91, 0.63, 0.42},
        {1.0,  1.0e-5, 1.0, 1.0e-5}, /* Alta asimetria entre escalas */
        {1.0e-5, 1.0, 1.0e-5, 1.0},  /* Asimetria inversa */
        {0.0,  0.0,  1.0, 1.0}       /* Casos con norma cero */
    };

    const dd106 u32_dd(IEEE754_F32_U, 0.0);
    const dd106 one_dd(1.0, 0.0);
    auto gamma_dd = [&](uint32_t n) -> dd106 {
        dd106 nu = dd_mul(dd106(static_cast<double>(n), 0.0), u32_dd);
        return dd_div(nu, dd_sub(one_dd, nu));
    };

    const dd106 g17_dd = gamma_dd(17);
    const dd106 g33_dd = gamma_dd(33);
    const dd106 g64_dd = gamma_dd(64);
    const dd106 g66_dd = gamma_dd(66);
    const dd106 g67_dd = gamma_dd(67);

    const dd106 coeff_q_dd = dd_add(dd_mul(dd106(8.0, 0.0), dd_mul(g64_dd, dd_add(one_dd, u32_dd))),
                                    dd_mul(dd106(4.0, 0.0), u32_dd));
    const dd106 coeff_c_dd = dd_mul(dd106(4.0, 0.0), u32_dd);
    const dd106 coeff_x_dd = dd_mul(dd106(2.0, 0.0), u32_dd);

    for (int s = 0; s < 10; ++s) {
        for (int r = 0; r < 4; ++r) {
            float q_norm = static_cast<float>(scales[s] * ratios[r][0]);
            float c_norm = static_cast<float>(scales[s] * ratios[r][1]);
            float dc_hi  = static_cast<float>(scales[s] * ratios[r][2]);
            float r_max  = static_cast<float>(scales[s] * ratios[r][3]);
            /* Probar tambien con perturbacion de 1 ULP */
            if ((s + r) % 2 == 1 && q_norm > 0.0f) {
                q_norm = std::nextafterf(q_norm, INFINITY);
            }
            float def = 2.95e-7f;

            PartitionCertifiedBounds b = compute_partition_certified_bounds(q_norm, c_norm, dc_hi, r_max, def);

            dd106 q_dd(static_cast<double>(q_norm), 0.0);
            dd106 c_dd(static_cast<double>(c_norm), 0.0);
            dd106 dc_dd(static_cast<double>(dc_hi), 0.0);
            dd106 rm_dd(static_cast<double>(r_max), 0.0);
            dd106 def_dd(static_cast<double>(def), 0.0);

            dd106 delta_vec_oracle = dd_add(dd_add(dd_mul(coeff_q_dd, q_dd), dd_mul(coeff_c_dd, c_dd)),
                                            dd_mul(coeff_x_dd, rm_dd));
            assert(dd_ge(dd106(static_cast<double>(b.delta_vec_p), 0.0), delta_vec_oracle) &&
                   "Oraculo 106-bit fallo: delta_vec_p debe ser >= delta_vec_oracle!");

            dd106 dv_dd(static_cast<double>(b.delta_vec_p), 0.0);
            dd106 tail_err_oracle = dd_add(dd_mul(g64_dd, dd_add(dc_dd, dv_dd)), dd_mul(u32_dd, rm_dd));
            dd106 fp_slack_oracle = dd_mul(dd_add(one_dd, dd_mul(dd106(4.0, 0.0), u32_dd)),
                                           dd_add(dv_dd, tail_err_oracle));
            assert(dd_ge(dd106(static_cast<double>(b.fp_slack_p), 0.0), fp_slack_oracle) &&
                   "Oraculo 106-bit fallo: fp_slack_p debe ser >= fp_slack_oracle!");

            dd106 max_vec_oracle = dd_add(dd_mul(dd_sqrt(dd_add(one_dd, def_dd)), dd_add(dc_dd, rm_dd)), dv_dd);
            dd106 base_sq_oracle = dd_mul(dd106(2.0, 0.0), dd_mul(dv_dd, max_vec_oracle));
            assert(dd_ge(dd106(static_cast<double>(b.sq_slack_16_p), 0.0), dd_mul(base_sq_oracle, dd_add(one_dd, g17_dd))) &&
                   "Oraculo 106-bit fallo: sq_slack_16_p debe ser >= cota exacta 16D!");
            assert(dd_ge(dd106(static_cast<double>(b.sq_slack_32_p), 0.0), dd_mul(base_sq_oracle, dd_add(one_dd, g33_dd))) &&
                   "Oraculo 106-bit fallo: sq_slack_32_p debe ser >= cota exacta 32D!");
            assert(dd_ge(dd106(static_cast<double>(b.sq_slack_64_p), 0.0), dd_mul(base_sq_oracle, dd_add(one_dd, g67_dd))) &&
                   "Oraculo 106-bit fallo: sq_slack_64_p debe ser >= cota exacta 64D!");

            dd106 common_num_dd = dd_mul(dd_sub(one_dd, g66_dd), dd_sub(one_dd, dd_mul(dd106(6.0, 0.0), u32_dd)));
            dd106 den_dd = dd_add(one_dd, def_dd);
            assert(dd_le(dd106(static_cast<double>(b.scale_safe_16), 0.0), dd_div(dd_mul(common_num_dd, dd_sub(one_dd, g17_dd)), den_dd)) &&
                   "Oraculo 106-bit fallo: scale_safe_16 debe ser <= factor exacto 16D!");
            assert(dd_le(dd106(static_cast<double>(b.scale_safe_32), 0.0), dd_div(dd_mul(common_num_dd, dd_sub(one_dd, g33_dd)), den_dd)) &&
                   "Oraculo 106-bit fallo: scale_safe_32 debe ser <= factor exacto 32D!");
            assert(dd_le(dd106(static_cast<double>(b.scale_safe_64), 0.0), dd_div(dd_mul(common_num_dd, dd_sub(one_dd, g67_dd)), den_dd)) &&
                   "Oraculo 106-bit fallo: scale_safe_64 debe ser <= factor exacto 64D!");

            assert(b.sq_slack_16_p <= b.sq_slack_32_p && b.sq_slack_32_p <= b.sq_slack_64_p &&
                   "Monotonia de holgura cuadratica por etapa (16D <= 32D <= 64D) violada!");
            assert(b.scale_safe_16 >= b.scale_safe_32 && b.scale_safe_32 >= b.scale_safe_64 &&
                   "Monotonia de factor de escala por etapa (16D >= 32D >= 64D) violada!");

            /* Verificar tambien compute_partition_distance_bounds y compute_radial_shell_interval */
            float d2_f32 = dc_hi * dc_hi;
            float r_min  = r_max * 0.25f;
            PartitionDistanceBounds pdb = compute_partition_distance_bounds(d2_f32, r_min, r_max);
            dd106 d2_dd(static_cast<double>(d2_f32), 0.0);
            dd106 one_m_g66_dd = dd_sub(one_dd, g66_dd);
            dd106 dc_true_low  = dd_sqrt(dd_mul(d2_dd, one_m_g66_dd));
            dd106 dc_true_high = dd_sqrt(dd_div(d2_dd, one_m_g66_dd));
            assert(dd_le(dd106(static_cast<double>(pdb.dc_low), 0.0), dc_true_low) &&
                   "Oraculo 106-bit fallo: dc_low debe ser <= dc_true_low!");
            assert(dd_ge(dd106(static_cast<double>(pdb.dc_high), 0.0), dc_true_high) &&
                   "Oraculo 106-bit fallo: dc_high debe ser >= dc_true_high!");

            float r_low = 0.0f, r_high = 0.0f, r_pad = 0.0f;
            float thresh_sq = (r_max * 0.5f) * (r_max * 0.5f);
            compute_radial_shell_interval(pdb.dc_low, pdb.dc_high, r_max, thresh_sq, r_low, r_high, r_pad);
            dd106 rho_oracle = dd_sqrt(dd_div(dd106(static_cast<double>(thresh_sq), 0.0), one_m_g66_dd));
            dd106 r_pad_oracle = dd_mul(u32_dd, rm_dd);
            assert(dd_ge(dd106(static_cast<double>(r_pad), 0.0), r_pad_oracle) &&
                   "Oraculo 106-bit fallo: r_pad debe ser >= r_pad_oracle!");
            dd106 raw_low_dd = dd_sub(dd_sub(dd106(static_cast<double>(pdb.dc_low), 0.0), rho_oracle),
                                      dd106(static_cast<double>(r_pad), 0.0));
            dd106 r_low_oracle = (raw_low_dd.hi > 0.0) ? raw_low_dd : dd106(0.0, 0.0);
            dd106 r_high_oracle = dd_add(dd_add(dd106(static_cast<double>(pdb.dc_high), 0.0), rho_oracle),
                                         dd106(static_cast<double>(r_pad), 0.0));
            assert(dd_le(dd106(static_cast<double>(r_low), 0.0), r_low_oracle) &&
                   "Oraculo 106-bit fallo: r_low debe ser <= r_low_oracle!");
            assert(dd_ge(dd106(static_cast<double>(r_high), 0.0), r_high_oracle) &&
                   "Oraculo 106-bit fallo: r_high debe ser >= r_high_oracle!");
        }
    }
}

/*
 * Verifica las 5 situaciones formales del contrato de entorno IEEE-754 y dominio:
 *   1. Ejecucion nominal (`OK_CERTIFIED`).
 *   2. `MXCSR` alterado (`FTZ=1` + redondeo != Nearest-Even): desactiva la poda, restaura
 *      temporalmente `MXCSR` canonico durante `ExactScan`, devuelve el `MXCSR` del llamador
 *      intacto al salir y preserva `RankMatch@K = 100%`.
 *   3. Consulta finita fuera del dominio certificado de poda (`2.0e16f > 1.0e15f`), pero dentro
 *      del dominio seguro de fallback (`<= 1.0e18f`): ejecuta `ExactScan` sin overflow y preserva
 *      `RankMatch@K = 100%` contra Ground Truth.
 *   4. Consulta no procesable con garantias (`1.0e20f > 1.0e18f` que desbordaria `ExactScan`,
 *      `INFINITY` y `NAN`): rechaza explicitamente la consulta (`query_rejected == 1`, `cnt == 0`,
 *      `status == REJECTED_UNSAFE_QUERY_DOMAIN`).
 *   5. Corpus invalido en construccion (`1.0e20f` o `NAN` en el corpus): `build_ivf_partitioned_index`
 *      rechaza construir el indice (`return false`, `certified_Contract_valid == false`), y cualquier
 *      consulta sobre dicho indice es rechazada con `status == REJECTED_INVALID_CORPUS_INDEX`.
 */
static void verify_runtime_environment_and_domain_contract() {
    BenchmarkDataset ds;
    generate_dataset_deterministic(42, 4096, 10, 16, 0.05f, MIX_R, ds);

    FlatAoSIndex aos_idx;
    IVFPartitionedIndex ivf_idx;
    build_flat_aos_index(ds, aos_idx);
    bool build_ok = build_ivf_partitioned_index(ds, 16, 8, ivf_idx);
    assert(build_ok && ivf_idx.certified_Contract_valid &&
           "El indice IVF base debe construirse y ser valido bajo contrato IEEE-754!");

    const uint32_t sel_idx = 2;
    const uint32_t target_cat = ds.header.selectivities_deci_percent[sel_idx] * 10U;
    const auto& el_ids = ds.eligible_per_sel[sel_idx];
    QueryPredicate pred{target_cat, el_ids.data(), static_cast<uint32_t>(el_ids.size()), sel_idx};

    const uint32_t k = 50;
    Neighbor gt[50], out5a[50], out5b[50];
    uint32_t gt_cnt = 0, cnt5a = 0, cnt5b = 0;
    QueryExecutionMetrics m_gt{}, m5a{}, m5b{};

    const float* qv = ds.queries[0].values;
    execute_plan0_ref_hito9(aos_idx, qv, pred, k, gt, &gt_cnt, m_gt);

    /* Caso 1: Estado nominal MXCSR -> OK_CERTIFIED */
    execute_plan5a_ivf_part_prune_avx2(ivf_idx, qv, pred, k, out5a, &cnt5a, m5a);
    execute_plan5b_ivf_full_exact_prune_avx2(ivf_idx, qv, pred, k, out5b, &cnt5b, m5b);
    assert(m5a.fallback_triggered == 0U && m5b.fallback_triggered == 0U &&
           m5a.query_rejected == 0U && m5b.query_rejected == 0U &&
           m5a.status == QueryExecutionStatus::OK_CERTIFIED &&
           m5b.status == QueryExecutionStatus::OK_CERTIFIED &&
           "En estado MXCSR nominal debe ejecutarse con OK_CERTIFIED!");

    /* Caso 2: Simular que el hilo host activa FTZ (0x8000) y modo de redondeo hacia abajo (0x2000) */
    uint32_t saved_csr = _mm_getcsr();
    uint32_t host_custom_csr = saved_csr | MXCSR_FTZ_BIT | 0x2000U;
    _mm_setcsr(host_custom_csr);
    assert(!check_ieee754_mxcsr_state() && "check_ieee754_mxcsr_state() debe detectar FTZ=1 y redondeo alterado!");

    execute_plan5a_ivf_part_prune_avx2(ivf_idx, qv, pred, k, out5a, &cnt5a, m5a);
    assert((_mm_getcsr() & MXCSR_CERTIFIED_MASK) == (host_custom_csr & MXCSR_CERTIFIED_MASK) &&
           "Plan 5A debe restaurar el estado MXCSR original del hilo tras ejecutar el fallback!");
    execute_plan5b_ivf_full_exact_prune_avx2(ivf_idx, qv, pred, k, out5b, &cnt5b, m5b);
    assert((_mm_getcsr() & MXCSR_CERTIFIED_MASK) == (host_custom_csr & MXCSR_CERTIFIED_MASK) &&
           "Plan 5B debe restaurar el estado MXCSR original del hilo tras ejecutar el fallback!");
    _mm_setcsr(saved_csr); /* Restaurar estado nominal para las siguientes pruebas */
    assert(check_ieee754_mxcsr_state() && "MXCSR restaurado correctamente!");

    assert(m5a.fallback_triggered == 1U && m5b.fallback_triggered == 1U &&
           m5a.query_rejected == 0U && m5b.query_rejected == 0U &&
           m5a.status == QueryExecutionStatus::FALLBACK_MXCSR_RESTORED &&
           m5b.status == QueryExecutionStatus::FALLBACK_MXCSR_RESTORED &&
           "Plan 5A/5B deben activar FALLBACK_MXCSR_RESTORED cuando MXCSR es no estandar!");
    assert(m5a.dists_evaluated == pred.eligible_count && m5b.dists_evaluated == pred.eligible_count &&
           "En modo fallback deben evaluarse todos los vectores elegibles sin podar!");
    assert(cnt5a == gt_cnt && cnt5b == gt_cnt);
    for (uint32_t r = 0; r < k; ++r) {
        assert(out5a[r].id == gt[r].id && out5b[r].id == gt[r].id &&
               out5a[r].dist_sq == gt[r].dist_sq && out5b[r].dist_sq == gt[r].dist_sq &&
               "El fallback con restauracion temporal de MXCSR debe preservar RankMatch@K = 100% y distancias exactas!");
    }

    /* Caso 3: Consulta finita fuera del dominio certificado de poda (2.0e16f > 1e15f), pero segura en ExactScan (<= 1e18f) */
    alignas(64) float q_out_safe[64];
    for (uint32_t d = 0; d < 64; ++d) q_out_safe[d] = qv[d];
    q_out_safe[7] = 2.0e16f;
    assert(!check_vector_domain_certified(q_out_safe, 64) &&
           "check_vector_domain_certified() debe rechazar |coord| = 2.0e16f > 1.0e15f!");
    assert(check_query_canonical_fallback_safe(q_out_safe, 64) &&
           "check_query_canonical_fallback_safe() debe aceptar |coord| = 2.0e16f <= 1.0e18f!");

    Neighbor gt_out[50];
    uint32_t gt_out_cnt = 0;
    execute_plan0_ref_hito9(aos_idx, q_out_safe, pred, k, gt_out, &gt_out_cnt, m_gt);
    execute_plan5a_ivf_part_prune_avx2(ivf_idx, q_out_safe, pred, k, out5a, &cnt5a, m5a);
    execute_plan5b_ivf_full_exact_prune_avx2(ivf_idx, q_out_safe, pred, k, out5b, &cnt5b, m5b);
    assert(m5a.fallback_triggered == 1U && m5b.fallback_triggered == 1U &&
           m5a.query_rejected == 0U && m5b.query_rejected == 0U &&
           m5a.status == QueryExecutionStatus::FALLBACK_QUERY_OUT_OF_CERT_DOMAIN &&
           m5b.status == QueryExecutionStatus::FALLBACK_QUERY_OUT_OF_CERT_DOMAIN &&
           "Coordenadas en (1e15, 1e18] deben activar FALLBACK_QUERY_OUT_OF_CERT_DOMAIN!");
    assert(cnt5a == gt_out_cnt && cnt5b == gt_out_cnt);
    for (uint32_t r = 0; r < k; ++r) {
        assert(std::isfinite(out5a[r].dist_sq) && std::isfinite(out5b[r].dist_sq));
        assert(out5a[r].id == gt_out[r].id && out5b[r].id == gt_out[r].id &&
               out5a[r].dist_sq == gt_out[r].dist_sq && out5b[r].dist_sq == gt_out[r].dist_sq &&
               "El fallback para consultas en (1e15, 1e18] debe producir el ranking exacto de Plan 0!");
    }

    /* Caso 4: Consultas no procesables con garantias (overflow 1.0e20f, INFINITY, NAN) -> rechazo explicito */
    const float unsafe_vals[3] = {1.0e20f, INFINITY, NAN};
    for (int u = 0; u < 3; ++u) {
        alignas(64) float q_unsafe[64];
        for (uint32_t d = 0; d < 64; ++d) q_unsafe[d] = qv[d];
        q_unsafe[7] = unsafe_vals[u];
        assert(!check_query_canonical_fallback_safe(q_unsafe, 64) &&
               "check_query_canonical_fallback_safe() debe rechazar entradas que causen overflow o NaN/Inf!");

        cnt5a = 999U; cnt5b = 999U;
        execute_plan5a_ivf_part_prune_avx2(ivf_idx, q_unsafe, pred, k, out5a, &cnt5a, m5a);
        execute_plan5b_ivf_full_exact_prune_avx2(ivf_idx, q_unsafe, pred, k, out5b, &cnt5b, m5b);
        assert(cnt5a == 0U && cnt5b == 0U &&
               m5a.fallback_triggered == 0U && m5b.fallback_triggered == 0U &&
               m5a.query_rejected == 1U && m5b.query_rejected == 1U &&
               m5a.status == QueryExecutionStatus::REJECTED_UNSAFE_QUERY_DOMAIN &&
               m5b.status == QueryExecutionStatus::REJECTED_UNSAFE_QUERY_DOMAIN &&
               "Consultas con riesgo de overflow o NaN/Inf deben ser rechazadas explicitamente!");
    }

    /* Caso 5: Corpus invalido en construccion (coordenada 1.0e20f o NAN en ds.vectors) -> rechazo del indice */
    float orig_coord = ds.vectors[13];
    const float bad_corpus_vals[2] = {1.0e20f, NAN};
    for (int b = 0; b < 2; ++b) {
        ds.vectors[13] = bad_corpus_vals[b];
        IVFPartitionedIndex bad_idx;
        bool bad_ok = build_ivf_partitioned_index(ds, 16, 8, bad_idx);
        assert(!bad_ok && !bad_idx.certified_Contract_valid && bad_idx.num_vectors == 0U &&
               "build_ivf_partitioned_index() debe rechazar construir el indice cuando el corpus es invalido!");

        cnt5a = 999U; cnt5b = 999U;
        execute_plan5a_ivf_part_prune_avx2(bad_idx, qv, pred, k, out5a, &cnt5a, m5a);
        execute_plan5b_ivf_full_exact_prune_avx2(bad_idx, qv, pred, k, out5b, &cnt5b, m5b);
        assert(cnt5a == 0U && cnt5b == 0U &&
               m5a.fallback_triggered == 0U && m5b.fallback_triggered == 0U &&
               m5a.query_rejected == 1U && m5b.query_rejected == 1U &&
               m5a.status == QueryExecutionStatus::REJECTED_INVALID_CORPUS_INDEX &&
               m5b.status == QueryExecutionStatus::REJECTED_INVALID_CORPUS_INDEX &&
               "Consultas sobre un indice rechazado deben fallar con REJECTED_INVALID_CORPUS_INDEX!");
    }
    ds.vectors[13] = orig_coord;
    _mm_setcsr(saved_csr); /* Limpiar flags sticky de excepcion tras pruebas extremas */
}

/*
 * Nivel 1A: IDSetRecall (SIN tolerancia de distancia `dist_sq <= kth_gt_dist` y CON verificacion
 * explicita de ausencia de IDs duplicados en la salida predicha).
 */
static double compute_id_set_recall(const Neighbor* pred, uint32_t pred_cnt,
                                    const Neighbor* gt, uint32_t gt_cnt, uint32_t k,
                                    uint64_t& inout_duplicate_id_errors) {
    uint32_t target = (gt_cnt < k) ? gt_cnt : k;
    if (target == 0) return 1.0;
    uint32_t n_p = (pred_cnt < k) ? pred_cnt : k;
    uint32_t hits = 0;
    for (uint32_t i = 0; i < n_p; ++i) {
        bool is_duplicate = false;
        for (uint32_t prev = 0; prev < i; ++prev) {
            if (pred[prev].id == pred[i].id) {
                is_duplicate = true;
                inout_duplicate_id_errors++;
                break;
            }
        }
        if (is_duplicate) continue;

        for (uint32_t j = 0; j < target; ++j) {
            if (pred[i].id == gt[j].id) {
                hits++;
                break;
            }
        }
    }
    return static_cast<double>(hits) / static_cast<double>(target);
}

/*
 * Nivel 1B + Nivel 2: Identidad de Rango Posicion-a-Posicion (RankMatch@K) e Igualdad Exacta
 * de Distancia Original (MaxOrigDistErr).
 */
static void verify_rank_and_original_distance(const Neighbor* pred, uint32_t pred_cnt,
                                              const Neighbor* gt, uint32_t gt_cnt, uint32_t k,
                                              uint64_t& out_rank_matches,
                                              uint64_t& out_rank_total,
                                              double& inout_max_orig_dist_diff) {
    uint32_t target = (gt_cnt < k) ? gt_cnt : k;
    for (uint32_t r = 0; r < target; ++r) {
        out_rank_total++;
        if (r < pred_cnt && pred[r].id == gt[r].id) {
            out_rank_matches++;
        }
        if (r < pred_cnt) {
            double diff = std::fabs(static_cast<double>(pred[r].dist_sq) - static_cast<double>(gt[r].dist_sq));
            if (diff > inout_max_orig_dist_diff) {
                inout_max_orig_dist_diff = diff;
            }
        }
    }
}

/*
 * Construye el Escenario Adversarial 2: Centroides alejados del origen (||c_p||_2 >> 0)
 * donde cada cluster tiene un desplazamiento diferente entre +500.0 y +2500.0 por coordenada.
 */
static void make_adversarial_off_origin_dataset(BenchmarkDataset& ds) {
    generate_dataset_deterministic(101, 20000, 80, 64, 0.05f, MIX_R, ds);
    const uint32_t nv = ds.header.num_vectors;
    const uint32_t nq = ds.header.num_queries;
    const uint32_t dim = ds.header.dimension;

    for (uint32_t i = 0; i < nv; ++i) {
        float* v = ds.vectors + static_cast<size_t>(i) * dim;
        uint32_t basin = i % 64U;
        for (uint32_t d = 0; d < dim; ++d) {
            float shift = 500.0f + 25.0f * static_cast<float>(basin) + 3.5f * static_cast<float>(d);
            v[d] += shift;
        }
    }
    for (uint32_t q = 0; q < nq; ++q) {
        float* qv = ds.queries[q].values;
        uint32_t basin = q % 64U;
        for (uint32_t d = 0; d < dim; ++d) {
            float shift = 500.0f + 25.0f * static_cast<float>(basin) + 3.5f * static_cast<float>(d);
            qv[d] += shift;
        }
    }
}

/*
 * Prepara los vectores base para el Escenario Adversarial 3 con empates exactos de distancia
 * y pares separados por 1 ULP.
 */
static void make_adversarial_ties_and_base_dataset(BenchmarkDataset& ds) {
    generate_dataset_deterministic(202, 20000, 60, 32, 0.05f, MIX_50_50, ds);
    const uint32_t dim = ds.header.dimension;

    for (uint32_t i = 0; i < ds.header.num_vectors; ++i) {
        if ((i % 2U) == 0U) {
            ds.categories[i] = 0;
        }
    }

    for (uint32_t q = 50; q < 60; ++q) {
        const float* qv = ds.queries[q].values;
        uint32_t base_vec = q * 200U;
        for (uint32_t m = 0; m < 16; ++m) {
            uint32_t vid = base_vec + m;
            float* target_v = ds.vectors + static_cast<size_t>(vid) * dim;
            ds.categories[vid] = 0;
            float offset = 0.010f + 0.001f * static_cast<float>(m / 2);
            for (uint32_t d = 0; d < dim; ++d) {
                target_v[d] = qv[d] + ((d == (m % dim)) ? -offset : offset);
            }
        }
    }

    uint32_t num_sels = ds.header.num_selectivities;
    for (uint32_t s = 0; s < num_sels; ++s) {
        uint32_t target_cat = ds.header.selectivities_deci_percent[s] * 10U;
        auto& el = ds.eligible_per_sel[s];
        el.clear();
        for (uint32_t i = 0; i < ds.header.num_vectors; ++i) {
            if (ds.categories[i] < target_cat) {
                el.push_back(i);
            }
        }
    }
}

/*
 * Construye y verifica explicitamente consultas con colinealidad exacta respecto del centroide
 * real c_p asignado por K-Means en ivf_idx, tanto sobre las proyecciones matematicas en double
 * como sobre los residuos float32 efectivamente almacenados en ivf_idx.part_aos_vectors.
 */
static void inject_and_verify_true_centroid_collinearity(BenchmarkDataset& ds,
                                                         const IVFPartitionedIndex& ivf_idx,
                                                         uint32_t target_cat) {
    const uint32_t P = ivf_idx.num_partitions;
    const uint32_t dim = ivf_idx.dim;
    double min_cos_radial64 = 1.0;
    double min_cos_tail16   = 1.0;
    double min_cos_tail32   = 1.0;
    double max_rel_gap_radial64 = 0.0;
    double max_rel_gap_tail16   = 0.0;
    double max_rel_gap_tail32   = 0.0;
    double max_unadjusted_ratio_err_radial = 0.0;
    double max_unadjusted_ratio_err_16     = 0.0;
    double max_unadjusted_ratio_err_32     = 0.0;
    uint32_t injected_pairs = 0;

    for (uint32_t q = 0; q < 50; ++q) {
        uint32_t chosen_p = 0;
        uint32_t chosen_pos = 0;
        bool found = false;
        for (uint32_t attempt = 0; attempt < P; ++attempt) {
            uint32_t p = (q + attempt) % P;
            const GeometricPartition& pm = ivf_idx.partitions[p];
            if (pm.count < 180) continue;
            for (int64_t i = static_cast<int64_t>(pm.aos_offset + pm.count) - 1;
                 i >= static_cast<int64_t>(pm.aos_offset); --i) {
                if (ivf_idx.part_aos_categories[i] < target_cat &&
                    ivf_idx.part_aos_tail32_norm[i] > 0.08f &&
                    ivf_idx.part_aos_tail48_norm[i] > 0.12f) {
                    chosen_p = p;
                    chosen_pos = static_cast<uint32_t>(i);
                    found = true;
                    break;
                }
            }
            if (found) break;
        }
        assert(found && "No se encontro vector elegible con cola no nula en las particiones!");

        const float* c_p = ivf_idx.centroids + static_cast<size_t>(chosen_p) * dim;
        const float* x_orig = ivf_idx.part_orig_vectors + static_cast<size_t>(chosen_pos) * dim;
        const float* x_rot_stored_f32 = ivf_idx.part_aos_vectors + static_cast<size_t>(chosen_pos) * dim;

        double x_rot[64];
        double A2 = 0.0, B2 = 0.0, C2 = 0.0;
        for (uint32_t j = 0; j < dim; ++j) {
            x_rot[j] = static_cast<double>(x_rot_stored_f32[j]);
            if (j < 16) A2 += x_rot[j] * x_rot[j];
            else if (j < 32) B2 += x_rot[j] * x_rot[j];
            else C2 += x_rot[j] * x_rot[j];
        }

        double q_rot_target[64];
        if (q < 15) {
            const double alpha = (q % 2 == 0) ? 0.08 : 0.98;
            for (uint32_t j = 0; j < dim; ++j) {
                q_rot_target[j] = alpha * x_rot[j];
            }
        } else if (q < 35) {
            if (q % 2 == 0) {
                const double alpha = 0.35;
                const double beta = std::sqrt((A2 + (1.0 - alpha * alpha) * (B2 + C2)) / A2);
                for (uint32_t j = 0; j < dim; ++j) {
                    q_rot_target[j] = (j < 16) ? (-beta * x_rot[j]) : (alpha * x_rot[j]);
                }
            } else {
                const double alpha = 0.98;
                for (uint32_t j = 0; j < dim; ++j) {
                    q_rot_target[j] = (j < 16) ? x_rot[j] : (alpha * x_rot[j]);
                }
            }
        } else {
            if (q % 2 == 0) {
                const double alpha = 0.35;
                const double gamma = std::sqrt((B2 + (1.0 - alpha * alpha) * C2) / B2);
                for (uint32_t j = 0; j < dim; ++j) {
                    if (j < 16) q_rot_target[j] = x_rot[j];
                    else if (j < 32) q_rot_target[j] = -gamma * x_rot[j];
                    else q_rot_target[j] = alpha * x_rot[j];
                }
            } else {
                const double alpha = 0.98;
                for (uint32_t j = 0; j < dim; ++j) {
                    q_rot_target[j] = (j < 32) ? x_rot[j] : (alpha * x_rot[j]);
                }
            }
        }

        float* qv = ds.queries[q].values;
        for (uint32_t d = 0; d < dim; ++d) {
            double acc = static_cast<double>(c_p[d]);
            for (uint32_t j = 0; j < dim; ++j) {
                const float* row = ivf_idx.ortho_basis + static_cast<size_t>(j) * dim;
                acc += static_cast<double>(row[d]) * q_rot_target[j];
            }
            qv[d] = static_cast<float>(acc);
        }

        double q_rot_actual[64];
        for (uint32_t j = 0; j < dim; ++j) {
            const float* row = ivf_idx.ortho_basis + static_cast<size_t>(j) * dim;
            double acc = 0.0;
            for (uint32_t d = 0; d < dim; ++d) {
                acc += static_cast<double>(row[d]) * (static_cast<double>(qv[d]) - static_cast<double>(c_p[d]));
            }
            q_rot_actual[j] = acc;
        }

        double d_orig_sq = 0.0;
        for (uint32_t d = 0; d < dim; ++d) {
            double df = static_cast<double>(qv[d]) - static_cast<double>(x_orig[d]);
            d_orig_sq += df * df;
        }

        if (q < 15) {
            double dot64 = 0.0, nq64 = 0.0, nx64 = 0.0, diff64 = 0.0;
            for (uint32_t j = 0; j < dim; ++j) {
                dot64 += q_rot_actual[j] * x_rot[j];
                nq64  += q_rot_actual[j] * q_rot_actual[j];
                nx64  += x_rot[j] * x_rot[j];
                double df = q_rot_actual[j] - x_rot[j];
                diff64 += df * df;
            }
            double cos64 = dot64 / (std::sqrt(nq64) * std::sqrt(nx64));
            if (cos64 < min_cos_radial64) min_cos_radial64 = cos64;

            double true_dist = std::sqrt(diff64);
            double rev_tri   = std::fabs(std::sqrt(nq64) - std::sqrt(nx64));
            double rel_gap   = std::fabs(true_dist - rev_tri) / true_dist;
            if (rel_gap > max_rel_gap_radial64) max_rel_gap_radial64 = rel_gap;

            double lb_radial_unadjusted = rev_tri * rev_tri;
            double ratio_err = std::fabs((d_orig_sq / lb_radial_unadjusted) - 1.0);
            if (ratio_err > max_unadjusted_ratio_err_radial) max_unadjusted_ratio_err_radial = ratio_err;
        } else if (q < 35) {
            double dot16 = 0.0, nq16 = 0.0, nx16 = 0.0, diff16 = 0.0, pref16 = 0.0;
            for (uint32_t j = 0; j < 16; ++j) {
                double df = q_rot_actual[j] - x_rot[j];
                pref16 += df * df;
            }
            for (uint32_t j = 16; j < dim; ++j) {
                dot16 += q_rot_actual[j] * x_rot[j];
                nq16  += q_rot_actual[j] * q_rot_actual[j];
                nx16  += x_rot[j] * x_rot[j];
                double df = q_rot_actual[j] - x_rot[j];
                diff16 += df * df;
            }
            double cos16 = dot16 / (std::sqrt(nq16) * std::sqrt(nx16));
            if (cos16 < min_cos_tail16) min_cos_tail16 = cos16;

            double true_tail_dist = std::sqrt(diff16);
            double rev_tri_bound  = std::fabs(std::sqrt(nq16) - std::sqrt(nx16));
            double rel_gap = std::fabs(true_tail_dist - rev_tri_bound) / true_tail_dist;
            if (rel_gap > max_rel_gap_tail16) max_rel_gap_tail16 = rel_gap;

            double lb16_unadjusted = pref16 + rev_tri_bound * rev_tri_bound;
            double ratio_err = std::fabs((d_orig_sq / lb16_unadjusted) - 1.0);
            if (ratio_err > max_unadjusted_ratio_err_16) max_unadjusted_ratio_err_16 = ratio_err;
        } else {
            double dot32 = 0.0, nq32 = 0.0, nx32 = 0.0, diff32 = 0.0, pref32 = 0.0;
            for (uint32_t j = 0; j < 32; ++j) {
                double df = q_rot_actual[j] - x_rot[j];
                pref32 += df * df;
            }
            for (uint32_t j = 32; j < dim; ++j) {
                dot32 += q_rot_actual[j] * x_rot[j];
                nq32  += q_rot_actual[j] * q_rot_actual[j];
                nx32  += x_rot[j] * x_rot[j];
                double df = q_rot_actual[j] - x_rot[j];
                diff32 += df * df;
            }
            double cos32 = dot32 / (std::sqrt(nq32) * std::sqrt(nx32));
            if (cos32 < min_cos_tail32) min_cos_tail32 = cos32;

            double true_tail_dist = std::sqrt(diff32);
            double rev_tri_bound  = std::fabs(std::sqrt(nq32) - std::sqrt(nx32));
            double rel_gap = std::fabs(true_tail_dist - rev_tri_bound) / true_tail_dist;
            if (rel_gap > max_rel_gap_tail32) max_rel_gap_tail32 = rel_gap;

            double lb32_unadjusted = pref32 + rev_tri_bound * rev_tri_bound;
            double ratio_err = std::fabs((d_orig_sq / lb32_unadjusted) - 1.0);
            if (ratio_err > max_unadjusted_ratio_err_32) max_unadjusted_ratio_err_32 = ratio_err;
        }
        injected_pairs++;
    }

    printf("[Verificacion Geom. Suite 3 respecto al centroide real c_p asignado por K-Means]\n");
    printf("  Pares colineales inyectados : %u (25 fuera de Top-K podables, 25 dentro de Top-K)\n", injected_pairs);
    printf("  Min cos(theta) respecto c_p : 64D=%.9f | Cola16..63=%.9f | Cola32..63=%.9f\n",
           min_cos_radial64, min_cos_tail16, min_cos_tail32);
    printf("  Max gap relativo desig. tri.: 64D=%.3e | Cola16..63=%.3e | Cola32..63=%.3e\n",
           max_rel_gap_radial64, max_rel_gap_tail16, max_rel_gap_tail32);
    printf("  Max |d_orig^2/LB_exact^2 - 1|: Radial=%.3e | PCA16=%.3e | PCA32=%.3e\n",
           max_unadjusted_ratio_err_radial, max_unadjusted_ratio_err_16, max_unadjusted_ratio_err_32);

    assert(min_cos_radial64 > 0.999999 && "Fallo de colinealidad 64D respecto del centroide real c_p!");
    assert(min_cos_tail16 > 0.999999 && "Fallo de colinealidad en cola 16..63 respecto del centroide real c_p!");
    assert(min_cos_tail32 > 0.999999 && "Fallo de colinealidad en cola 32..63 respecto del centroide real c_p!");
    assert(max_rel_gap_radial64 < 1e-6 && "La cota radial 64D no alcanza la igualdad en la desigualdad triangular!");
    assert(max_rel_gap_tail16 < 1e-6 && "La cola 16..63 no alcanza la igualdad en la desigualdad triangular!");
    assert(max_rel_gap_tail32 < 1e-6 && "La cola 32..63 no alcanza la igualdad en la desigualdad triangular!");
    assert(max_unadjusted_ratio_err_radial < 1e-5 && "La cota radial sin holgura debe coincidir con d_orig^2!");
    assert(max_unadjusted_ratio_err_16 < 1e-5 && "La cota 16D sin holgura debe coincidir con d_orig^2!");
    assert(max_unadjusted_ratio_err_32 < 1e-5 && "La cota 32D sin holgura debe coincidir con d_orig^2!");
}

static void run_certification_suite(const char* suite_name,
                                    BenchmarkDataset& ds,
                                    uint32_t num_partitions,
                                    bool inject_true_collinear = false) {
    std::cout << "\n--------------------------------------------------------------------------------\n";
    std::cout << "SUITE DE CERTIFICACION: " << suite_name
              << " (N=" << ds.header.num_vectors << ", Q=" << ds.header.num_queries
              << ", P=" << num_partitions << ")\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    FlatAoSIndex aos_idx;
    FlatSoAIndex soa_idx;
    FlatAoSoAIndex aosoa_idx;
    IVFPartitionedIndex ivf_idx;

    build_flat_aos_index(ds, aos_idx);
    build_flat_soa_index(ds, soa_idx);
    build_flat_aosoa_index(ds, aosoa_idx);
    build_ivf_partitioned_index(ds, num_partitions, 12, ivf_idx);
    assert(ivf_idx.certified_Contract_valid && "El indice IVF de la suite debe cumplir el contrato IEEE-754!");

    printf("Defecto de ortogonalidad medido ||Q*Q^T - I||_F = %.3e | Contrato IEEE-754 = OK\n",
           ivf_idx.ortho_defect);

    const uint32_t sel_idx = 2; /* 15.0% selectividad */
    const uint32_t target_cat = ds.header.selectivities_deci_percent[sel_idx] * 10U;
    const auto& el_ids = ds.eligible_per_sel[sel_idx];
    QueryPredicate pred{target_cat, el_ids.data(), static_cast<uint32_t>(el_ids.size()), sel_idx};

    if (inject_true_collinear) {
        inject_and_verify_true_centroid_collinearity(ds, ivf_idx, target_cat);
    }

    const uint32_t k = 100;
    Neighbor gt[100], out[100];
    uint32_t gt_cnt = 0, out_cnt = 0;
    QueryExecutionMetrics m{};

    double id_set_r10[9] = {0}, id_set_r100[9] = {0};
    uint64_t dup_id_errors[9] = {0};
    uint64_t rank_matches_10[9] = {0}, rank_totals_10[9] = {0};
    uint64_t rank_matches_100[9] = {0}, rank_totals_100[9] = {0};
    double max_orig_dist_err[9] = {0};

    PruningAuditReport audit{};
    audit.min_ratio_orig_over_lb = 1e30;
    audit.max_signed_excess = -1e30;
    audit.max_positive_excess = 0.0;

    PruningAuditReport audit6{};
    audit6.min_ratio_orig_over_lb = 1e30;
    audit6.max_signed_excess = -1e30;
    audit6.max_positive_excess = 0.0;

    const char* names[9] = {
        "Plan 1  (Flat_Scalar_AoS)",
        "Plan 2A (Flat_AVX2_AoS_1x)",
        "Plan 2B (Flat_AVX2_AoS_4x)",
        "Plan 3A (Flat_AVX2_SoA)",
        "Plan 3B (Flat_AVX2_AoSoA)",
        "Plan 4  (IVF_Flat_AVX2)",
        "Plan 5A (IVF_PartPrune_AVX2)",
        "Plan 5B (IVF_FullExactPrune_AVX2)",
        "Plan 6  (IVF_Adaptive_CostModel)"
    };

    for (uint32_t q = 0; q < ds.header.num_queries; ++q) {
        const float* qv = ds.queries[q].values;
        execute_plan0_ref_hito9(aos_idx, qv, pred, k, gt, &gt_cnt, m);

        for (int p = 0; p < 9; ++p) {
            if (p == 0) execute_plan1_flat_scalar_aos(aos_idx, qv, pred, k, out, &out_cnt, m);
            else if (p == 1) execute_plan2a_flat_avx2_aos_1x(aos_idx, qv, pred, k, out, &out_cnt, m);
            else if (p == 2) execute_plan2b_flat_avx2_aos_4x(aos_idx, qv, pred, k, out, &out_cnt, m);
            else if (p == 3) execute_plan3a_flat_avx2_soa(soa_idx, qv, pred, k, out, &out_cnt, m);
            else if (p == 4) execute_plan3b_flat_avx2_aosoa(aosoa_idx, qv, pred, k, out, &out_cnt, m);
            else if (p == 5) execute_plan4_ivf_flat_avx2(ivf_idx, qv, pred, k, out, &out_cnt, m);
            else if (p == 6) execute_plan5a_ivf_part_prune_avx2(ivf_idx, qv, pred, k, out, &out_cnt, m);
            else if (p == 7) execute_plan5b_ivf_full_exact_prune_avx2(ivf_idx, qv, pred, k, out, &out_cnt, m);
            else if (p == 8) execute_plan6_ivf_adaptive_cost_model_avx2(ivf_idx, qv, pred, k, out, &out_cnt, m);

            id_set_r10[p] += compute_id_set_recall(out, out_cnt, gt, gt_cnt, 10, dup_id_errors[p]);
            id_set_r100[p] += compute_id_set_recall(out, out_cnt, gt, gt_cnt, 100, dup_id_errors[p]);
            double dummy_err = 0.0;
            verify_rank_and_original_distance(out, out_cnt, gt, gt_cnt, 10,
                                              rank_matches_10[p], rank_totals_10[p], dummy_err);
            verify_rank_and_original_distance(out, out_cnt, gt, gt_cnt, 100,
                                              rank_matches_100[p], rank_totals_100[p], max_orig_dist_err[p]);
        }

        /* Nivel 3: Auditar cada vector podado por Plan 5B y Plan 6 en las 5 etapas mediante el kernel unificado */
        audit_plan5b_pruning_soundness(ivf_idx, qv, pred, k, gt, gt_cnt, audit);
        audit_plan6_adaptive_soundness(ivf_idx, qv, pred, k, gt, gt_cnt, audit6);
    }

    double nq = static_cast<double>(ds.header.num_queries);
    printf("%-34s | %12s | %13s | %12s | %13s | %7s | %13s\n",
           "Plan", "IDSetRec@10", "IDSetRec@100", "RankMatch@10", "RankMatch@100", "DupIDs", "MaxOrigDistErr");
    printf("-----------------------------------------------------------------------------------------------------------------------\n");
    for (int i = 0; i < 9; ++i) {
        double r10 = (id_set_r10[i] / nq) * 100.0;
        double r100 = (id_set_r100[i] / nq) * 100.0;
        double rm10 = 100.0 * static_cast<double>(rank_matches_10[i]) / static_cast<double>(rank_totals_10[i]);
        double rm100 = 100.0 * static_cast<double>(rank_matches_100[i]) / static_cast<double>(rank_totals_100[i]);
        printf("%-34s | %11.4f%% | %12.4f%% | %11.4f%% | %12.4f%% | %7llu | %13.3e\n",
               names[i], r10, r100, rm10, rm100,
               static_cast<unsigned long long>(dup_id_errors[i]), max_orig_dist_err[i]);

        assert(dup_id_errors[i] == 0 && "Nivel 1A fallo: Se detectaron IDs duplicados en la salida!");
        assert(r10 == 100.0 && "Nivel 1A fallo: IDSetRecall@10 debe ser 100.0000%!");
        assert(r100 == 100.0 && "Nivel 1A fallo: IDSetRecall@100 debe ser 100.0000%!");
        assert(rm10 == 100.0 && "Nivel 1B fallo: RankMatch@10 debe ser 100.0000%!");
        assert(rm100 == 100.0 && "Nivel 1B fallo: RankMatch@100 debe ser 100.0000%!");
        assert(max_orig_dist_err[i] == 0.0 && "Nivel 2 fallo: La distancia debe coincidir bit a bit en espacio original!");
    }

    uint64_t total_pruned6 = audit6.pruned_by_partition
                           + audit6.pruned_by_radial_shell
                           + audit6.pruned_by_pca_16d
                           + audit6.pruned_by_pca_32d
                           + audit6.pruned_by_pca_64d;
    uint64_t total_accounted6 = total_pruned6 + audit6.evaluated_full_64d;
    assert(audit6.contract_fallbacks == 0 && total_accounted6 == audit6.total_eligible_checked);
    assert(audit6.strict_violations_partition == 0 && audit6.strict_violations_radial == 0 &&
           audit6.strict_violations_pca16 == 0 && audit6.strict_violations_pca32 == 0 &&
           audit6.strict_violations_pca64 == 0 && audit6.false_prunings_topk == 0);
    printf("  [Plan 6 Adaptive Audit] Elegibles=%llu | Podados(Part=%llu, Rad=%llu, PCA16=%llu, PCA32=%llu, PCA64=%llu) | Violaciones=0\n",
           static_cast<unsigned long long>(audit6.total_eligible_checked),
           static_cast<unsigned long long>(audit6.pruned_by_partition),
           static_cast<unsigned long long>(audit6.pruned_by_radial_shell),
           static_cast<unsigned long long>(audit6.pruned_by_pca_16d),
           static_cast<unsigned long long>(audit6.pruned_by_pca_32d),
           static_cast<unsigned long long>(audit6.pruned_by_pca_64d));

    uint64_t total_pruned = audit.pruned_by_partition
                          + audit.pruned_by_radial_shell
                          + audit.pruned_by_pca_16d
                          + audit.pruned_by_pca_32d
                          + audit.pruned_by_pca_64d;
    uint64_t total_accounted = total_pruned + audit.evaluated_full_64d;

    printf("\n[Nivel 3 - Auditoria Unificada de las 5 Etapas de Poda en 5B (Cero Tolerancia: LB_safe^2 <= d_orig^2)]\n");
    printf("  Elegibles auditados            : %llu (Podados=%llu + EvaluadosOrig=%llu | Fallbacks=%llu)\n",
           static_cast<unsigned long long>(audit.total_eligible_checked),
           static_cast<unsigned long long>(total_pruned),
           static_cast<unsigned long long>(audit.evaluated_full_64d),
           static_cast<unsigned long long>(audit.contract_fallbacks));
    printf("  Podados por etapa              : Part=%llu | Radial=%llu | PCA16=%llu | PCA32=%llu | PCA64=%llu\n",
           static_cast<unsigned long long>(audit.pruned_by_partition),
           static_cast<unsigned long long>(audit.pruned_by_radial_shell),
           static_cast<unsigned long long>(audit.pruned_by_pca_16d),
           static_cast<unsigned long long>(audit.pruned_by_pca_32d),
           static_cast<unsigned long long>(audit.pruned_by_pca_64d));
    printf("  Violaciones estrictas (Delta>0): Part=%llu | Radial=%llu | PCA16=%llu | PCA32=%llu | PCA64=%llu\n",
           static_cast<unsigned long long>(audit.strict_violations_partition),
           static_cast<unsigned long long>(audit.strict_violations_radial),
           static_cast<unsigned long long>(audit.strict_violations_pca16),
           static_cast<unsigned long long>(audit.strict_violations_pca32),
           static_cast<unsigned long long>(audit.strict_violations_pca64));
    printf("  Eventos margen diag (0<D<=eps) : Part=%llu | Radial=%llu | PCA16=%llu | PCA32=%llu | PCA64=%llu\n",
           static_cast<unsigned long long>(audit.diag_margin_events_partition),
           static_cast<unsigned long long>(audit.diag_margin_events_radial),
           static_cast<unsigned long long>(audit.diag_margin_events_pca16),
           static_cast<unsigned long long>(audit.diag_margin_events_pca32),
           static_cast<unsigned long long>(audit.diag_margin_events_pca64));
    printf("  Violaciones > eps_diag         : Part=%llu | Radial=%llu | PCA16=%llu | PCA32=%llu | PCA64=%llu\n",
           static_cast<unsigned long long>(audit.diag_exceed_violations_partition),
           static_cast<unsigned long long>(audit.diag_exceed_violations_radial),
           static_cast<unsigned long long>(audit.diag_exceed_violations_pca16),
           static_cast<unsigned long long>(audit.diag_exceed_violations_pca32),
           static_cast<unsigned long long>(audit.diag_exceed_violations_pca64));
    printf("  Falsos descartes Top-K         : %llu\n", static_cast<unsigned long long>(audit.false_prunings_topk));
    printf("  Max exceso con signo (LB^2-d^2): %.6e (requerido <= 0.000000e+00)\n", audit.max_signed_excess);
    printf("  Max exceso positivo            : %.6e (requerido == 0.000000e+00)\n", audit.max_positive_excess);
    printf("  Min ratio d_orig^2 / LB_safe^2 : %.6f (requerido >= 1.000000)\n", audit.min_ratio_orig_over_lb);

    assert(audit.contract_fallbacks == 0 && "En las suites nominales no debe ocurrir ningun fallback por MXCSR/dominio!");
    assert(total_accounted == audit.total_eligible_checked &&
           "Contabilidad incompleta: Podados(5 etapas) + EvaluadosOrig debe igualar total_eligible_checked!");
    assert(audit.strict_violations_partition == 0 && "Violacion estricta (LB_safe^2 > d_orig^2) en poda de particion!");
    assert(audit.strict_violations_radial == 0 && "Violacion estricta (LB_safe^2 > d_orig^2) en poda radial!");
    assert(audit.strict_violations_pca16 == 0 && "Violacion estricta (LB_safe^2 > d_orig^2) en poda PCA 16D!");
    assert(audit.strict_violations_pca32 == 0 && "Violacion estricta (LB_safe^2 > d_orig^2) en poda PCA 32D!");
    assert(audit.strict_violations_pca64 == 0 && "Violacion estricta (LB_safe^2 > d_orig^2) en poda PCA 64D!");
    assert(audit.diag_margin_events_partition == 0 && "Evento en margen diagnostico en particion!");
    assert(audit.diag_margin_events_radial == 0 && "Evento en margen diagnostico en poda radial!");
    assert(audit.diag_margin_events_pca16 == 0 && "Evento en margen diagnostico en poda PCA 16D!");
    assert(audit.diag_margin_events_pca32 == 0 && "Evento en margen diagnostico en poda PCA 32D!");
    assert(audit.diag_margin_events_pca64 == 0 && "Evento en margen diagnostico en poda PCA 64D!");
    assert(audit.max_positive_excess == 0.0 && "Se registro exceso positivo (LB_safe^2 > d_orig^2)!");
    assert(audit.max_signed_excess <= 0.0 && "El maximo exceso con signo debe ser <= 0.0!");
    assert(audit.min_ratio_orig_over_lb >= 1.0 && "El ratio minimo d_orig^2 / LB_safe^2 debe ser >= 1.0!");
    assert(audit.false_prunings_topk == 0 && "Un vecino verdadero del Top-K fue podado!");
}

int main() {
    std::cout << "================================================================================\n";
    std::cout << "ALG-Flat 2.0 (Hito 2A.3): Contrato Formal de Entorno IEEE-754 y Kernel Unificado\n";
    std::cout << "  - Separacion formal: MXCSR restaurado vs Fallback seguro (<=1e18) vs Rechazo (>1e18/Corpus)\n";
    std::cout << "  - Aritmetica de intervalos en binary64 + Oraculo DoubleDouble 106-bit (Dekker + FMA)\n";
    std::cout << "  - Kernel unificado templado (AuditMode=false/true) sin duplicacion produccion/auditor\n";
    std::cout << "  - Nivel 1A/1B: IDSetRecall (sin duplicados) + RankMatch posicion a posicion\n";
    std::cout << "  - Nivel 3    : Exceso con signo Delta = LB_safe^2 - d_orig^2 <= 0.0 (0 tol)\n";
    std::cout << "================================================================================\n";

    verify_directed_rounding_and_interval_oracle();
    printf("[OK] Oraculo DoubleDouble 106-bit (u_106 = 2^-106 = 2.465e-32; ABI LDBL_MANT_DIG=%d, DBL_MANT_DIG=%d) verificado\n",
           LDBL_MANT_DIG, DBL_MANT_DIG);

    verify_runtime_environment_and_domain_contract();
    printf("[OK] Contrato 5 casos verificado: Nominal, MXCSR restaurado, Fallback (1e15..1e18], Rechazo Query (>1e18/Inf/NaN) y Rechazo Corpus\n");

    uint32_t raw_csr = 0;
    check_ieee754_mxcsr_state(&raw_csr);
    printf("Estado MXCSR activo: 0x%04X (Round=0, FTZ=0, DAZ=0) | CertMax=%.1e | FallbackMax=%.1e\n",
           raw_csr,
           static_cast<double>(CERTIFIED_MAX_COORD_ABS),
           static_cast<double>(CANONICAL_FALLBACK_MAX_COORD_ABS));
    printf("Constantes IEEE-754 derivadas (u_32 = 2^-24 = %.3e, u_64 = 2^-53 = %.3e):\n",
           IEEE754_F32_U, IEEE754_F64_U);
    printf("  gamma_17 = %.3e | gamma_33 = %.3e | gamma_64 = %.3e | gamma_66 = %.3e | gamma_67 = %.3e\n",
           static_cast<double>(GAMMA_17_F32),
           static_cast<double>(GAMMA_33_F32),
           static_cast<double>(GAMMA_64_F32),
           static_cast<double>(GAMMA_66_F32),
           static_cast<double>(GAMMA_67_F32));

    /* Suite 1: Dataset completo de benchmark (N=100,000, M50/50, sigma=0.09, P=128) */
    {
        BenchmarkDataset ds_full;
        generate_dataset_deterministic(8, 100000, 100, 64, 0.09f, MIX_50_50, ds_full);
        run_certification_suite("1_Full_Benchmark_100k_Mix_s009", ds_full, 128, false);
    }

    /* Suite 2: Centroides alejados del origen (||c_p|| >> 0, prueba de invariancia traslacional de colas) */
    {
        BenchmarkDataset ds_off;
        make_adversarial_off_origin_dataset(ds_off);
        run_certification_suite("2_Adversarial_FarOffOrigin_Centroids", ds_off, 64, false);
    }

    /* Suite 3: Colinealidad real respecto del centroide asignado c_p en colas 16..63 / 32..63 y empates */
    {
        BenchmarkDataset ds_col;
        make_adversarial_ties_and_base_dataset(ds_col);
        run_certification_suite("3_Adversarial_TrueCentroidCollinear_And_Ties", ds_col, 64, true);
    }

    /* Suite 4: Alta localidad con >92% de poda (N=100,000, R, sigma=0.03, P=128) */
    {
        BenchmarkDataset ds_tight;
        generate_dataset_deterministic(8, 100000, 100, 64, 0.03f, MIX_R, ds_tight);
        run_certification_suite("4_Full_Benchmark_100k_Tight_s003", ds_tight, 128, false);
    }

    std::cout << "\n================================================================================\n";
    std::cout << "CERTIFICACION HITO 2A.3 SUPERADA: CONTRATO IEEE-754 + KERNEL UNIFICADO VALIDADOS\n";
    std::cout << "================================================================================\n";
    return 0;
}
