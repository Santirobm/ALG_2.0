/*
 * Copyright (c) 2026 Santiago Robledo.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef ALG_FLAT_IEEE754_BOUNDS_H
#define ALG_FLAT_IEEE754_BOUNDS_H

#include "alg_flat_types.h"
#include <stdint.h>
#include <string.h>
#include <cmath>
#include <algorithm>

namespace alg_flat {

/*
 * ============================================================================
 * CONTRATO FORMAL DE EJECUCION IEEE-754, ARITMETICA DE INTERVALOS EN BINARY64
 * Y DERIVACION DE HOLGURAS POR ETAPA (HITO 2A.3)
 * ============================================================================
 * Referencia matematica:
 *   N. J. Higham, "Accuracy and Stability of Numerical Algorithms", 2nd Ed., SIAM.
 *   - Redondeo unitario en IEEE-754 binary32 (float):
 *       u_32 = 2^-24 = 5.9604644775390625e-8
 *   - Redondeo unitario en IEEE-754 binary64 (double):
 *       u_64 = 2^-53 = 1.1102230246251565e-16
 *   - Constante estandar de acumulacion de redondeo de Higham:
 *       gamma_n = (n * u_32) / (1.0 - n * u_32)
 *
 * ============================================================================
 * SECCION 1: CONTRATO DE ENTORNO IEEE-754 (MXCSR) Y DOMINIO DE ENTRADA ADMITIDO
 * ============================================================================
 * Las cotas inferiores certificadas de los Planes 5A y 5B son teoremas condicionales
 * a dos precondiciones verificadas en tiempo de construccion y en cada consulta:
 *
 *   1. Estado del registro SIMD MXCSR (x86_64 SSE/AVX2):
 *      - Modo de redondeo (bits 13..14, mask 0x6000): DEBE ser Round-to-Nearest-Even (0x0000).
 *      - Flush-To-Zero (FTZ, bit 15, mask 0x8000): DEBE estar desactivado (0).
 *      - Denormals-Are-Zero (DAZ, bit 6, mask 0x0040): DEBE estar desactivado (0).
 *      Si otro hilo o libreria host modifica MXCSR activando FTZ/DAZ o cambiando el
 *      modo de redondeo, `check_ieee754_mxcsr_state()` lo detecta en O(1) (~1 ciclo)
 *      y los planes 5A/5B desactivan inmediatamente toda poda geometrica y conmutan a
 *      escaneo exacto sin poda (`fallback_triggered = 1`).
 *
 *   2. Dominio numerico de coordenadas admitido (`CERTIFIED_MAX_COORD_ABS = 1.0e15f`):
 *      - Toda coordenada c de cualquier vector del corpus x y de cualquier consulta q
 *        debe ser finita (`std::isfinite(c)`) y cumplir |c| <= 1.0e15f.
 *      - Justificacion de ausencia de overflow:
 *        Para D = 64 y |x_d|, |q_d| <= 1.0e15f, la diferencia maxima por coordenada es
 *        |q_d - x_d| <= 2.0e15f, su cuadrado es <= 4.0e30f, y la suma en 64 dimensiones
 *        es <= 64 * 4.0e30f = 2.56e32f << FLT_MAX (3.4028235e38f). Tampoco hay overflow
 *        en la rotacion ortogonal Q^T (64x64).
 *      - Justificacion de terminos subnormales:
 *        Con FTZ=0 y DAZ=0 verificados en MXCSR, el hardware IEEE-754 ejecuta subnormales
 *        graduales exactos en sumas/restas (`fl(a - b) = a - b` cuando a - b es subnormal)
 *        y cualquier underflow en un producto cuadrado individual (|diff| < 1.175e-19f)
 *        aporta a lo sumo 2^-149 (~1.4e-45f) por coordenada, que queda integramente
 *        absorbido por la constante aditiva de seguridad subnormal `SUBNORMAL_SQ_SLACK_F64`
 *        sumada en `sq_slack_{16,32,64}_p`.
 *
 * ============================================================================
 * SECCION 2: PRESUPUESTO DE ERROR EN BINARY64 E INTERVALOS CON REDONDEO HACIA AFUERA
 * ============================================================================
 * Para garantizar que ninguna cantidad intermedia en `double` (`binary64`) quede
 * subestimada (si es cota superior) o sobrestimada (si es cota inferior) antes de
 * convertirla a `float32`, utilizamos:
 *   1. Primitivas elementales de intervalo en `double` (`add_up_f64`, `sub_down_f64`,
 *      `mul_up_f64`, `mul_down_f64`, `div_up_f64`, `div_down_f64`, `sqrt_up_f64`,
 *      `sqrt_down_f64`) implementadas mediante incremento/decremento directo de 1 ULP
 *      en la representacion IEEE-754 `uint64_t` (`next_ulp_up_pos_f64` / `next_ulp_down_pos_f64`),
 *      sin llamadas a libreria C (`nextafter`/`nextafterf` de `msvcrt.dll`).
 *      Dado que cada operacion elemental en IEEE-754 `binary64` tiene error <= 0.5 ULP
 *      <= 1 ULP, avanzar 1 ULP en la direccion conservadora acota de forma exacta cada
 *      operacion individual en ~1 ciclo de CPU.
 *   2. Ademas, `round_up_f32(double x)` y `round_down_f32(double x)` aplican un margen
 *      adicional de `16 * u_64` y corrigen la conversion a `float32` avanzando 1 ULP
 *      en la representacion `uint32_t` cuando es necesario.
 *      Presupuesto explicito de operaciones `binary64` por expresion:
 *        - `gamma_up_f64(n)` / `one_minus_gamma_down_f64(n)`: 3 ops -> <= 3 u_64.
 *        - `delta_vec_p`: 3 productos + 2 sumas = 5 ops (profundidad 3) -> <= 4 u_64.
 *        - `fp_slack_p`: 6 ops (profundidad 5) -> <= 6 u_64.
 *        - `sq_slack_{16,32,64}_p`: 7 ops (profundidad 6) -> <= 7 u_64.
 *        - `scale_safe_{16,32,64}`: 6 ops (profundidad 4) -> <= 5 u_64.
 *        - `dc_low`, `dc_high`, `lb_part_sq`: <= 5 ops -> <= 5 u_64.
 *        - `rho_upper`, `r_pad`, `r_low`, `r_high`: <= 4 ops -> <= 4 u_64.
 *      En todos los casos, la profundidad de arbol en `binary64` es <= 7 operaciones
 *      (<< 16 * u_64), y ademas cada paso intermedio usa redondeo hacia afuera a 1 ULP.
 * ============================================================================
 */

static constexpr double IEEE754_F32_U = 5.9604644775390625e-8;  /* 2^-24 exacto */
static constexpr double IEEE754_F64_U = 1.1102230246251565e-16; /* 2^-53 exacto */

/* Dominio maximo por coordenada para certificar la poda geometrica (PCA 64D + cotas) sin overflow */
static constexpr float CERTIFIED_MAX_COORD_ABS = 1.0e15f;

/*
 * Dominio maximo por coordenada de consulta para que el fallback canonico `ExactScan` (`l2_hito9_ref`)
 * frente a un corpus certificado (|x_d| <= 1.0e15f) sea garantizadamente finito y libre de overflow en float32:
 *   64 * (1.0e18 + 1.0e15)^2 * (1 + gamma_66) = 6.413e37 << FLT_MAX (3.4028235e38).
 * Si alguna coordenada de consulta excede CANONICAL_FALLBACK_MAX_COORD_ABS o es NaN/Inf,
 * incluso `ExactScan` puede desbordar a `Inf` o `NaN` y romper el orden total del heap, por lo que
 * la consulta debe ser rechazada explicitamente.
 */
static constexpr float CANONICAL_FALLBACK_MAX_COORD_ABS = 1.0e18f;

/* Holgura aditiva para cubrir cualquier posible underflow en cuadrados subnormales en 64D */
static constexpr double SUBNORMAL_SQ_SLACK_F64 = 64.0 * 1.401298464324817e-45; /* 64 * FLT_TRUE_MIN */

/* Bits de control relevantes en MXCSR (x86_64 SSE/AVX2) */
static constexpr uint32_t MXCSR_ROUND_MASK = 0x6000U; /* Bits 13..14: 0 = Round-to-Nearest-Even */
static constexpr uint32_t MXCSR_FTZ_BIT    = 0x8000U; /* Bit 15: Flush-To-Zero */
static constexpr uint32_t MXCSR_DAZ_BIT    = 0x0040U; /* Bit 6:  Denormals-Are-Zero */
static constexpr uint32_t MXCSR_CERTIFIED_MASK = MXCSR_ROUND_MASK | MXCSR_FTZ_BIT | MXCSR_DAZ_BIT;

/*
 * Verifica en O(1) (~1 ciclo de reloj) que el registro MXCSR del hilo actual
 * mantenga Round-to-Nearest-Even (0), FTZ=0 y DAZ=0.
 */
inline bool check_ieee754_mxcsr_state(uint32_t* out_raw_mxcsr = nullptr) {
    uint32_t csr = _mm_getcsr();
    if (out_raw_mxcsr) *out_raw_mxcsr = csr;
    return (csr & MXCSR_CERTIFIED_MASK) == 0U;
}

/*
 * Guardia RAII que establece temporalmente un entorno MXCSR canonico (Round-to-Nearest-Even,
 * FTZ=0, DAZ=0) durante la ejecucion del fallback `ExactScan` y restaura el estado original
 * del hilo al salir del ambito.
 */
struct CanonicalMxcsrScope {
    uint32_t saved_csr;
    bool modified;

    inline CanonicalMxcsrScope() {
        saved_csr = _mm_getcsr();
        if ((saved_csr & MXCSR_CERTIFIED_MASK) != 0U) {
            uint32_t clean_csr = (saved_csr & ~MXCSR_CERTIFIED_MASK);
            _mm_setcsr(clean_csr);
            modified = true;
        } else {
            modified = false;
        }
    }

    inline ~CanonicalMxcsrScope() {
        if (modified) {
            _mm_setcsr(saved_csr);
        }
    }
};

/*
 * Verifica con AVX2 que todas las coordenadas de un buffer de `count` floats sean
 * finitas y cumplan |v[i]| <= max_abs.
 */
inline bool check_vector_domain_bounded(const float* v, size_t count, float max_abs) {
    if (!v) return false;
    size_t i = 0;
    const __m256 v_sign_mask = _mm256_set1_ps(-0.0f);
    const __m256 v_max_abs   = _mm256_set1_ps(max_abs);
    for (; i + 8 <= count; i += 8) {
        __m256 x = _mm256_loadu_ps(v + i);
        __m256 abs_x = _mm256_andnot_ps(v_sign_mask, x);
        /* _CMP_LE_OQ devuelve false (0) si abs_x > max_abs o si x es NaN/Inf */
        __m256 ok = _mm256_cmp_ps(abs_x, v_max_abs, _CMP_LE_OQ);
        if (_mm256_movemask_ps(ok) != 0xFF) return false;
    }
    for (; i < count; ++i) {
        float x = v[i];
        if (!std::isfinite(x)) return false;
        if (std::fabs(x) > max_abs) return false;
    }
    return true;
}

inline bool check_vector_domain_certified(const float* v, size_t count) {
    return check_vector_domain_bounded(v, count, CERTIFIED_MAX_COORD_ABS);
}

inline bool check_query_canonical_fallback_safe(const float* q, size_t count) {
    return check_vector_domain_bounded(q, count, CANONICAL_FALLBACK_MAX_COORD_ABS);
}

/*
 * Manipulacion exacta de 1 ULP en IEEE-754 binary64 y binary32 para magnitudes >= 0
 * libre de llamadas a msvcrt.dll (1 instruccion entera tras bit-cast).
 */
inline double next_ulp_up_pos_f64(double x) {
    if (x < 0.0) return 0.0;
    if (x == 0.0) {
        uint64_t one = 1ULL;
        double res;
        memcpy(&res, &one, sizeof(double));
        return res;
    }
    uint64_t bits;
    memcpy(&bits, &x, sizeof(double));
    if ((bits & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL) return x;
    bits++;
    double res;
    memcpy(&res, &bits, sizeof(double));
    return res;
}

inline double next_ulp_down_pos_f64(double x) {
    if (x <= 0.0) return 0.0;
    uint64_t bits;
    memcpy(&bits, &x, sizeof(double));
    if ((bits & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL) return x;
    bits--;
    double res;
    memcpy(&res, &bits, sizeof(double));
    return res;
}

inline float next_ulp_up_pos_f32(float x) {
    if (x < 0.0f) return 0.0f;
    if (x == 0.0f) {
        uint32_t one = 1U;
        float res;
        memcpy(&res, &one, sizeof(float));
        return res;
    }
    uint32_t bits;
    memcpy(&bits, &x, sizeof(float));
    if ((bits & 0x7F800000U) == 0x7F800000U) return x;
    bits++;
    float res;
    memcpy(&res, &bits, sizeof(float));
    return res;
}

inline float next_ulp_down_pos_f32(float x) {
    if (x <= 0.0f) return 0.0f;
    uint32_t bits;
    memcpy(&bits, &x, sizeof(float));
    if ((bits & 0x7F800000U) == 0x7F800000U) return x;
    bits--;
    float res;
    memcpy(&res, &bits, sizeof(float));
    return res;
}

/*
 * Primitivas de aritmetica de intervalos en binary64 (redondeo hacia afuera exacto a 1 ULP
 * para cantidades no negativas):
 */
inline double add_up_f64(double a, double b) {
    return next_ulp_up_pos_f64(a + b);
}
inline double sub_down_f64(double a, double b) {
    double d = a - b;
    return (d > 0.0) ? next_ulp_down_pos_f64(d) : 0.0;
}
inline double mul_up_f64(double a, double b) {
    if (a <= 0.0 || b <= 0.0) return 0.0;
    return next_ulp_up_pos_f64(a * b);
}
inline double mul_down_f64(double a, double b) {
    if (a <= 0.0 || b <= 0.0) return 0.0;
    return next_ulp_down_pos_f64(a * b);
}
inline double div_up_f64(double a, double b) {
    if (a <= 0.0) return 0.0;
    return next_ulp_up_pos_f64(a / b);
}
inline double div_down_f64(double a, double b) {
    if (a <= 0.0) return 0.0;
    return next_ulp_down_pos_f64(a / b);
}
inline double sqrt_up_f64(double a) {
    if (a <= 0.0) return 0.0;
    return next_ulp_up_pos_f64(std::sqrt(a));
}
inline double sqrt_down_f64(double a) {
    if (a <= 0.0) return 0.0;
    return next_ulp_down_pos_f64(std::sqrt(a));
}

constexpr double ieee754_gamma(uint32_t n) {
    return (static_cast<double>(n) * IEEE754_F32_U) / (1.0 - static_cast<double>(n) * IEEE754_F32_U);
}

/* Cota superior rigurosa de gamma_n en binary64 (redondeada hacia +inf) */
inline double gamma_up_f64(uint32_t n) {
    double num = mul_up_f64(static_cast<double>(n), IEEE754_F32_U);
    double den = sub_down_f64(1.0, num);
    return div_up_f64(num, den);
}

/* Cota inferior rigurosa de (1 - gamma_n) en binary64 (redondeada hacia 0) */
inline double one_minus_gamma_down_f64(uint32_t n) {
    return sub_down_f64(1.0, gamma_up_f64(n));
}

/*
 * Conversion dirigida hacia +infinito de binary64 a binary32 (float32):
 * Avanza 1 ULP en binary64 y garantiza mediante next_ulp_up_pos_f32 que
 * static_cast<double>(f) >= x_up > x (sin constantes heuristicas adicionales,
 * dado que cada operacion intermedia ya usa aritmetica de intervalos hacia afuera).
 */
inline float round_up_f32(double x) {
    if (x <= 0.0) return 0.0f;
    double x_up = next_ulp_up_pos_f64(x);
    float f = static_cast<float>(x_up);
    if (static_cast<double>(f) < x_up) {
        f = next_ulp_up_pos_f32(f);
    }
    return f;
}

/*
 * Conversion dirigida hacia 0 de binary64 a binary32 (float32) para magnitudes >= 0:
 * Retrocede 1 ULP en binary64 y garantiza mediante next_ulp_down_pos_f32 que
 * 0 <= static_cast<double>(f) <= x_down < x.
 */
inline float round_down_f32(double x) {
    if (x <= 0.0) return 0.0f;
    double x_down = next_ulp_down_pos_f64(x);
    float f = static_cast<float>(x_down);
    if (static_cast<double>(f) > x_down) {
        f = next_ulp_down_pos_f32(f);
    }
    return f;
}

static constexpr float GAMMA_17_F32 = static_cast<float>(ieee754_gamma(17));
static constexpr float GAMMA_33_F32 = static_cast<float>(ieee754_gamma(33));
static constexpr float GAMMA_64_F32 = static_cast<float>(ieee754_gamma(64));
static constexpr float GAMMA_66_F32 = static_cast<float>(ieee754_gamma(66));
static constexpr float GAMMA_67_F32 = static_cast<float>(ieee754_gamma(67));
static constexpr float U_F32        = static_cast<float>(IEEE754_F32_U);

/* Coeficientes nominales en double para delta_vec(q, p) */
static constexpr double ROT64_L2_COEFF_Q_F64 =
    8.0 * ieee754_gamma(64) * (1.0 + IEEE754_F32_U) + 4.0 * IEEE754_F32_U;
static constexpr double ROT64_L2_COEFF_C_F64 = 4.0 * IEEE754_F32_U;
static constexpr double ROT64_L2_COEFF_X_F64 = 2.0 * IEEE754_F32_U;

/* Compuerta conservadora para diferencias entre arboles de suma SIMD en 64D */
static constexpr float SIMD_TREE_GATE_REL_EPS = static_cast<float>(2.0 * ieee754_gamma(67));

struct PartitionCertifiedBounds {
    float delta_vec_p;      /* Redondeado hacia +inf */
    float fp_slack_p;       /* Redondeado hacia +inf */
    float sq_slack_16_p;    /* Etapa 16D (gamma_17), redondeado hacia +inf */
    float sq_slack_32_p;    /* Etapa 32D (gamma_33), redondeado hacia +inf */
    float sq_slack_64_p;    /* Etapa 64D (gamma_67), redondeado hacia +inf */
    float scale_safe_16;    /* Etapa 16D (gamma_17), redondeado hacia 0 */
    float scale_safe_32;    /* Etapa 32D (gamma_33), redondeado hacia 0 */
    float scale_safe_64;    /* Etapa 64D (gamma_67), redondeado hacia 0 */
};

/*
 * Calcula las cotas y escalas certificadas de una particion utilizando aritmetica de
 * intervalos con redondeo hacia afuera en binary64 (presupuesto <= 7 operaciones double)
 * y conversion dirigida a float32.
 */
inline PartitionCertifiedBounds compute_partition_certified_bounds(float q_mean_norm,
                                                                   float c_mean_norm,
                                                                   float dc_high,
                                                                   float r_max_p,
                                                                   float ortho_defect) {
    const double q_norm = static_cast<double>(q_mean_norm);
    const double c_norm = static_cast<double>(c_mean_norm);
    const double dc_hi  = static_cast<double>(dc_high);
    const double r_max  = static_cast<double>(r_max_p);
    const double def    = static_cast<double>(ortho_defect);

    /* Coeficientes con redondeo hacia +inf en binary64 (3 ops) */
    const double g64_up      = gamma_up_f64(64);
    const double coeff_q_up  = add_up_f64(mul_up_f64(8.0, mul_up_f64(g64_up, add_up_f64(1.0, IEEE754_F32_U))),
                                          mul_up_f64(4.0, IEEE754_F32_U));
    const double coeff_c_up  = mul_up_f64(4.0, IEEE754_F32_U);
    const double coeff_x_up  = mul_up_f64(2.0, IEEE754_F32_U);

    /* Paso 4: Perturbacion L2 maxima delta_vec_p (5 ops en binary64 -> redondeo hacia +inf) */
    const double term_q      = mul_up_f64(coeff_q_up, q_norm);
    const double term_c      = mul_up_f64(coeff_c_up, c_norm);
    const double term_x      = mul_up_f64(coeff_x_up, r_max);
    const double delta_vec_d = add_up_f64(add_up_f64(term_q, term_c), term_x);
    const float delta_vec_p  = round_up_f32(delta_vec_d);
    const double dv          = static_cast<double>(delta_vec_p);

    /* Paso 5: Holgura lineal de las normas de cola 48D y 32D (6 ops en binary64 -> hacia +inf) */
    const double tail_norm_err_d = add_up_f64(mul_up_f64(g64_up, add_up_f64(dc_hi, dv)),
                                              mul_up_f64(IEEE754_F32_U, r_max));
    const double fp_slack_d      = mul_up_f64(add_up_f64(1.0, mul_up_f64(4.0, IEEE754_F32_U)),
                                              add_up_f64(dv, tail_norm_err_d));
    const float fp_slack_p       = round_up_f32(fp_slack_d);

    /* Paso 6: Holguras cuadraticas por etapa (7 ops en binary64 + SUBNORMAL_SQ_SLACK_F64 -> hacia +inf) */
    const double max_vec_norm = add_up_f64(mul_up_f64(sqrt_up_f64(add_up_f64(1.0, def)),
                                                      add_up_f64(dc_hi, r_max)),
                                           dv);
    const double base_sq_err  = mul_up_f64(2.0, mul_up_f64(dv, max_vec_norm));
    const float sq_slack_16_p = round_up_f32(add_up_f64(mul_up_f64(base_sq_err, add_up_f64(1.0, gamma_up_f64(17))),
                                                        SUBNORMAL_SQ_SLACK_F64));
    const float sq_slack_32_p = round_up_f32(add_up_f64(mul_up_f64(base_sq_err, add_up_f64(1.0, gamma_up_f64(33))),
                                                        SUBNORMAL_SQ_SLACK_F64));
    const float sq_slack_64_p = round_up_f32(add_up_f64(mul_up_f64(base_sq_err, add_up_f64(1.0, gamma_up_f64(67))),
                                                        SUBNORMAL_SQ_SLACK_F64));

    /* Factores de escala inferiores hacia fl_orig(||q - x||_2^2) (6 ops en binary64 -> hacia 0) */
    const double den_up      = add_up_f64(1.0, def);
    const double scalar_down = sub_down_f64(1.0, mul_up_f64(6.0, IEEE754_F32_U));
    const double common_down = mul_down_f64(one_minus_gamma_down_f64(66), scalar_down);
    const float scale_safe_16 = round_down_f32(div_down_f64(mul_down_f64(common_down, one_minus_gamma_down_f64(17)), den_up));
    const float scale_safe_32 = round_down_f32(div_down_f64(mul_down_f64(common_down, one_minus_gamma_down_f64(33)), den_up));
    const float scale_safe_64 = round_down_f32(div_down_f64(mul_down_f64(common_down, one_minus_gamma_down_f64(67)), den_up));

    return {
        delta_vec_p,
        fp_slack_p,
        sq_slack_16_p,
        sq_slack_32_p,
        sq_slack_64_p,
        scale_safe_16,
        scale_safe_32,
        scale_safe_64
    };
}

/*
 * Calcula de forma conservadora (con redondeo hacia afuera en binary64 y conversion dirigida)
 * las distancias al centroide dc_low (hacia 0), dc_high (hacia +inf) y la cota inferior de
 * particion lb_part_sq (hacia 0).
 */
struct PartitionDistanceBounds {
    float dc;
    float dc_low;
    float dc_high;
    float lb_part_sq;
};

inline PartitionDistanceBounds compute_partition_distance_bounds(float d2_f32,
                                                                 float min_radius,
                                                                 float max_radius) {
    const double d2_d = static_cast<double>(d2_f32);
    const double dc_d = std::sqrt(d2_d);
    const double one_m_g66_down = one_minus_gamma_down_f64(66);

    const float dc_low  = round_down_f32(mul_down_f64(sqrt_down_f64(d2_d), sqrt_down_f64(one_m_g66_down)));
    const float dc_high = round_up_f32(div_up_f64(sqrt_up_f64(d2_d), sqrt_down_f64(one_m_g66_down)));

    double raw_lb = 0.0;
    if (static_cast<double>(dc_low) > static_cast<double>(max_radius)) {
        raw_lb = sub_down_f64(static_cast<double>(dc_low), static_cast<double>(max_radius));
    } else if (static_cast<double>(dc_high) < static_cast<double>(min_radius)) {
        raw_lb = sub_down_f64(static_cast<double>(min_radius), static_cast<double>(dc_high));
    }
    const double lb_sq_d = mul_down_f64(mul_down_f64(raw_lb, raw_lb), one_m_g66_down);
    const float lb_part_sq = round_down_f32(lb_sq_d);

    return {static_cast<float>(dc_d), dc_low, dc_high, lb_part_sq};
}

/*
 * Calcula el intervalo radial conservador [r_low, r_high] con aritmetica de intervalos en binary64:
 *   - r_low se redondea hacia 0 (round_down_f32)
 *   - r_high se redondea hacia +inf (round_up_f32)
 */
inline void compute_radial_shell_interval(float dc_low,
                                          float dc_high,
                                          float max_radius,
                                          float threshold_sq,
                                          float& out_r_low,
                                          float& out_r_high,
                                          float& out_r_pad) {
    const double one_m_g66_down = one_minus_gamma_down_f64(66);
    const double rho_upper_d = sqrt_up_f64(div_up_f64(static_cast<double>(threshold_sq), one_m_g66_down));
    const float rho_upper = round_up_f32(rho_upper_d);
    const float r_pad = round_up_f32(mul_up_f64(IEEE754_F32_U, static_cast<double>(max_radius)));

    const double r_low_d  = sub_down_f64(sub_down_f64(static_cast<double>(dc_low), static_cast<double>(rho_upper)),
                                         static_cast<double>(r_pad));
    const double r_high_d = add_up_f64(add_up_f64(static_cast<double>(dc_high), static_cast<double>(rho_upper)),
                                       static_cast<double>(r_pad));

    out_r_low  = (r_low_d > 0.0) ? round_down_f32(r_low_d) : 0.0f;
    out_r_high = round_up_f32(r_high_d);
    out_r_pad  = r_pad;
}

} // namespace alg_flat

#endif // ALG_FLAT_IEEE754_BOUNDS_H
