#ifndef ALG_FLAT_IEEE754_BOUNDS_H
#define ALG_FLAT_IEEE754_BOUNDS_H

#include <stdint.h>
#include <cmath>

namespace alg_flat {

/*
 * ============================================================================
 * DERIVACION FORMAL DE HOLGURAS NUMERICAS IEEE-754 BINARY32 CON REDONDEO
 * DIRIGIDO CONSERVADOR Y PROFUNDIDAD POR ETAPA (HITO 2A.2)
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
 * Principio de Redondeo Dirigido (Cero Tolerancia):
 *   Todas las magnitudes intermedias se evaluan en IEEE-754 binary64 (double),
 *   absorbiendo el redondeo de las operaciones double mediante el factor
 *   (1 +- 16 * u_64), y se convierten a binary32 (float) mediante redondeo
 *   explicitamente dirigido con std::nextafterf:
 *     - Toda cota superior de error o radio (delta_vec, fp_slack_p, sq_slack_{16,32,64},
 *       dc_high, rho_orig_upper, r_pad, r_high) se redondea HACIA +INFINITO via round_up_f32().
 *     - Todo factor de escala inferior o radio minimo (scale_safe_{16,32,64}, scale_part,
 *       dc_low, r_low) se redondea HACIA CERO via round_down_f32().
 *
 * Cadena completa de errores acotados y profundidades por etapa:
 *   1. Distancia canonica en espacio original `l2_hito9_ref(q, x, 64)`:
 *      Resta por coordenada (1u) + FMA en 8 carriles + 3 sumas horizontales:
 *      Profundidad secuencial conservadora <= 66 operaciones (gamma_66):
 *        |fl_orig(||q - x||_2^2) - ||q - x||_2^2| <= gamma_66 * ||q - x||_2^2
 *
 *   2. Transformacion ortogonal almacenada en float32 (Q^T de 64x64):
 *      Defecto de ortogonalidad medido en binary64 y redondeado hacia +inf:
 *        delta_ortho = round_up_f32(||Q * Q^T - I_64||_F)
 *      Para todo v in R^64:
 *        (1 - delta_ortho) ||v||_2^2 <= ||Q^T v||_2^2 <= (1 + delta_ortho) ||v||_2^2
 *
 *   3. Vectores de base y centroides rotados en build-time (acumulados en binary64
 *      y redondeados una sola vez a binary32):
 *      - ||x_rot_p - Q^T(x - c_p)||_2 <= 2 * u_32 * r_max(p)
 *      - ||c_rot_p - Q^T(c_p - mu)||_2 <= 2 * u_32 * ||c_p - mu||_2
 *
 *   4. Rotacion de la consulta centrada en query-time y resta por particion:
 *      Cada una de las 64 filas de q_rot = fl(Q^T * fl(q - mu)) tiene error acotado
 *      por gamma_64 * (1 + u_32) * ||q - mu||_2. Sumando en norma L2 sobre las 64 filas
 *      (factor sqrt(64) = 8) y anadiendo la resta fl(q_rot - c_rot_p), la perturbacion
 *      L2 total delta_vec(q, p) entre u = (q_rot_p - x_rot_p) y w = Q^T(q - x) cumple:
 *        ||u_S - w_S||_2 <= delta_vec(q, p)   para todo S subseteq {0..63}
 *      donde:
 *        delta_vec(q, p) = round_up_f32(
 *            (8 * gamma_64 * (1 + u_32) + 4 * u_32) * ||q - mu||_2
 *            + 4 * u_32 * ||c_p - mu||_2
 *            + 2 * u_32 * r_max(p)
 *        )
 *
 *   5. Holgura lineal de normas de cola (48D y 32D):
 *      - Norma almacenada r_tail(x, p) calculada en binary64 y redondeada a float32:
 *        error <= u_32 * r_max(p).
 *      - Norma de consulta r_tail(q, p) calculada con AVX2 FMA + sqrtf:
 *        error <= gamma_64 * (dc_high + delta_vec(q, p)).
 *      - Holgura lineal redondeada hacia +infinito:
 *          fp_slack_p = round_up_f32(
 *              (1 + 4 * u_32) * (delta_vec + gamma_64 * (dc_high + delta_vec) + u_32 * r_max(p))
 *          )
 *
 *   6. Profundidades de redondeo separadas por etapa (16D, 32D y 64D):
 *      - Etapa PCA 16D (S = 16):
 *        Resta por coordenada (1u) + cuadrado/FMA de 2 vectores AVX2 + reduccion hadd/hsum:
 *        profundidad secuencial <= 17 operaciones -> factor gamma_17.
 *      - Etapa PCA 32D (S = 32):
 *        Suma de 2 bloques de 16D (d16 + l2_16_avx2) sobre 32 coordenadas + resta:
 *        profundidad secuencial <= 33 operaciones -> factor gamma_33.
 *      - Etapa PCA 64D (S = 64):
 *        Suma completa de las 64 coordenadas rotadas (d32 + 2 bloques finales de 16D):
 *        profundidad secuencial <= 67 operaciones -> factor gamma_67.
 *
 *      Para cada etapa s in {16, 32, 64} con factor gamma_{n(s)} (n(16)=17, n(32)=33, n(64)=67):
 *        - Sea M_{q,p} = sqrt(1 + delta_ortho) * (dc_high + r_max(p)) + delta_vec(q, p).
 *        - La holgura cuadratica del prefijo de s dimensiones se redondea HACIA +INFINITO:
 *            sq_slack_{s,p} = round_up_f32( 2.0 * (1.0 + gamma_{n(s)}) * delta_vec(q, p) * M_{q,p} )
 *        - El factor de escala hacia fl_orig(||q - x||_2^2), que absorbe ademas las
 *          operaciones escalares finales en float32 de la cota (factor (1 - 6 * u_32)),
 *          se redondea HACIA CERO:
 *            scale_safe_s = round_down_f32(
 *                ((1.0 - gamma_66) * (1.0 - gamma_{n(s)}) * (1.0 - 6.0 * u_32)) / (1.0 + delta_ortho)
 *            )
 * ============================================================================
 */

static constexpr double IEEE754_F32_U = 5.9604644775390625e-8; /* 2^-24 exacto */
static constexpr double IEEE754_F64_U = 1.1102230246251565e-16; /* 2^-53 exacto */

constexpr double ieee754_gamma(uint32_t n) {
    return (static_cast<double>(n) * IEEE754_F32_U) / (1.0 - static_cast<double>(n) * IEEE754_F32_U);
}

/*
 * Redondeo dirigido hacia +infinito (cota superior estricta en float32):
 * Amplifica por (1 + 16 * u_64) para cubrir el redondeo intermedio en double y
 * garantiza mediante std::nextafterf que static_cast<double>(f) >= x_up > x.
 */
inline float round_up_f32(double x) {
    if (x <= 0.0) return 0.0f;
    double x_up = x * (1.0 + 16.0 * IEEE754_F64_U);
    float f = static_cast<float>(x_up);
    if (static_cast<double>(f) < x_up) {
        f = std::nextafterf(f, INFINITY);
    }
    return f;
}

/*
 * Redondeo dirigido hacia 0 (cota inferior estricta en float32 para magnitudes >= 0):
 * Reduce por (1 - 16 * u_64) para cubrir el redondeo intermedio en double y
 * garantiza mediante std::nextafterf que 0 <= static_cast<double>(f) <= x_down < x.
 */
inline float round_down_f32(double x) {
    if (x <= 0.0) return 0.0f;
    double x_down = x * (1.0 - 16.0 * IEEE754_F64_U);
    float f = static_cast<float>(x_down);
    if (static_cast<double>(f) > x_down) {
        f = std::nextafterf(f, 0.0f);
    }
    return f;
}

static constexpr float GAMMA_17_F32 = static_cast<float>(ieee754_gamma(17));
static constexpr float GAMMA_33_F32 = static_cast<float>(ieee754_gamma(33));
static constexpr float GAMMA_64_F32 = static_cast<float>(ieee754_gamma(64));
static constexpr float GAMMA_66_F32 = static_cast<float>(ieee754_gamma(66));
static constexpr float GAMMA_67_F32 = static_cast<float>(ieee754_gamma(67));
static constexpr float U_F32        = static_cast<float>(IEEE754_F32_U);

/* Coeficientes exactos en double para delta_vec(q, p) */
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

    /* Paso 4: Perturbacion L2 maxima delta_vec_p redondeada hacia +inf */
    const double delta_vec_d = ROT64_L2_COEFF_Q_F64 * q_norm
                             + ROT64_L2_COEFF_C_F64 * c_norm
                             + ROT64_L2_COEFF_X_F64 * r_max;
    const float delta_vec_p  = round_up_f32(delta_vec_d);
    const double dv          = static_cast<double>(delta_vec_p);

    /* Paso 5: Holgura lineal de las normas de cola 48D y 32D redondeada hacia +inf */
    const double tail_norm_err_d = ieee754_gamma(64) * (dc_hi + dv) + IEEE754_F32_U * r_max;
    const double fp_slack_d      = (1.0 + 4.0 * IEEE754_F32_U) * (dv + tail_norm_err_d);
    const float fp_slack_p       = round_up_f32(fp_slack_d);

    /* Paso 6: Holguras cuadraticas por etapa (16D -> gamma_17, 32D -> gamma_33, 64D -> gamma_67)
     *         redondeadas hacia +inf */
    const double max_vec_norm = std::sqrt(1.0 + def) * (dc_hi + r_max) + dv;
    const double base_sq_err  = 2.0 * dv * max_vec_norm;
    const float sq_slack_16_p = round_up_f32(base_sq_err * (1.0 + ieee754_gamma(17)));
    const float sq_slack_32_p = round_up_f32(base_sq_err * (1.0 + ieee754_gamma(33)));
    const float sq_slack_64_p = round_up_f32(base_sq_err * (1.0 + ieee754_gamma(67)));

    /* Factores de escala inferiores hacia fl_orig(||q - x||_2^2) redondeados hacia 0 */
    const double den = 1.0 + def;
    const double common_num = (1.0 - ieee754_gamma(66)) * (1.0 - 6.0 * IEEE754_F32_U);
    const float scale_safe_16 = round_down_f32((common_num * (1.0 - ieee754_gamma(17))) / den);
    const float scale_safe_32 = round_down_f32((common_num * (1.0 - ieee754_gamma(33))) / den);
    const float scale_safe_64 = round_down_f32((common_num * (1.0 - ieee754_gamma(67))) / den);

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
 * Calcula de forma conservadora (con redondeo dirigido) las distancias al centroide
 * dc_low (hacia 0), dc_high (hacia +inf) y la cota inferior de particion lb_part_sq (hacia 0).
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
    const double dc_d = std::sqrt(static_cast<double>(d2_f32));
    /* l2_sq_avx2_1x_aos tiene profundidad secuencial <= 66 -> gamma_66 */
    const float dc_low  = round_down_f32(dc_d * std::sqrt(1.0 - ieee754_gamma(66)));
    const float dc_high = round_up_f32(dc_d / std::sqrt(1.0 - ieee754_gamma(66)));

    double raw_lb = 0.0;
    if (static_cast<double>(dc_low) > static_cast<double>(max_radius)) {
        raw_lb = static_cast<double>(dc_low) - static_cast<double>(max_radius);
    } else if (static_cast<double>(dc_high) < static_cast<double>(min_radius)) {
        raw_lb = static_cast<double>(min_radius) - static_cast<double>(dc_high);
    }
    const double lb_sq_d = (raw_lb * raw_lb) * (1.0 - ieee754_gamma(66));
    const float lb_part_sq = round_down_f32(lb_sq_d);

    return {static_cast<float>(dc_d), dc_low, dc_high, lb_part_sq};
}

/*
 * Calcula el intervalo radial conservador [r_low, r_high] fuera del cual todo vector x
 * cumple fl_orig(||q - x||_2^2) > threshold_sq:
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
    const double rho_upper_d = std::sqrt(static_cast<double>(threshold_sq) / (1.0 - ieee754_gamma(66)));
    const float rho_upper = round_up_f32(rho_upper_d);
    const float r_pad = round_up_f32(IEEE754_F32_U * static_cast<double>(max_radius));

    const double r_low_d  = static_cast<double>(dc_low) - static_cast<double>(rho_upper) - static_cast<double>(r_pad);
    const double r_high_d = static_cast<double>(dc_high) + static_cast<double>(rho_upper) + static_cast<double>(r_pad);

    out_r_low  = (r_low_d > 0.0) ? round_down_f32(r_low_d) : 0.0f;
    out_r_high = round_up_f32(r_high_d);
    out_r_pad  = r_pad;
}

} // namespace alg_flat

#endif // ALG_FLAT_IEEE754_BOUNDS_H
