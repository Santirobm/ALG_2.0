#ifndef ALG_FLAT_IEEE754_BOUNDS_H
#define ALG_FLAT_IEEE754_BOUNDS_H

#include <stdint.h>
#include <math.h>

namespace alg_flat {

/*
 * ============================================================================
 * DERIVACION FORMAL DE HOLGURAS NUMERICAS IEEE-754 BINARY32 (HITO 2A.1)
 * ============================================================================
 * Referencia matematica:
 *   N. J. Higham, "Accuracy and Stability of Numerical Algorithms", 2nd Ed., SIAM.
 *   - Redondeo unitario en IEEE-754 binary32 (float):
 *       u = 2^-24 = 5.9604644775390625e-8f
 *   - Constante estandar de acumulacion de redondeo de Higham:
 *       gamma_n = (n * u) / (1.0 - n * u)
 *
 * Hipotesis de rango numerico:
 *   - Dimension D = 64.
 *   - Coordenadas finitas en float32 tales que ||q - x||_2^2 no sufre overflow (> 1e38)
 *     ni underflow subnormal (< 1e-38).
 *
 * Cadena completa de errores acotados:
 *   1. Distancia canonica en espacio original `l2_hito9_ref(q, x, 64)`:
 *      Resta por coordenada (1u) + FMA en 8 registros de 8 pasos + 3 sumas horizontales:
 *      Profundidad secuencial conservadora <= 66 operaciones:
 *        |fl_orig(||q - x||_2^2) - ||q - x||_2^2| <= gamma_66 * ||q - x||_2^2
 *        |fl(sqrt(fl_orig(||q - x||_2^2))) - ||q - x||_2| <= gamma_66 * ||q - x||_2
 *
 *   2. Transformacion ortogonal almacenada en float32 (Q^T de 64x64):
 *      - Defecto de ortogonalidad medido exactamente en binary64 al construir el indice:
 *          delta_ortho = ||Q * Q^T - I_64||_F
 *        Para todo vector v in R^64:
 *          (1 - delta_ortho) ||v||_2^2 <= ||Q^T v||_2^2 <= (1 + delta_ortho) ||v||_2^2
 *
 *   3. Vectores de base y centroides rotados en build-time (acumulados en binary64
 *      y redondeados una sola vez a binary32):
 *      - x_rot_p = fl_32(Q^T * (x - c_p)):
 *          ||x_rot_p - Q^T(x - c_p)||_2 <= u * sqrt(1 + delta_ortho) * ||x - c_p||_2
 *                                       <= 2u * r_max(p)
 *      - c_rot_p = fl_32(Q^T * (c_p - mu)):
 *          ||c_rot_p - Q^T(c_p - mu)||_2 <= 2u * ||c_p - mu||_2
 *
 *   4. Rotacion de la consulta centrada en query-time (AVX2 doble acumulador FMA):
 *      - Para cada una de las 64 filas de Q^T, el producto interno de 64 elementos con
 *        doble acumulador AVX2 (bloques de 16 -> arbol de reduccion) tiene profundidad
 *        de redondeo <= 16 pasos (cota conservadora secuencial por fila: gamma_64).
 *      - Por el Teorema 3.1 de Higham, el error en cada fila r cumple:
 *          |e_r| <= gamma_64 * ||row_r||_2 * (1 + u) * ||q - mu||_2
 *      - Sumando en norma L2 sobre las 64 filas (factor sqrt(64) = 8):
 *          ||q_rot - Q^T(q - mu)||_2 <= (8 * gamma_64 * (1 + u) + 2u) * ||q - mu||_2
 *
 *   5. Resta por particion q_rot_p = fl(q_rot - c_rot_p):
 *      - Introduce error adicional <= u * ||q_rot - c_rot_p||_2
 *        <= 2u * (||q - mu||_2 + ||c_p - mu||_2).
 *      - Por tanto, la perturbacion L2 total delta_vec(q, p) entre el residuo
 *        calculado (q_rot_p - x_rot_p) y el vector exacto Q^T(q - x) en cualquier
 *        subconjunto de dimensiones S subseteq {0..63} cumple:
 *          || (q_rot_p - x_rot_p)_S - (Q^T(q - x))_S ||_2 <= delta_vec(q, p)
 *        donde:
 *          delta_vec(q, p) = (8 * gamma_64 * (1 + u) + 4u) * ||q - mu||_2
 *                          + 4u * ||c_p - mu||_2
 *                          + 2u * r_max(p)
 *
 *   6. Calculo de normas de cola (48D y 32D) y redondeo de la cota:
 *      - Norma almacenada r_tail(x, p) calculada en binary64 y redondeada a float32:
 *        error <= u * r_max(p).
 *      - Norma de consulta r_tail(q, p) calculada con AVX2 FMA + sqrtf:
 *        error <= gamma_64 * (dc_high + delta_vec(q, p)).
 *      - Por tanto, la holgura lineal certificada para la cola dimensional es:
 *          fp_slack_p = (1 + 4u) * (delta_vec(q, p)
 *                                 + gamma_64 * (dc_high + delta_vec(q, p))
 *                                 + u * r_max(p))
 *
 *   7. Acumulacion parcial FMA en 16D / 32D / 64D y paso al cuadrado:
 *      - Sea M_{q,p} = dc_high + r_max(p) + delta_vec(q, p) cota superior de ||q_rot_p - x_rot_p||_2.
 *      - Como ||u - v||_2 <= delta_vec implica ||v||_2^2 >= ||u||_2^2 - 2 * delta_vec * ||u||_2,
 *        y la suma parcial AVX2 D_s^2 tiene error relativo <= gamma_16:
 *          sq_slack_p = (2.0 * delta_vec(q, p) * M_{q,p}) / (1.0 - gamma_16)
 *      - Finalmente, para pasar de ||Q^T(q - x)||_2^2 a fl_orig(||q - x||_2^2) y absorber
 *        las 4 operaciones finales de punto flotante que evaluan la cota:
 *          scale_safe = ((1.0 - gamma_66) * (1.0 - gamma_16) * (1.0 - 6.0 * u)) / (1.0 + delta_ortho)
 * ============================================================================
 */

static constexpr double IEEE754_F32_U = 5.9604644775390625e-8; /* 2^-24 exacto */

constexpr double ieee754_gamma(uint32_t n) {
    return (static_cast<double>(n) * IEEE754_F32_U) / (1.0 - static_cast<double>(n) * IEEE754_F32_U);
}

static constexpr float GAMMA_16_F32 = static_cast<float>(ieee754_gamma(16));
static constexpr float GAMMA_64_F32 = static_cast<float>(ieee754_gamma(64));
static constexpr float GAMMA_66_F32 = static_cast<float>(ieee754_gamma(66));
static constexpr float U_F32        = static_cast<float>(IEEE754_F32_U);

/* Coeficiente exacto de propagacion L2 en rotacion de 64 filas: 8 * gamma_64 * (1 + u) + 4u */
static constexpr float ROT64_L2_COEFF_Q = static_cast<float>(
    8.0 * ieee754_gamma(64) * (1.0 + IEEE754_F32_U) + 4.0 * IEEE754_F32_U
);
static constexpr float ROT64_L2_COEFF_C = static_cast<float>(4.0 * IEEE754_F32_U);
static constexpr float ROT64_L2_COEFF_X = static_cast<float>(2.0 * IEEE754_F32_U);

/* Compuerta conservadora para diferencias entre arboles de suma SIMD en 64D: 2 * gamma_66 */
static constexpr float SIMD_TREE_GATE_REL_EPS = static_cast<float>(2.0 * ieee754_gamma(66));

struct PartitionCertifiedBounds {
    float fp_slack_p;
    float sq_slack_p;
    float scale_safe;
};

inline PartitionCertifiedBounds compute_partition_certified_bounds(float q_mean_norm,
                                                                   float c_mean_norm,
                                                                   float dc_high,
                                                                   float r_max_p,
                                                                   float ortho_defect) {
    /* Paso 5: Perturbacion L2 maxima entre (q_rot_p - x_rot_p) y Q^T(q - x) */
    float delta_vec = ROT64_L2_COEFF_Q * q_mean_norm
                    + ROT64_L2_COEFF_C * c_mean_norm
                    + ROT64_L2_COEFF_X * r_max_p;

    /* Paso 6: Holgura lineal de las normas de cola 48D y 32D */
    float tail_norm_err = GAMMA_64_F32 * (dc_high + delta_vec) + U_F32 * r_max_p;
    float fp_slack = (1.0f + 4.0f * U_F32) * (delta_vec + tail_norm_err);

    /* Paso 7: Holgura cuadratica para la suma parcial de prefijo (16D / 32D / 64D) */
    float max_vec_norm = dc_high + r_max_p + delta_vec;
    float sq_slack = (2.0f * delta_vec * max_vec_norm) / (1.0f - GAMMA_16_F32);

    /* Factor de escala inferior entre cota rotada en float32 y fl_orig(||q - x||_2^2) */
    double num = (1.0 - ieee754_gamma(66)) * (1.0 - ieee754_gamma(16)) * (1.0 - 6.0 * IEEE754_F32_U);
    double den = 1.0 + static_cast<double>(ortho_defect);
    float scale_safe = static_cast<float>(num / den);

    return {fp_slack, sq_slack, scale_safe};
}

} // namespace alg_flat

#endif // ALG_FLAT_IEEE754_BOUNDS_H
