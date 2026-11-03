// inspect_hex_bits.cpp
#include "../../include/simd_kernels.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>
#include <immintrin.h>

using namespace alg_flat;

int main() {
    const std::string corpus_file = "data/hito3_3/D64/corpus.f32";
    const std::string queries_file = "data/hito3_3/D64/queries.f32";
    const uint32_t N = 90000;
    const uint32_t dim = 64;

    std::vector<float> corpus(static_cast<size_t>(N) * dim);
    std::ifstream f_corp(corpus_file, std::ios::binary);
    f_corp.read(reinterpret_cast<char*>(corpus.data()), corpus.size() * sizeof(float));
    f_corp.close();

    std::vector<float> queries(10000 * dim);
    std::ifstream f_q(queries_file, std::ios::binary);
    f_q.read(reinterpret_cast<char*>(queries.data()), queries.size() * sizeof(float));
    f_q.close();

    uint32_t q_idx = 165; // Query #65 in COLD
    const float* q = &queries[q_idx * dim];

    uint32_t id_A = 50563;
    uint32_t id_B = 80871;
    const float* v_A = &corpus[id_A * dim];
    const float* v_B = &corpus[id_B * dim];

    // 1. Double precision accumulation
    double d_A_double = 0.0, d_B_double = 0.0;
    for (uint32_t d = 0; d < dim; ++d) {
        double diff_a = static_cast<double>(q[d]) - static_cast<double>(v_A[d]);
        double diff_b = static_cast<double>(q[d]) - static_cast<double>(v_B[d]);
        d_A_double += diff_a * diff_a;
        d_B_double += diff_b * diff_b;
    }

    // 2. Standard scalar float32 accumulation
    float d_A_float_scalar = 0.0f, d_B_float_scalar = 0.0f;
    for (uint32_t d = 0; d < dim; ++d) {
        float diff_a = q[d] - v_A[d];
        float diff_b = q[d] - v_B[d];
        d_A_float_scalar += diff_a * diff_a;
        d_B_float_scalar += diff_b * diff_b;
    }

    // 3. Exact production kernel: l2_sq_avx2_4x_aos
    // We pass v_A as slot 0 and v_B as slot 1 (and dummy in slots 2,3)
    __m128 d4 = l2_sq_avx2_4x_aos(q, v_A, v_B, v_A, v_B, dim);
    alignas(16) float res4[4];
    _mm_store_ps(res4, d4);
    float d_A_prod_kernel = res4[0];
    float d_B_prod_kernel = res4[1];

    // 4. Exact 1x AoS production kernel: l2_sq_avx2_1x_aos
    float d_A_1x_kernel = l2_sq_avx2_1x_aos(q, v_A, dim);
    float d_B_1x_kernel = l2_sq_avx2_1x_aos(q, v_B, dim);

    uint64_t hex_A_dbl, hex_B_dbl;
    uint32_t hex_A_flt_sc, hex_B_flt_sc;
    uint32_t hex_A_prod, hex_B_prod;
    uint32_t hex_A_1x, hex_B_1x;

    memcpy(&hex_A_dbl, &d_A_double, 8);
    memcpy(&hex_B_dbl, &d_B_double, 8);
    memcpy(&hex_A_flt_sc, &d_A_float_scalar, 4);
    memcpy(&hex_B_flt_sc, &d_B_float_scalar, 4);
    memcpy(&hex_A_prod, &d_A_prod_kernel, 4);
    memcpy(&hex_B_prod, &d_B_prod_kernel, 4);
    memcpy(&hex_A_1x, &d_A_1x_kernel, 4);
    memcpy(&hex_B_1x, &d_B_1x_kernel, 4);

    std::cout << "================================================================================" << std::endl;
    std::cout << "HEXADECIMAL BIT ANALYSIS: QUERY 165 (CANDIDATES 50563 vs 80871)" << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << "BINARY64 DOUBLE:" << std::endl;
    std::cout << "  Candidate A (50563): " << std::setprecision(17) << d_A_double 
              << " | Hex: 0x" << std::hex << std::setw(16) << std::setfill('0') << hex_A_dbl << std::dec << std::endl;
    std::cout << "  Candidate B (80871): " << std::setprecision(17) << d_B_double 
              << " | Hex: 0x" << std::hex << std::setw(16) << std::setfill('0') << hex_B_dbl << std::dec << std::endl;
    std::cout << "  Delta double: " << std::abs(d_A_double - d_B_double) << std::endl;

    std::cout << "\nEXACT PRODUCTION KERNEL (l2_sq_avx2_4x_aos):" << std::endl;
    std::cout << "  Candidate A (50563): " << std::setprecision(9) << d_A_prod_kernel 
              << " | Hex: 0x" << std::hex << std::setw(8) << std::setfill('0') << hex_A_prod << std::dec << std::endl;
    std::cout << "  Candidate B (80871): " << std::setprecision(9) << d_B_prod_kernel 
              << " | Hex: 0x" << std::hex << std::setw(8) << std::setfill('0') << hex_B_prod << std::dec << std::endl;
    std::cout << "  AVX2 4x Hex difference: " << (hex_A_prod == hex_B_prod ? "EXACT BIT MATCH (0 ULP)" : "DIFFERENT BITS") << std::endl;
    if (hex_A_prod != hex_B_prod) {
        int32_t ulp_diff = static_cast<int32_t>(hex_A_prod) - static_cast<int32_t>(hex_B_prod);
        std::cout << "  ULP difference: " << ulp_diff << " ULP" << std::endl;
    }

    std::cout << "\nEXACT 1X KERNEL (l2_sq_avx2_1x_aos):" << std::endl;
    std::cout << "  Candidate A (50563): " << std::setprecision(9) << d_A_1x_kernel 
              << " | Hex: 0x" << std::hex << std::setw(8) << std::setfill('0') << hex_A_1x << std::dec << std::endl;
    std::cout << "  Candidate B (80871): " << std::setprecision(9) << d_B_1x_kernel 
              << " | Hex: 0x" << std::hex << std::setw(8) << std::setfill('0') << hex_B_1x << std::dec << std::endl;
    std::cout << "  AVX2 1x Hex difference: " << (hex_A_1x == hex_B_1x ? "EXACT BIT MATCH (0 ULP)" : "DIFFERENT BITS") << std::endl;

    std::cout << "\nFLOAT32 SCALAR:" << std::endl;
    std::cout << "  Candidate A (50563): " << std::setprecision(9) << d_A_float_scalar 
              << " | Hex: 0x" << std::hex << std::setw(8) << std::setfill('0') << hex_A_flt_sc << std::dec << std::endl;
    std::cout << "  Candidate B (80871): " << std::setprecision(9) << d_B_float_scalar 
              << " | Hex: 0x" << std::hex << std::setw(8) << std::setfill('0') << hex_B_flt_sc << std::dec << std::endl;
    std::cout << "  Scalar Float Hex difference: " << (hex_A_flt_sc == hex_B_flt_sc ? "EXACT BIT MATCH (0 ULP)" : "DIFFERENT BITS") << std::endl;

    return 0;
}
