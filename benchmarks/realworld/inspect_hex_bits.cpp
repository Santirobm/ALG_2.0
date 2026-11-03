// inspect_hex_bits.cpp
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>
#include <immintrin.h>

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

    // 3. AVX2 vector accumulation (matching execution_plans.cpp)
    __m256 sum_A = _mm256_setzero_ps();
    __m256 sum_B = _mm256_setzero_ps();
    for (uint32_t d = 0; d < dim; d += 8) {
        __m256 q_vec = _mm256_loadu_ps(q + d);
        __m256 va_vec = _mm256_loadu_ps(v_A + d);
        __m256 vb_vec = _mm256_loadu_ps(v_B + d);
        __m256 diff_a = _mm256_sub_ps(q_vec, va_vec);
        __m256 diff_b = _mm256_sub_ps(q_vec, vb_vec);
        sum_A = _mm256_fmadd_ps(diff_a, diff_a, sum_A);
        sum_B = _mm256_fmadd_ps(diff_b, diff_b, sum_B);
    }
    // Horizontal add
    __m128 lo_A = _mm256_castps256_ps128(sum_A);
    __m128 hi_A = _mm256_extractf128_ps(sum_A, 1);
    __m128 s_A = _mm_add_ps(lo_A, hi_A);
    s_A = _mm_hadd_ps(s_A, s_A);
    s_A = _mm_hadd_ps(s_A, s_A);
    float d_A_avx2 = _mm_cvtss_f32(s_A);

    __m128 lo_B = _mm256_castps256_ps128(sum_B);
    __m128 hi_B = _mm256_extractf128_ps(sum_B, 1);
    __m128 s_B = _mm_add_ps(lo_B, hi_B);
    s_B = _mm_hadd_ps(s_B, s_B);
    s_B = _mm_hadd_ps(s_B, s_B);
    float d_B_avx2 = _mm_cvtss_f32(s_B);

    uint64_t hex_A_dbl, hex_B_dbl;
    uint32_t hex_A_flt_sc, hex_B_flt_sc;
    uint32_t hex_A_avx2, hex_B_avx2;
    memcpy(&hex_A_dbl, &d_A_double, 8);
    memcpy(&hex_B_dbl, &d_B_double, 8);
    memcpy(&hex_A_flt_sc, &d_A_float_scalar, 4);
    memcpy(&hex_B_flt_sc, &d_B_float_scalar, 4);
    memcpy(&hex_A_avx2, &d_A_avx2, 4);
    memcpy(&hex_B_avx2, &d_B_avx2, 4);

    std::cout << "================================================================================" << std::endl;
    std::cout << "HEXADECIMAL BIT ANALYSIS: QUERY 165 (CANDIDATES 50563 vs 80871)" << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << "BINARY64 DOUBLE:" << std::endl;
    std::cout << "  Candidate A (50563): " << std::setprecision(17) << d_A_double 
              << " | Hex: 0x" << std::hex << std::setw(16) << std::setfill('0') << hex_A_dbl << std::dec << std::endl;
    std::cout << "  Candidate B (80871): " << std::setprecision(17) << d_B_double 
              << " | Hex: 0x" << std::hex << std::setw(16) << std::setfill('0') << hex_B_dbl << std::dec << std::endl;
    std::cout << "  Delta double: " << std::abs(d_A_double - d_B_double) << std::endl;

    std::cout << "\nFLOAT32 AVX2 (FMA + HADD):" << std::endl;
    std::cout << "  Candidate A (50563): " << std::setprecision(9) << d_A_avx2 
              << " | Hex: 0x" << std::hex << std::setw(8) << std::setfill('0') << hex_A_avx2 << std::dec << std::endl;
    std::cout << "  Candidate B (80871): " << std::setprecision(9) << d_B_avx2 
              << " | Hex: 0x" << std::hex << std::setw(8) << std::setfill('0') << hex_B_avx2 << std::dec << std::endl;
    std::cout << "  AVX2 Float Hex difference: " << (hex_A_avx2 == hex_B_avx2 ? "EXACT BIT MATCH (0 ULP)" : "DIFFERENT BITS") << std::endl;
    if (hex_A_avx2 != hex_B_avx2) {
        int32_t ulp_diff = static_cast<int32_t>(hex_A_avx2) - static_cast<int32_t>(hex_B_avx2);
        std::cout << "  ULP difference: " << ulp_diff << " ULP" << std::endl;
    }

    std::cout << "\nFLOAT32 SCALAR:" << std::endl;
    std::cout << "  Candidate A (50563): " << std::setprecision(9) << d_A_float_scalar 
              << " | Hex: 0x" << std::hex << std::setw(8) << std::setfill('0') << hex_A_flt_sc << std::dec << std::endl;
    std::cout << "  Candidate B (80871): " << std::setprecision(9) << d_B_float_scalar 
              << " | Hex: 0x" << std::hex << std::setw(8) << std::setfill('0') << hex_B_flt_sc << std::dec << std::endl;
    std::cout << "  Scalar Float Hex difference: " << (hex_A_flt_sc == hex_B_flt_sc ? "EXACT BIT MATCH (0 ULP)" : "DIFFERENT BITS") << std::endl;
    if (hex_A_flt_sc != hex_B_flt_sc) {
        int32_t ulp_diff = static_cast<int32_t>(hex_A_flt_sc) - static_cast<int32_t>(hex_B_flt_sc);
        std::cout << "  ULP difference: " << ulp_diff << " ULP" << std::endl;
    }

    return 0;
}
