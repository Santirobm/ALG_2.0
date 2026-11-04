/*
 * Copyright (c) 2026 Santiago Robledo.
 * SPDX-License-Identifier: Apache-2.0
 */
// test_pure_cpp_faiss.cpp
#include <iostream>
#include <cstdint>
#include <vector>

#define FAISS_API __declspec(dllimport)

namespace faiss {
    using idx_t = int64_t;

    struct IDSelector {
        virtual bool is_member(idx_t id) const = 0;
        virtual ~IDSelector() {}
    };

    struct FAISS_API IDSelectorBitmap : IDSelector {
        size_t n;
        const uint8_t* bitmap;
        IDSelectorBitmap(size_t n, const uint8_t* bitmap);
        virtual ~IDSelectorBitmap();
        virtual bool is_member(idx_t id) const override;
    };

    struct SearchParameters {
        IDSelector* sel = nullptr;
        virtual ~SearchParameters() {}
    };

    enum MetricType {
        METRIC_INNER_PRODUCT = 0,
        METRIC_L2 = 1,
    };

    struct FAISS_API Index {
        int d;
        idx_t ntotal;
        bool verbose;
        bool is_trained;
        MetricType metric_type;
        float metric_arg;

        virtual ~Index();
        virtual void train(idx_t n, const float* x);
        virtual void add(idx_t n, const float* x) = 0;
        virtual void add_with_ids(idx_t n, const float* x, const idx_t* xids);
        virtual void search(idx_t n, const float* x, idx_t k, float* distances, idx_t* labels, const SearchParameters* params = nullptr) const = 0;
    };

    struct FAISS_API IndexFlat : Index {
        std::vector<uint8_t> codes;
        size_t code_size;

        virtual ~IndexFlat();
        virtual void add(idx_t n, const float* x) override;
        virtual void reset();
        virtual void search(idx_t n, const float* x, idx_t k, float* distances, idx_t* labels, const SearchParameters* params = nullptr) const override;
    };

    struct FAISS_API IndexFlatL2 : IndexFlat {
        IndexFlatL2(idx_t d);
        virtual ~IndexFlatL2();
    };
}

int main() {
    std::cout << "Testing pure C++ FAISS linking..." << std::endl;
    int d = 4;
    faiss::IndexFlatL2 index(d);
    std::cout << "Created IndexFlatL2, d=" << index.d << std::endl;

    float data[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    index.add(2, data);
    std::cout << "Added 2 vectors, ntotal=" << index.ntotal << std::endl;

    uint8_t bitmap[1] = {0x01}; // only id 0 is member
    faiss::IDSelectorBitmap sel(2, bitmap);
    std::cout << "Created IDSelectorBitmap" << std::endl;

    faiss::SearchParameters params;
    params.sel = &sel;

    float query[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float dist[2] = {-1.0f, -1.0f};
    faiss::idx_t labels[2] = {-1, -1};

    index.search(1, query, 2, dist, labels, &params);
    std::cout << "Search done: label[0]=" << labels[0] << " dist[0]=" << dist[0]
              << " label[1]=" << labels[1] << " dist[1]=" << dist[1] << std::endl;

    return 0;
}
