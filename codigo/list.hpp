#pragma once
#include "descritor.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

// Distância χ² entre dois histogramas
inline float chi2_distance(const std::vector<float>& h1, const std::vector<float>& h2) {
    float dist = 0.0f;
    for (size_t i = 0; i < h1.size(); i++) {
        float num = (h1[i] - h2[i]) * (h1[i] - h2[i]);
        float den = h1[i] + h2[i] + 1e-8;
        dist += num / den;
    }
    return 0.5f * dist;
}

class ListIndex {
private:
    std::vector<Record> records;

public:
    void add(const Record& rec) {
        records.push_back(rec);
    }

    // Busca top-K mais similares
    std::vector<Record> query(const Record& q, int K) {
        std::vector<std::pair<float, Record>> dists;
        for (auto& r : records) {
            float d = chi2_distance(q.hist, r.hist);
            dists.push_back({d, r});
        }
        std::sort(dists.begin(), dists.end(),
                  [](auto& a, auto& b){ return a.first < b.first; });

        std::vector<Record> result;
        for (int i = 0; i < K && i < (int)dists.size(); i++) {
            result.push_back(dists[i].second);
        }
        return result;
    }
};
