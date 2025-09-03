#pragma once
#include "../descritor/imagedescriptor.hpp"
#include <unordered_map>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

inline std::string makeHashKey(const Descriptor& d) {
    // Hue dominante (0..7)
    int hBin = 0;
    float maxVal = -1.0f;
    for (int i = 0; i < 8; i++) { // primeiros 8 bins correspondem ao Hue
        float val = 0.0f;
        for (int s = 0; s < 3; s++) {
            for (int v = 0; v < 3; v++) {
                int idx = i * (3 * 3) + s * 3 + v;
                val += d.hist[idx];
            }
        }
        if (val > maxVal) {
            maxVal = val;
            hBin = i;
        }
    }

    // Saturação média → 0=baixa,1=média,2=alta
    int sBin = (d.muS < 0.33) ? 0 : (d.muS < 0.66 ? 1 : 2);

    // Value médio aproximado: vamos pegar bins de V somando ao longo do histograma
    float meanV = 0.0f;
    float total = 0.0f;
    for (int h = 0; h < 8; h++) {
        for (int s = 0; s < 3; s++) {
            for (int v = 0; v < 3; v++) {
                int idx = h * 9 + s * 3 + v;
                meanV += v * d.hist[idx];
                total += d.hist[idx];
            }
        }
    }
    meanV /= (total + 1e-8);
    int vBin = (meanV < 1.0) ? 0 : (meanV < 2.0 ? 1 : 2);

    return std::to_string(hBin) + "-" + std::to_string(sBin) + "-" + std::to_string(vBin);
}

class HashIndex {
private:
    std::unordered_map<std::string, std::vector<Record>> table;

public:
    void add(const Record& rec, const Descriptor& d) {
        std::string key = makeHashKey(d);
        table[key].push_back(rec);
    }

    std::vector<Record> query(const Record& q, const Descriptor& d, int K) {
        std::string key = makeHashKey(d);
        std::vector<Record> candidates;

        // pega o bucket
        if (table.find(key) != table.end()) {
            candidates = table[key];
        }

        // fallback: se poucos candidatos, pega buckets vizinhos
        if ((int)candidates.size() < K) {
            for (auto& [k, bucket] : table) {
                if (k == key) continue;
                candidates.insert(candidates.end(), bucket.begin(), bucket.end());
                if ((int)candidates.size() >= K*2) break; // limita
            }
        }

        // refina pelo histograma completo
        std::vector<std::pair<float, Record>> dists;
        for (auto& r : candidates) {
            float dist = chi2_distance(q.hist, r.hist);
            dists.push_back({dist, r});
        }
        std::sort(dists.begin(), dists.end(),
                  [](auto& a, auto& b) { return a.first < b.first; });

        std::vector<Record> result;
        for (int i = 0; i < K && i < (int)dists.size(); i++) {
            result.push_back(dists[i].second);
        }
        return result;
    }
};
