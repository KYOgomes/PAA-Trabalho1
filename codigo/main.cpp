#include "imagedescriptor.hpp"
#include "list.hpp"
#include "quadtreeQr.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;

// --- Função auxiliar: garante carregamento limpo da imagem ---
cv::Mat loadImage(const std::string& path) {
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Erro: não foi possível abrir " << path << std::endl;
        return {};
    }
    if (img.channels() == 4)
        cv::cvtColor(img, img, cv::COLOR_BGRA2BGR);
    cv::resize(img, img, cv::Size(256, 256));
    return img;
}

// --- Busca com Lista ---
void runList(const std::string& queryPath, int K) {
    ListIndex index;
    std::string imagesDir = "images";

    for (const auto& entry : fs::directory_iterator(imagesDir)) {
        if (entry.is_regular_file()) {
            std::string path = entry.path().string();
            if (path == queryPath) continue;

            cv::Mat img = loadImage(path);
            if (img.empty()) continue;

            Descriptor d = computeDescriptor(img);
            Record r{0, path, d.hist, d.muH, d.muS};
            index.add(r);
        }
    }

    cv::Mat qimg = loadImage(queryPath);
    if (qimg.empty()) return;

    Descriptor dq = computeDescriptor(qimg);
    Record q{0, queryPath, dq.hist, dq.muH, dq.muS};

    auto results = index.query(q, K);

    std::cout << "Resultados (Lista Sequencial):\n";
    for (auto& r : results) {
        std::cout << r.filepath << "\n";
    }
}

// --- Busca com Quadtree ---
void runQuadtree(const std::string& queryPath, int K) {
    Rect boundary{0.5f, 0.5f, 0.5f, 0.5f};
    Quadtree qt(boundary, 4);

    for (const auto& entry : fs::directory_iterator("images")) {
        if (entry.is_regular_file()) {
            cv::Mat img = loadImage(entry.path().string());
            if (img.empty()) continue;

            Descriptor d = computeDescriptor(img);
            Record r{0, entry.path().string(), d.hist, d.muH, d.muS};
            Point p{d.muH, d.muS, r};
            qt.insert(p);
        }
    }

    cv::Mat qimg = loadImage(queryPath);
    if (qimg.empty()) return;

    Descriptor dq = computeDescriptor(qimg);
    Record rq{0, queryPath, dq.hist, dq.muH, dq.muS};
    Point qpoint{dq.muH, dq.muS, rq};

    float radius = 0.2f; // aumentar raio para pegar mais candidatos
    Rect range{qpoint.x, qpoint.y, radius, radius};
    std::vector<Point> candidates;
    qt.queryRange(range, candidates);

    std::vector<std::pair<float, Record>> dists;
    for (auto& p : candidates) {
        float d = chi2_distance(rq.hist, p.record.hist);
        dists.push_back({d, p.record});
    }
    std::sort(dists.begin(), dists.end(),
              [](auto& a, auto& b) { return a.first < b.first; });

    std::cout << "Resultados (Quadtree):\n";
    for (int i = 0; i < K && i < (int)dists.size(); i++) {
        std::cout << dists[i].second.filepath << "\n";
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Uso: " << argv[0] << " <imagem_query>" << std::endl;
        return -1;
    }

    std::string queryPath = argv[1];
    int K = 5;

    std::cout << "Escolha o método:\n";
    std::cout << "1 - Lista Sequencial\n";
    std::cout << "2 - Quadtree\n";
    int choice;
    std::cin >> choice;

    if (choice == 1) {
        runList(queryPath, K);
    } else if (choice == 2) {
        runQuadtree(queryPath, K);
    } else {
        std::cout << "Opção inválida.\n";
    }

    return 0;
}
