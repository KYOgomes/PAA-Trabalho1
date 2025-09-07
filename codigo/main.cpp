#include "descritor/imagedescriptor.hpp"
#include "estruturas/list.hpp"
#include "estruturas/quadtree.hpp"
#include "estruturas/hash.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <ctime>     
#include <iomanip>    
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
            if (path == queryPath) continue; // já estava correto

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
        if (r.filepath == queryPath) continue; // redundância extra para segurança
        std::cout << r.filepath << "\n";
    }
}

// --- Busca com Quadtree ---
void runQuadtree(const std::string& queryPath, int K) {
    Rect boundary{0.5f, 0.5f, 0.5f, 0.5f};
    Quadtree qt(boundary, 4);

    for (const auto& entry : fs::directory_iterator("images")) {
        if (entry.is_regular_file()) {
            std::string path = entry.path().string();
            if (path == queryPath) continue; // evita inserir a query na árvore

            cv::Mat img = loadImage(path);
            if (img.empty()) continue;

            Descriptor d = computeDescriptor(img);
            Record r{0, path, d.hist, d.muH, d.muS};
            Point p{d.muH, d.muS, r};
            qt.insert(p);
        }
    }

    cv::Mat qimg = loadImage(queryPath);
    if (qimg.empty()) return;

    Descriptor dq = computeDescriptor(qimg);
    Record rq{0, queryPath, dq.hist, dq.muH, dq.muS};
    Point qpoint{dq.muH, dq.muS, rq};

    float radius = 0.2f;
    Rect range{qpoint.x, qpoint.y, radius, radius};
    std::vector<Point> candidates;
    qt.queryRange(range, candidates);

    std::vector<std::pair<float, Record>> dists;
    for (auto& p : candidates) {
        if (p.record.filepath == queryPath) continue; // IGNORA a própria imagem
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

// --- Busca com Hash ---
void runHash(const std::string& queryPath, int K) {
    HashIndex hindex;

    for (const auto& entry : fs::directory_iterator("images")) {
        if (entry.is_regular_file()) {
            std::string path = entry.path().string();
            if (path == queryPath) continue; // evita inserir a query no hash

            cv::Mat img = loadImage(path);
            if (img.empty()) continue;

            Descriptor d = computeDescriptor(img);
            Record r{0, path, d.hist, d.muH, d.muS};
            hindex.add(r, d);
        }
    }

    cv::Mat qimg = loadImage(queryPath);
    if (qimg.empty()) return;

    Descriptor dq = computeDescriptor(qimg);
    Record rq{0, queryPath, dq.hist, dq.muH, dq.muS};

    auto results = hindex.query(rq, dq, K);

    std::cout << "Resultados (Hash):\n";
    for (auto& r : results) {
        if (r.filepath == queryPath) continue; // IGNORA a própria imagem
        std::cout << r.filepath << "\n";
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Uso: " << argv[0] << " <imagem_query>" << std::endl;
        return -1;
    }

    std::string queryPath = argv[1];
    int K = 3;

    std::cout << "Escolha o método:\n";
    std::cout << "1 - Lista Sequencial\n";
    std::cout << "2 - Quadtree\n";
    std::cout << "3 - Hash\n";
    int choice;
    std::cin >> choice;

    clock_t start = clock(); // início da contagem

    if (choice == 1) {
        runList(queryPath, K);
    } else if (choice == 2) {
        runQuadtree(queryPath, K);
    } else if (choice == 3) {
        runHash(queryPath, K);
    } else {
        std::cout << "Opção inválida.\n";
    }


     clock_t end = clock(); // fim da contagem
    double elapsed = static_cast<double>(end - start) / CLOCKS_PER_SEC;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Tempo de execução: " << elapsed << " segundos\n";
    return 0;
}
