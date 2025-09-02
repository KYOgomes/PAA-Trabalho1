#include "imagedescriptor.hpp"
#include "list.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem> // C++17
namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Uso: " << argv[0] << " <imagem_query>" << std::endl;
        return -1;
    }

    std::string queryPath = argv[1];
    std::string imagesDir = "images"; // pasta fixa

    // Cria índice baseado em lista
    ListIndex index;

    // Varre todos os arquivos da pasta "images/"
    for (const auto& entry : fs::directory_iterator(imagesDir)) {
        if (entry.is_regular_file()) {
            std::string path = entry.path().string();

            // Pula se for a mesma imagem da query
            if (path == queryPath) continue;

            cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
            if (img.empty()) {
                std::cerr << "Não foi possível abrir " << path << std::endl;
                continue;
            }

            // Garante que não tenha alfa e redimensiona
            if (img.channels() == 4)
                cv::cvtColor(img, img, cv::COLOR_BGRA2BGR);
            cv::resize(img, img, cv::Size(256,256));

            Descriptor d = computeDescriptor(img);
            Record r {0, path, d.hist, d.muH, d.muS};
            index.add(r);
        }
    }

    // Query
    cv::Mat qimg = cv::imread(queryPath, cv::IMREAD_COLOR);
    if (qimg.empty()) {
        std::cerr << "Erro: não foi possível abrir a imagem de query " << queryPath << std::endl;
        return -1;
    }

    if (qimg.channels() == 4)
        cv::cvtColor(qimg, qimg, cv::COLOR_BGRA2BGR);
    cv::resize(qimg, qimg, cv::Size(256,256));

    Descriptor dq = computeDescriptor(qimg);
    Record q {0, queryPath, dq.hist, dq.muH, dq.muS};

    auto results = index.query(q, 5);

    std::cout << "Top-3 imagens similares:\n";
    for (auto& r : results) {
        std::cout << r.filepath << "\n";
    }
}
