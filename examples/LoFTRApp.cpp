#include "ort-loftr/LoFTR.hpp"
#include <utility>

int main(int argc, char* argv[])
{
    if (argc != 4) {
        std::cerr << "Usage: [apps] [path/to/onnx/loftr] [path/to/image1] [path/to/image2]" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string ONNX_MODEL_PATH = argv[1];
    const std::vector<std::string> IMAGE_PATHS = {argv[2], argv[3]};

    std::vector<cv::Mat> images;
    std::vector<cv::Mat> grays;
    std::transform(IMAGE_PATHS.begin(), IMAGE_PATHS.end(), std::back_inserter(images),
                   [](const auto& imagePath) { return cv::imread(imagePath); });
    for (int i = 0; i < 2; ++i) {
        if (images[i].empty()) {
            throw std::runtime_error("failed to open " + IMAGE_PATHS[i]);
        }
    }

    std::transform(IMAGE_PATHS.begin(), IMAGE_PATHS.end(), std::back_inserter(grays),
                   [](const auto& imagePath) { return cv::imread(imagePath, 0); });

    Ort::LoFTR osh(ONNX_MODEL_PATH, 0);
    auto matchedKpts = osh.inference(osh, grays[0], grays[1]);
    auto matches = osh.getMatches(matchedKpts.first, matchedKpts.second);

    
    cv::Mat matchesImage;
    cv::drawMatches(images[0], matchedKpts.first, images[1], matchedKpts.second, matches, matchesImage, cv::Scalar::all(-1),
                    cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imwrite("loftr.jpg", matchesImage);
    cv::imshow("loftr", matchesImage);
    cv::waitKey();

    return EXIT_SUCCESS;
}