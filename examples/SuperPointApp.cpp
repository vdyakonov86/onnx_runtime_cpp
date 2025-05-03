/**
 * @file    SuperPointApp.cpp
 *
 * @author  btran
 *
 */

#include "ort-superpoint/SuperPoint.hpp"
#include "ort-superpoint/Utility.hpp"
#include <opencv2/features2d.hpp>

namespace
{
using KeyPointAndDesc = std::pair<std::vector<cv::KeyPoint>, cv::Mat>;

}  // namespace

int main(int argc, char* argv[])
{
    if (argc != 4) {
        std::cerr << "Usage: [apps] [path/to/onnx/super/point] [path/to/image1] [path/to/image2]" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string ONNX_MODEL_PATH = argv[1];
    const std::vector<std::string> IMAGE_PATHS = {argv[2], argv[3]};

    Ort::SuperPoint osh(ONNX_MODEL_PATH, 0);

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


    std::vector<KeyPointAndDesc> results;
    std::transform(grays.begin(), grays.end(), std::back_inserter(results),
                   [&osh](const auto& gray) { return osh.inference(osh, gray); });

    cv::BFMatcher matcher(cv::NORM_L2, true /* crossCheck */);
    std::vector<cv::DMatch> knnMatches;
    matcher.match(results[0].second, results[1].second, knnMatches);

    cv::Mat matchesImage;
    cv::drawMatches(images[0], results[0].first, images[1], results[1].first, knnMatches, matchesImage,
                    cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imwrite("super_point_good_matches.jpg", matchesImage);
    cv::imshow("super_point_good_matches", matchesImage);
    cv::waitKey();

    return EXIT_SUCCESS;
}