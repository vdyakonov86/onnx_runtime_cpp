#include "ort-superpoint/SuperPoint.hpp"
#include "ort-superpoint/Utility.hpp"
#include "ort-superglue/SuperGlue.hpp"

namespace
{
using KeyPointAndDesc = std::pair<std::vector<cv::KeyPoint>, cv::Mat>;
}  // namespace

int main(int argc, char* argv[])
{
    if (argc != 5) {
        std::cerr
            << "Usage: [apps] [path/to/onnx/super/point] [path/to/onnx/super/glue] [path/to/image1] [path/to/image2]"
            << std::endl;
        return EXIT_FAILURE;
    }

    const std::string ONNX_MODEL_PATH = argv[1];
    const std::string SUPERGLUE_ONNX_MODEL_PATH = argv[2];
    const std::vector<std::string> IMAGE_PATHS = {argv[3], argv[4]};

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


    Ort::SuperPoint superPointOsh(ONNX_MODEL_PATH, 0);

    std::vector<KeyPointAndDesc> superPointResults;
    std::transform(grays.begin(), grays.end(), std::back_inserter(superPointResults),
                   [&superPointOsh](const auto &gray) { return superPointOsh.inference(superPointOsh, gray);});
    
    // superglue
    // std::pair<std::vector<cv::KeyPoint>, cv::Mat> obs1, obs2;

    Ort::SuperGlue superGlueOsh(SUPERGLUE_ONNX_MODEL_PATH, 0);
    auto matches = superGlueOsh.inference(superGlueOsh, superPointResults[0], superPointResults[1], images[0].size());

    cv::Mat matchesImage;
    cv::drawMatches(images[0], superPointResults[0].first, images[1], superPointResults[1].first, matches,
                    matchesImage, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imwrite("super_point_super_glue_good_matches.jpg", matchesImage);
    cv::imshow("super_point_super_glue_good_matches", matchesImage);
    cv::waitKey();

    return EXIT_SUCCESS;
}
