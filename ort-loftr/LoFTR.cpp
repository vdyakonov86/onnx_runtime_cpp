#include "ort-loftr/LoFTR.hpp"

namespace Ort
{
void LoFTR::preprocess(float* dst, const unsigned char* src, const int64_t targetImgWidth,
                       const int64_t targetImgHeight, const int numChannels) const
{
    for (int i = 0; i < targetImgHeight; ++i) {
        for (int j = 0; j < targetImgWidth; ++j) {
            for (int c = 0; c < numChannels; ++c) {
                dst[c * targetImgHeight * targetImgWidth + i * targetImgWidth + j] =
                    (src[i * targetImgWidth * numChannels + j * numChannels + c] / 255.0);
            }
        }
    }
}

std::vector<std::vector<int64_t>> LoFTR::getInputShapes() {
    return std::vector<std::vector<int64_t>>{
        {1, Ort::LoFTR::IMG_CHANNEL, Ort::LoFTR::IMG_H, Ort::LoFTR::IMG_W},
        {1, Ort::LoFTR::IMG_CHANNEL, Ort::LoFTR::IMG_H, Ort::LoFTR::IMG_W}};
}

std::pair<std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>> 
LoFTR::inference(LoFTR& loftr, const cv::Mat& queryImg, const cv::Mat& refImg, float confidenceThresh) {
    auto inputShapes = loftr.getInputShapes();
    loftr.updateInputShapes(inputShapes);

    std::vector<float> queryData(Ort::LoFTR::IMG_CHANNEL * Ort::LoFTR::IMG_H * Ort::LoFTR::IMG_W);
    std::vector<float> refData(Ort::LoFTR::IMG_CHANNEL * Ort::LoFTR::IMG_H * Ort::LoFTR::IMG_W);

    int origQueryW = queryImg.cols, origQueryH = queryImg.rows;
    int origRefW = refImg.cols, origRefH = refImg.rows;

    cv::Mat scaledQueryImg, scaledRefImg;
    cv::resize(queryImg, scaledQueryImg, cv::Size(Ort::LoFTR::IMG_W, Ort::LoFTR::IMG_H), 0, 0, cv::INTER_CUBIC);
    cv::resize(refImg, scaledRefImg, cv::Size(Ort::LoFTR::IMG_W, Ort::LoFTR::IMG_H), 0, 0, cv::INTER_CUBIC);

    loftr.preprocess(queryData.data(), scaledQueryImg.data, Ort::LoFTR::IMG_W, Ort::LoFTR::IMG_H, Ort::LoFTR::IMG_CHANNEL);
    loftr.preprocess(refData.data(), scaledRefImg.data, Ort::LoFTR::IMG_W, Ort::LoFTR::IMG_H, Ort::LoFTR::IMG_CHANNEL);
    auto inferenceOutput = loftr({queryData.data(), refData.data()});

    // inferenceOutput[0].second: keypoints0 of shape [num kpt x 2]
    // inferenceOutput[1].second: keypoints1 of shape [num kpt x 2]
    // inferenceOutput[2].second: confidences of shape [num kpt]

    int numKeyPoints = inferenceOutput[2].second[0];
    std::vector<cv::KeyPoint> queryKpts, refKpts;
    queryKpts.reserve(numKeyPoints);
    refKpts.reserve(numKeyPoints);

    for (int i = 0; i < numKeyPoints; ++i) {
        float confidence = inferenceOutput[2].first[i];
        if (confidence < confidenceThresh) {
            continue;
        }
        float queryX = inferenceOutput[0].first[i * 2 + 0];
        float queryY = inferenceOutput[0].first[i * 2 + 1];
        float refX = inferenceOutput[1].first[i * 2 + 0];
        float refY = inferenceOutput[1].first[i * 2 + 1];
        cv::KeyPoint queryKpt, refKpt;
        queryKpt.pt.x = queryX * origQueryW / Ort::LoFTR::IMG_W;
        queryKpt.pt.y = queryY * origQueryH / Ort::LoFTR::IMG_H;

        refKpt.pt.x = refX * origRefW / Ort::LoFTR::IMG_W;
        refKpt.pt.y = refY * origRefH / Ort::LoFTR::IMG_H;

        queryKpts.emplace_back(std::move(queryKpt));
        refKpts.emplace_back(std::move(refKpt));
    }

    return std::make_pair(queryKpts, refKpts);
}

std::vector<cv::DMatch> LoFTR::getMatches(const std::vector<cv::KeyPoint>& queryKpts, const std::vector<cv::KeyPoint>& refKpts) {
    std::vector<cv::DMatch> matches;
    for (int i = 0; i < queryKpts.size(); ++i) {
        cv::DMatch match;
        match.imgIdx = 0;
        match.queryIdx = i;
        match.trainIdx = i;
        matches.emplace_back(std::move(match));
    }
    return matches;
}
}  // namespace Ort
