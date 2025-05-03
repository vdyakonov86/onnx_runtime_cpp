#pragma once

#include <opencv2/opencv.hpp>
#include <ort_utility/ort_utility.hpp>

namespace Ort
{
class LoFTR : public OrtSessionHandler
{
 public:
    static constexpr int64_t IMG_H = 480;
    static constexpr int64_t IMG_W = 640;
    static constexpr int64_t IMG_CHANNEL = 1;
    static constexpr float CONFIDENCE_THRESHOLD = 0.1;

    using OrtSessionHandler::OrtSessionHandler;

    void preprocess(float* dst,                     //
                    const unsigned char* src,       //
                    const int64_t targetImgWidth,   //
                    const int64_t targetImgHeight,  //
                    const int numChannels) const;
   
   std::vector<std::vector<int64_t>> getInputShapes();

   std::pair<std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>>
   inference(LoFTR& loftr, const cv::Mat& queryImg, const cv::Mat& refImg, float confidenceThresh = CONFIDENCE_THRESHOLD);
   
   std::vector<cv::DMatch> getMatches(const std::vector<cv::KeyPoint>& queryKpts, const std::vector<cv::KeyPoint>& refKpts);
};

}  // namespace Ort
