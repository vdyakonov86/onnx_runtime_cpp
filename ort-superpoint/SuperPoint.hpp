/**
 * @file    SuperPoint.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <ort_utility/ort_utility.hpp>
#include <vector>

namespace Ort
{
typedef std::pair<std::vector<cv::KeyPoint>, cv::Mat> KeyPointAndDesc;

class SuperPoint : public OrtSessionHandler
{
 public:
    static constexpr int64_t IMG_H = 480;
    static constexpr int64_t IMG_W = 640;
    static constexpr int64_t IMG_CHANNEL = 1;

    using OrtSessionHandler::OrtSessionHandler;

    KeyPointAndDesc inference(SuperPoint& superPoint, const cv::Mat& image, int borderRemove = 4, float confidenceThresh = 0.015, bool alignCorners = true, int distThresh = 2);

    std::vector<std::vector<int64_t>> getInputShapes();

    void preprocess(float* dst,                     //
                    const unsigned char* src,       //
                    const int64_t targetImgWidth,   //
                    const int64_t targetImgHeight,  //
                    const int numChannels) const;

    std::vector<int> nmsFast(const std::vector<cv::KeyPoint>& keyPoints, int height, int width,
                             int distThresh = 2) const;

    /**
     *  @brief detect super point
     *
     *  @return vector of detected key points
     */
    std::vector<cv::KeyPoint> getKeyPoints(const std::vector<Ort::OrtSessionHandler::DataOutputType>& inferenceOutput,
                                           int borderRemove, float confidenceThresh) const;

    /**
     *  @brief estimate super point's keypoint descriptor
     *
     *  @return keypoint Mat of shape [num key point x 256]
     */
    cv::Mat getDescriptors(const cv::Mat& coarseDescriptors, const std::vector<cv::KeyPoint>& keyPoints, int height,
                           int width, bool alignCorners) const;

    std::vector<cv::DMatch> getMatches(const cv::Mat& queryDesc, const cv::Mat& refDesc, std::string matcherType = "flann", const float ratio_thresh = 0.7f);
};
}  // namespace Ort
