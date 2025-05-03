 #pragma once

 #include <opencv2/opencv.hpp>
 #include <ort_utility/ort_utility.hpp>
 #include <vector>
 
 namespace Ort
 {
  typedef std::pair<std::vector<cv::KeyPoint>, cv::Mat> KeyPointAndDesc;
 class SuperGlue : public OrtSessionHandler
 {
   public:
      using OrtSessionHandler::OrtSessionHandler;

      std::vector<cv::DMatch> getMatches(const std::vector<int64_t>& matchIndices);

      std::vector<std::vector<int64_t>> getInputShapes(const int keypts0_num, const int keypts1_num);

      void normalizeDescriptors(cv::Mat* descriptors);

      std::vector<cv::DMatch> inference(SuperGlue& superGlue, std::vector<KeyPointAndDesc>& superPointResults, cv::Mat& image);
 };
 }  // namespace Ort
 