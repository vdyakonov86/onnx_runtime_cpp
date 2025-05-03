#include "ort-superpoint/SuperPoint.hpp"
#include "ort-superpoint/Utility.hpp"
#include "ort-superglue/SuperGlue.hpp"
 
 namespace Ort
 {
    std::vector<cv::DMatch> SuperGlue::inference(
        SuperGlue& superGlue,
        std::vector<KeyPointAndDesc>& superPointResults,
        cv::Mat& image)
    {
        int numKeypoints0 = superPointResults[0].first.size();
        int numKeypoints1 = superPointResults[1].first.size();

        std::vector<std::vector<int64_t>> inputShapes = superGlue.getInputShapes(numKeypoints0, numKeypoints1);
        superGlue.updateInputShapes(inputShapes);

        for (auto& curKeyPointAndDesc : superPointResults) {
            superGlue.normalizeDescriptors(&curKeyPointAndDesc.second);
        }

        std::vector<std::vector<float>> imageShapes(2);
        std::vector<std::vector<float>> scores(2);
        std::vector<std::vector<float>> keypoints(2);
        std::vector<std::vector<float>> descriptors(2);
    
        cv::Mat buffer;
        for (int i = 0; i < 2; ++i) {
            imageShapes[i] = {1, 1, static_cast<float>(image.rows), static_cast<float>(image.cols)};
            std::transform(superPointResults[i].first.begin(), superPointResults[i].first.end(),
                           std::back_inserter(scores[i]), [](const cv::KeyPoint& keypoint) { return keypoint.response; });
            for (const auto& k : superPointResults[i].first) {
                keypoints[i].emplace_back(k.pt.y);
                keypoints[i].emplace_back(k.pt.x);
            }
    
            transposeNDWrapper(superPointResults[i].second, {1, 0}, buffer);
            std::copy(buffer.begin<float>(), buffer.end<float>(), std::back_inserter(descriptors[i]));
            buffer.release();
        }

        auto superGlueOrtOutput = superGlue({imageShapes[0].data(), scores[0].data(), keypoints[0].data(), descriptors[0].data(),
                    imageShapes[1].data(), scores[1].data(), keypoints[1].data(), descriptors[1].data()});

        // match keypoints 0 to keypoints 1
        std::vector<int64_t> matchIndices(reinterpret_cast<int64_t*>(superGlueOrtOutput[0].first), reinterpret_cast<int64_t*>(superGlueOrtOutput[0].first) + numKeypoints0);

        std::vector<cv::DMatch> matches = superGlue.getMatches(matchIndices);
        return matches;
    }

    std::vector<std::vector<int64_t>> SuperGlue::getInputShapes(const int keypts0_num, const int keypts1_num) {
        std::vector<std::vector<int64_t>> shape{
            {4},
            {1, keypts0_num},
            {1, keypts0_num, 2},
            {1, 256, keypts0_num},
            {4},
            {1, keypts1_num},
            {1, keypts1_num, 2},
            {1, 256, keypts1_num}};
        return shape;
    }

    std::vector<cv::DMatch> SuperGlue::getMatches(const std::vector<int64_t>& matchIndices) {
        std::vector<cv::DMatch> goodMatches;

        for (std::size_t i = 0; i < matchIndices.size(); ++i) {
            if (matchIndices[i] < 0) {
                continue;
            }
            cv::DMatch match;
            match.imgIdx = 0;
            match.queryIdx = i;
            match.trainIdx = matchIndices[i];
            goodMatches.emplace_back(match);
        }
        return goodMatches;
    }

    void SuperGlue::normalizeDescriptors(cv::Mat* descriptors) {
        cv::Mat rsquaredSumMat;
        cv::reduce(descriptors->mul(*descriptors), rsquaredSumMat, 1, cv::REDUCE_SUM);
        cv::sqrt(rsquaredSumMat, rsquaredSumMat);
        for (int i = 0; i < descriptors->rows; ++i) {
            float rsquaredSum = std::max<float>(rsquaredSumMat.ptr<float>()[i], 1e-12);
            descriptors->row(i) /= rsquaredSum;
        }
    }
 }  // namespace Ort
 