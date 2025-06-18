#include "ort-superpoint/SuperPoint.hpp"
#include "ort-superpoint/Utility.hpp"
#include "ort-lightglue/LightGlue.hpp"
 
 namespace Ort
 {
    std::vector<cv::DMatch> LightGlue::inference(
        LightGlue& lightGlue,
        std::pair<std::vector<cv::KeyPoint>, cv::Mat>& firstObservation,
        std::pair<std::vector<cv::KeyPoint>, cv::Mat>& secondObservation,
        cv::Size imageSize)
    {
        std::vector<std::pair<std::vector<cv::KeyPoint>, cv::Mat>> observations {firstObservation, secondObservation};
        int numKeypoints0 = observations[0].first.size();
        int numKeypoints1 = observations[1].first.size();

        std::vector<std::vector<int64_t>> inputShapes = lightGlue.getInputShapes(numKeypoints0, numKeypoints1);
        lightGlue.updateInputShapes(inputShapes);

        for (auto& curKeyPointAndDesc : observations) {
            lightGlue.normalizeDescriptors(&curKeyPointAndDesc.second);
        }

        std::vector<std::vector<float>> keypoints(2);
        std::vector<std::vector<float>> descriptors(2);
    
        cv::Mat buffer;
        for (int i = 0; i < 2; ++i) {
            for (const auto& k : observations[i].first) {
                keypoints[i].emplace_back(k.pt.y);
                keypoints[i].emplace_back(k.pt.x);
            }
    
            transposeNDWrapper(observations[i].second, {1, 0}, buffer);
            std::copy(buffer.begin<float>(), buffer.end<float>(), std::back_inserter(descriptors[i]));
            buffer.release();
        }

        auto lightGlueOrtOutput = lightGlue({keypoints[0].data(), keypoints[1].data(), descriptors[0].data(), descriptors[1].data()});

        auto shape = lightGlueOrtOutput[0].second;
        // match keypoints 0 to keypoints 1
        std::cout << "shape: " << shape[0] << " " << shape[1] << std::endl;
        std::cout << "numKeypoints0: " << numKeypoints0 << std::endl;

        // // match keypoints 0 to keypoints 1
        // std::vector<int64_t> matchIndices(reinterpret_cast<int64_t*>(lightGlueOrtOutput[0].first), reinterpret_cast<int64_t*>(lightGlueOrtOutput[0].first) + numKeypoints0);

        // int64_t* matches_ptr = reinterpret_cast<int64_t*>(lightGlueOrtOutput[0].first);
        // float* scores_ptr = reinterpret_cast<float*>(lightGlueOrtOutput[1].first);

        // Узнаем количество совпадений (размерность matches0[0])
        // size_t num_matches = lightGlueOrtOutput[0].second / (2 * sizeof(int64_t));  // matches0 имеет форму [num_matches, 2]
        // std::vector<cv::DMatch> matches = lightGlue.getMatches(matchIndices);
        std::vector<cv::DMatch> matches;
        return matches;
    }

    std::vector<std::vector<int64_t>> LightGlue::getInputShapes(const int keypts0_num, const int keypts1_num) {
        std::vector<std::vector<int64_t>> shape{
            {1, keypts0_num, 2},
            {1, keypts1_num, 2},
            {1, keypts0_num, 256},
            {1, keypts1_num, 256}};
        return shape;
    }

    std::vector<cv::DMatch> LightGlue::getMatches(const std::vector<int64_t>& matchIndices) {
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

    void LightGlue::normalizeDescriptors(cv::Mat* descriptors) {
        cv::Mat rsquaredSumMat;
        cv::reduce(descriptors->mul(*descriptors), rsquaredSumMat, 1, cv::REDUCE_SUM);
        cv::sqrt(rsquaredSumMat, rsquaredSumMat);
        for (int i = 0; i < descriptors->rows; ++i) {
            float rsquaredSum = std::max<float>(rsquaredSumMat.ptr<float>()[i], 1e-12);
            descriptors->row(i) /= rsquaredSum;
        }
    }
 }  // namespace Ort
 