/**
 * @file    ObjectDetectionOrtSessionHandler.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <string>
#include <vector>

#include "ImageRecognitionOrtSessionHandlerBase.hpp"

namespace Ort
{
class ObjectDetectionOrtSessionHandler : public ImageRecognitionOrtSessionHandlerBase
{
 public:
    ObjectDetectionOrtSessionHandler(
        const uint16_t numClasses,                           //
        const std::string& modelPath,                        //
        const boost::optional<size_t>& gpuIdx = boost::none,  //
        const boost::optional<std::vector<std::vector<int64_t>>>& inputShapes = boost::none);

    ~ObjectDetectionOrtSessionHandler();
};
}  // namespace Ort
