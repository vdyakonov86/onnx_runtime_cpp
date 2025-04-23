/**
 * @file    ObjectDetectionOrtSessionHandler.cpp
 *
 * @author  btran
 *
 */

#include "ort_utility/ort_utility.hpp"

namespace Ort
{
ObjectDetectionOrtSessionHandler::ObjectDetectionOrtSessionHandler(
    const uint16_t numClasses,            //
    const std::string& modelPath,         //
    const boost::optional<size_t>& gpuIdx,  //
    const boost::optional<std::vector<std::vector<int64_t>>>& inputShapes)
    : ImageRecognitionOrtSessionHandlerBase(numClasses, modelPath, gpuIdx, inputShapes)
{
}

ObjectDetectionOrtSessionHandler::~ObjectDetectionOrtSessionHandler()
{
}

}  // namespace Ort
