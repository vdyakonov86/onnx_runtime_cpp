/**
 * @file    OrtSessionHandler.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <boost/optional.hpp>

namespace Ort
{
class OrtSessionHandler
{
 public:
    // DataOutputType->(pointer to output data, shape of output data)
    using DataOutputType = std::pair<float*, std::vector<int64_t>>;

    explicit OrtSessionHandler(const std::string& modelPath,  //
                               const boost::optional<size_t>& gpuIdx = boost::none,
                               const boost::optional<std::vector<std::vector<int64_t>>>& inputShapes = boost::none);
    ~OrtSessionHandler();

    // multiple inputs, multiple outputs
    std::vector<DataOutputType> operator()(const std::vector<float*>& inputImgData) const;

    void updateInputShapes(const std::vector<std::vector<int64_t>>& inputShapes);

 private:
    class OrtSessionHandlerIml;
    std::unique_ptr<OrtSessionHandlerIml> m_piml;
};
}  // namespace Ort
