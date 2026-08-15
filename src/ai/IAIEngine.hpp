#pragma once

#include <string>
#include <cstdint>
#include <vector>
#include <filesystem>

namespace f1_pulse::ai {
    /// @brief
    struct EngineConfig {
        std::filesystem::path model_path;
        uint32_t context_size;
        int32_t gpu_layers;
    };

    /// @brief
    class IAIEngine {
    public:
        /// @brief
        virtual ~IAIEngine() = default;

        /// @brief
        virtual auto init(const EngineConfig& config) -> bool = 0;

        /// @brief
        virtual auto embed(const std::string& data) -> std::vector<float> = 0;

        /// @brief
        virtual auto infer(const std::string& prompt) -> std::string = 0;

        /// @brief
        virtual auto is_ready() const noexcept -> bool = 0;
    };
} //namespace f1_pulse::ai