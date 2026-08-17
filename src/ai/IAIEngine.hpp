#pragma once

#include <string>
#include <string_view>
#include <cstdint>
#include <vector>
#include <filesystem>

namespace f1_pulse::ai {
    /// @brief
    struct EngineConfig {
        std::filesystem::path model_path;
        uint32_t context_size{4096};
        int32_t gpu_layers{99};
        uint32_t threads{4};
        bool enable_embeddings{false};
    };

    /// @brief
    struct SamplingParams {
        std::vector<std::string> stop_sequences{};
        float temperature{0.7f};
        float top_p{0.9f};
        int32_t top_k{40};
        uint32_t max_tokens{1024};
    };

    /// @brief
    class IAIEngine {
    public:
        /// @brief
        virtual ~IAIEngine() = default;

        /// @brief
        virtual auto init(const EngineConfig& config) -> bool = 0;

        /// @brief
        virtual auto embed(std::string_view data) -> std::vector<float> = 0;

        /// @brief
        virtual auto infer(std::string_view prompt, const SamplingParams& params = {}) -> std::string = 0;

        /// @brief
        virtual auto is_ready() const noexcept -> bool = 0;
    };
} //namespace f1_pulse::ai