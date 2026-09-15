#pragma once

#include <string>
#include <string_view>
#include <cstdint>
#include <vector>
#include <filesystem>

namespace f1_pulse::ai {
    /// @brief Configuration used to load and set up a model in IAIEngine::init().
    struct EngineConfig {
        std::filesystem::path model_path;  ///< Path to a local GGUF model file.
        uint32_t context_size{4096};       ///< Context window size in tokens; 0 = use the model's trained context size.
        int32_t gpu_layers{99};            ///< Number of model layers to offload to the GPU; 0 = CPU-only.
        uint32_t threads{4};               ///< CPU threads used for decoding/generation.
        bool enable_embeddings{false};     ///< Enable embeddings output so embed() can be used.
    };

    /// @brief Per-call parameters controlling text generation in IAIEngine::infer().
    struct SamplingParams {
        std::vector<std::string> stop_sequences{}; ///< Generation stops as soon as any of these substrings appears in the output.
        float temperature{0.7f};                   ///< Sampling temperature; higher values increase randomness.
        float top_p{0.9f};                         ///< Nucleus sampling threshold: keep the smallest token set with cumulative probability >= top_p.
        int32_t top_k{40};                         ///< Restrict sampling to the top_k most likely tokens.
        uint32_t max_tokens{1024};                 ///< Maximum number of tokens to generate before stopping.
    };

    /// @brief Abstraction over a local LLM backend (e.g. llama.cpp), providing
    /// model loading, text embedding, and text generation.
    class IAIEngine {
    public:
        /// @brief Default destructor; derived engines are responsible for releasing
        /// any native model/context resources they own.
        virtual ~IAIEngine() = default;

        /// @brief Loads a model and prepares the engine for use according to `config`.
        /// @return true on success; false if the model could not be loaded or the
        /// engine was already initialized.
        virtual auto init(const EngineConfig& config) -> bool = 0;

        /// @brief Computes an embedding vector for `data`.
        /// @return The embedding, or an empty vector on failure (e.g. engine not
        /// ready, empty input, or embeddings not enabled in EngineConfig).
        virtual auto embed(std::string_view data) -> std::vector<float> = 0;

        /// @brief Generates a text completion for `prompt` using `params`.
        /// @return The generated text, or an empty string on failure (e.g. engine
        /// not ready or empty prompt).
        virtual auto infer(std::string_view prompt, const SamplingParams& params = {}) -> std::string = 0;

        /// @brief Reports whether the engine has been successfully initialized and
        /// is ready to serve embed()/infer() calls.
        virtual auto is_ready() const noexcept -> bool = 0;
    };
} //namespace f1_pulse::ai