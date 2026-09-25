#pragma once

#include <string>
#include <string_view>
#include <cstdint>
#include <vector>
#include <filesystem>
#include <expected>

namespace f1_pulse::ai {
    /// @brief Identifies the category of failure returned by IAIEngine::init().
    enum class EngineErrorCode
    {
        AlreadyInitialized,    ///< init() was called on an engine that is already initialized.
        ModelPathFindFailed,   ///< The model file could not be located at the path specified in EngineConfig::model_path.
        ModelLoadFailed,       ///< The model file was found but could not be loaded (corrupt, unsupported format, or insufficient memory).
        ContextCreationFailed  ///< The backend failed to create a context with the requested parameters (e.g. context_size too large for available VRAM).
    };

    /// @brief Carries a machine-readable error category and a human-readable
    /// description returned by IAIEngine::init() on failure.
    struct EngineError{
        EngineErrorCode code;
        std::string message;
    };

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
        /// @param config Settings describing the model file, context size, GPU offload, and thread count.
        /// @return An empty expected on success, or an EngineError describing the
        /// failure (e.g. AlreadyInitialized, ModelPathFindFailed, ModelLoadFailed,
        /// or ContextCreationFailed).
        virtual auto init(const EngineConfig& config) -> std::expected<void, EngineError> = 0;

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