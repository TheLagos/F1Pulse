#pragma once

#include <string>
#include <string_view>
#include <cstdint>
#include <vector>
#include <filesystem>
#include <expected>
#include <variant>
#include <functional>
#include <stop_token>

namespace f1_pulse::ai {
    /// @brief Identifies the category of failure for any engine operation.
    enum class EngineErrorCode {
        AlreadyInitialized,    ///< init() was called on an engine that is already initialized.
        ModelFileNotFound,     ///< The model file could not be located at the path specified in EngineConfig::model_path.
        ModelLoadFailed,       ///< The model file was found but could not be loaded (corrupt, unsupported format, or insufficient memory).
        ContextCreationFailed, ///< The backend failed to create a context with the requested parameters (e.g. context_size too large for available VRAM).
        EmptyInput,            ///< The prompt or input data passed to embed() or infer() was empty.
        NotInitialized,        ///< embed() or infer() was called before a successful init().
        TokenizationFailed,    ///< The backend failed to tokenize the input (e.g. vocabulary not loaded or input too long).
        DecodeFailed,          ///< llama_decode() returned a non-zero status during prefill or generation.
        ContextOverflow,       ///< The number of tokens in the request exceeds the context size allocated at init().
        EmbeddingsDisabled     ///< embed() was called but EngineConfig::enable_embeddings was false at init().
    };

    /// @brief Carries a machine-readable error category and a human-readable
    /// description for any failed engine operation.
    struct EngineError {
        EngineErrorCode code;
        std::string message;
    };

    /// @brief A fragment of generated text; contains valid UTF-8 but may span less than one token boundary.
    struct TextChunk {
        std::string text;
    };

    /// @brief Terminal event delivered when generation completes normally, including when max_tokens is reached.
    struct Done {};

    /// @brief Terminal event delivered when generation was stopped early via std::stop_token.
    struct Cancelled {};

    /// @brief Terminal event delivered when generation fails; carries the error detail.
    struct Failed {
        EngineError error;
    };

    /// @brief Variant passed to the infer() callback: zero or more TextChunk events followed by exactly one terminal event (Done, Cancelled, or Failed).
    using GenerationEvent = std::variant<TextChunk, Done, Cancelled, Failed>;

    /// @brief Callback type invoked by infer() to deliver GenerationEvent values.
    using CallbackFunction = std::function<void(const GenerationEvent&)>;

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
        std::vector<std::string> stop_sequences{}; ///< Generation stops as soon as any of these substrings appears in the output; the matched sequence is not delivered to the callback.
        float temperature{0.7f};                   ///< Sampling temperature; higher values increase randomness, 0 = greedy decoding.
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
        /// @return An empty expected on success, or an EngineError with AlreadyInitialized,
        /// ModelFileNotFound, ModelLoadFailed, or ContextCreationFailed on failure.
        virtual auto init(const EngineConfig& config) -> std::expected<void, EngineError> = 0;

        /// @brief Computes an embedding vector for `data`.
        /// @return The embedding vector on success, or an EngineError if the engine
        /// is not ready, the input is empty, or embeddings are not enabled in EngineConfig.
        virtual auto embed(std::string_view data) -> std::expected<std::vector<float>, EngineError> = 0;

        /// @brief Streams a text completion for `prompt` to `callback`, then returns.
        /// The callback runs synchronously on the calling thread; the internal mutex is
        /// held for the entire generation so concurrent infer()/embed() calls block.
        /// Exactly one terminal event (Done, Cancelled, or Failed) is always delivered
        /// last, even when the engine is not ready or the prompt is empty.
        /// Cancellation is checked between tokens and yields Cancelled.
        /// The callback must not throw or call any method on the same engine.
        /// @param prompt    Text prompt to complete.
        /// @param callback  Receives TextChunk events during generation, then exactly one terminal event.
        /// @param stoken    Cancellation token; checked between tokens.
        /// @param params    Sampling parameters for this call.
        virtual auto infer(std::string_view prompt, const CallbackFunction& callback, std::stop_token stoken, const SamplingParams& params) -> void = 0;

        /// @brief Reports whether the engine has been successfully initialized and
        /// is ready to serve embed()/infer() calls.
        virtual auto is_ready() const noexcept -> bool = 0;
    };
} //namespace f1_pulse::ai