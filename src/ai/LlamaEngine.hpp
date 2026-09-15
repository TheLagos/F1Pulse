#pragma once

#include "IAIEngine.hpp"
#include <atomic>
#include <memory>
#include <mutex>

struct llama_model;
struct llama_context;
using llama_token = int32_t;

namespace f1_pulse::ai {
    struct ModelCleaner {
        void operator() (llama_model* model) const noexcept;
    };

    struct ContextCleaner {
        void operator() (llama_context* context) const noexcept;
    };

    using unique_model_ptr = std::unique_ptr<llama_model, ModelCleaner>;
    using unique_context_ptr = std::unique_ptr<llama_context, ContextCleaner>;

    /// @brief Wraps a single llama.cpp model/context pair.
    ///
    /// Thread-safety: init()/embed()/infer() serialize on an internal mutex, so it
    /// is safe to call them from different threads without external locking (e.g.
    /// an infer() running on a worker thread while another thread calls embed()).
    /// However llama_context itself is not reentrant, so concurrent calls are only
    /// serialized, not parallelized - two infer() calls from different threads will
    /// block on each other rather than run in parallel. is_ready() is lock-free and
    /// safe to poll from any thread (e.g. a UI thread) without blocking on an
    /// in-flight infer(). The engine is neither copyable nor movable: it owns a
    /// mutex and native handles that must not be relocated while another thread
    /// might be using them.
    class LlamaEngine : public IAIEngine {
    public:
        LlamaEngine() = default;
        ~LlamaEngine() override = default;

        LlamaEngine(const LlamaEngine&) = delete;
        LlamaEngine& operator=(const LlamaEngine&) = delete;

        LlamaEngine(LlamaEngine&&) = delete;
        LlamaEngine& operator=(LlamaEngine&&) = delete;

        auto init(const EngineConfig& config) -> bool override;
        auto embed(std::string_view data) -> std::vector<float> override;
        auto infer(std::string_view prompt, const SamplingParams& params = {}) -> std::string override;
        auto is_ready() const noexcept -> bool override;

    private:
        auto tokenize(std::string_view text, bool add_special = true) const -> std::vector<llama_token>;

        /// @brief Submits `tokens` to the context for decoding (prefill or a single
        /// generation step), starting at KV-cache position `start_pos`. Only the
        /// logits of the last token in the batch are requested. Owns and frees its
        /// llama_batch internally (RAII), so it is safe to call on any error path.
        auto decode_tokens(const std::vector<llama_token>& tokens, int32_t start_pos) const -> bool;

        mutable std::mutex m_mutex;
        std::atomic<bool> m_ready{false};

        unique_model_ptr m_model;
        unique_context_ptr m_context;

        EngineConfig m_config{};
        SamplingParams m_params{};
    };
} //namespace f1_pulse::ai