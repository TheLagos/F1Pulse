#pragma once

#include "IAIEngine.hpp"
#include <memory>

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

    class LlamaEngine : public IAIEngine {
    public:
        LlamaEngine() = default;
        ~LlamaEngine() override = default;

        LlamaEngine(const LlamaEngine&) = delete;
        LlamaEngine& operator=(const LlamaEngine&) = delete;

        LlamaEngine(LlamaEngine&&) noexcept = default;
        LlamaEngine& operator=(LlamaEngine&&) noexcept = default;

        auto init(const EngineConfig& config) -> bool override;
        auto embed(std::string_view data) -> std::vector<float> override;
        auto infer(std::string_view prompt, const SamplingParams& params = {}) -> std::string override;
        auto is_ready() const noexcept -> bool override;

    private:
        auto tokenize(std::string_view text, bool add_special = true) const -> std::vector<llama_token>;

        unique_model_ptr m_model;
        unique_context_ptr m_context;

        EngineConfig m_config{};
        SamplingParams m_params{};
    };
} //namespace f1_pulse::ai