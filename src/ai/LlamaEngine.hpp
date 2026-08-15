#pragma once

#include "IAIEngine.hpp"
#include <memory>

struct llama_model;
struct llama_context;

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
        auto embed(const std::string& data) -> std::vector<float> override;
        auto infer(const std::string& prompt) -> std::string override;
        auto is_ready() const noexcept -> bool override;

    private:
        unique_model_ptr m_model;
        unique_context_ptr m_context;
        EngineConfig m_config;
    };
} //namespace f1_pulse::ai