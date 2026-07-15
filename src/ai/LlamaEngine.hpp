#pragma once

#include "IAIEngine.hpp"

struct llama_model;
struct llama_context;

namespace f1_pulse::ai {
    class LlamaEngine : public IAIEngine {
    public:
        LlamaEngine() = default;
        ~LlamaEngine() override;

        LlamaEngine(const LlamaEngine&) = delete;
        LlamaEngine& operator=(const LlamaEngine&) = delete;

        LlamaEngine(LlamaEngine&&) noexcept = delete;
        LlamaEngine& operator=(LlamaEngine&&) noexcept = delete;

        auto init(const EngineConfig& config) -> bool override;
        auto embed(const std::string& data) -> std::vector<float> override;
        auto infer(const std::string& prompt) -> std::string override;
        auto is_ready() const noexcept -> bool override;

    private:
        llama_model* m_model{nullptr};
        llama_context* m_context{nullptr};
        EngineConfig m_config;
    };
} //namespace f1_pulse::ai