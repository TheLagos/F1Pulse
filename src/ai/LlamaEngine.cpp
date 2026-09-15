#include "LlamaEngine.hpp"
#include "llama.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace 
{
    /// @brief Returns the offset in `text` where the earliest matching stop
    /// sequence begins, or std::nullopt if none of `stop_sequences` occur yet.
    auto find_stop_sequence(std::string_view text, const std::vector<std::string>& stop_sequences) -> std::optional<size_t>
    {
        std::optional<size_t> earliest;

        for (const auto& stop : stop_sequences)
        {
            if (stop.empty())
            {
                continue;
            }

            if (const auto pos = text.find(stop); pos != std::string_view::npos)
            {
                earliest = earliest ? std::min(*earliest, pos) : pos;
            }
        }

        return earliest;
    }
} // namespace

namespace f1_pulse::ai 
{
    void ModelCleaner::operator() (llama_model* model) const noexcept
    {
        if(model)
        {
            llama_model_free(model);
        }
    }

    void ContextCleaner::operator() (llama_context* context) const noexcept
    {
        if(context)
        {
            llama_free(context);
        }
    }

    auto LlamaEngine::tokenize(std::string_view text, bool add_special) const -> std::vector<llama_token>
    {
        if (!is_ready() || text.empty())
        {
            return {};
        }

        const auto* vocab = llama_model_get_vocab(m_model.get());
        const auto text_size = static_cast<int32_t>(text.size());

        int32_t tokens_count = llama_tokenize(vocab, text.data(), text_size, nullptr, 0, add_special, false) * -1;

        if (tokens_count <= 0)
        {
            return {};
        }

        std::vector<llama_token> tokens(tokens_count);
        int32_t result = llama_tokenize(vocab, text.data(), text_size, tokens.data(), tokens_count, add_special, false);

        if (result < 0)
        {
            return {};
        }

        return tokens;
    }

    auto LlamaEngine::decode_tokens(const std::vector<llama_token>& tokens, int32_t start_pos) const -> bool
    {
        if (tokens.empty())
        {
            return false;
        }

        const auto n_tokens = static_cast<int32_t>(tokens.size());
        auto batch = llama_batch_init(n_tokens, 0, 1);

        // RAII guard so llama_batch_free() always runs, even on early return above/below.
        struct BatchGuard
        {
            llama_batch& batch;
            ~BatchGuard() { llama_batch_free(batch); }
        } guard{batch};

        for (int32_t i = 0; i < n_tokens; ++i)
        {
            batch.token[i] = tokens[i];
            batch.pos[i] = start_pos + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (i == n_tokens - 1);
        }
        batch.n_tokens = n_tokens;

        const int32_t decode_status = llama_decode(m_context.get(), batch);
        if (decode_status != 0)
        {
            std::cerr << "Decode error: llama_decode failed, error code - " << decode_status << "!\n";
            return false;
        }

        return true;
    }

    auto LlamaEngine::init(const EngineConfig& config) -> bool {
        std::lock_guard<std::mutex> lock(m_mutex);

        if (is_ready())
        {
            std::cerr << "LlamaEngine is already initialized!\n";
            return false;
        }

        if (config.model_path.empty())
        {
            std::cerr << "Model path is empty!\n";
            return false;
        }

        if (!std::filesystem::is_regular_file(config.model_path))
        {
            std::cerr << "Model file not found or it's not a regular file: " << config.model_path.string() << '\n';
            return false;
        }

        static bool backend_init = []() {
            llama_backend_init();
            return true;
        }();
        (void)backend_init;

        // params uploading

        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = config.gpu_layers;

        unique_model_ptr model(llama_model_load_from_file(config.model_path.string().c_str(), model_params));
        if (!model)
        {
            std::cerr << "Failed to load GGUF model from: " << config.model_path.string() << '\n';
            return false;
        }

        // KV-cache calculating

        const auto train_ctx = static_cast<uint32_t>(llama_model_n_ctx_train(model.get()));
        const auto ctx = (config.context_size == 0) ? train_ctx : std::min(config.context_size, train_ctx);

        // context params setting

        auto context_params = llama_context_default_params();
        context_params.n_ctx = ctx;
        context_params.embeddings = config.enable_embeddings;

        const auto threads = (config.threads > 0) ? config.threads : 4;
        context_params.n_threads = threads;

        // context creating

        unique_context_ptr context(llama_init_from_model(model.get(), context_params));
        if (!context)
        {
            std::cerr << "Failed to create llama_context for model: " << config.model_path.string() << '\n';
            return false;
        }

        m_model = std::move(model);
        m_context = std::move(context);
        m_config = config;

        // Published last, after m_model/m_context are fully assigned, so a concurrent
        // is_ready() poll from another thread never observes "ready" with a partially
        // constructed engine.
        m_ready.store(true, std::memory_order_release);

        return true;
    }

    auto LlamaEngine::embed(std::string_view data) -> std::vector<float> {
        if (data.empty())
        {
            std::cerr << "Embed error: The data is empty!" << '\n';
            return {};
        }

        if (!is_ready())
        {
            std::cerr << "Embed error: The engine is not initialized!" << '\n';
            return {};
        }

        // Serializes access to m_model/m_context: llama_context is not reentrant,
        // so concurrent embed()/infer() calls from different threads must not
        // interleave their llama_decode()/sampling calls on the same context.
        std::lock_guard<std::mutex> lock(m_mutex);

        llama_memory_clear(llama_get_memory(m_context.get()), true);

        // tokenization

        std::vector<llama_token> tokens = tokenize(data, true);
        if (tokens.empty())
        {
            std::cerr << "Embed error: Failed to tokenize the input data!" << '\n';
            return {};
        }

        // embedding

        if (!decode_tokens(tokens, 0))
        {
            return {};
        }

        const int32_t embedding_dim = llama_model_n_embd(m_model.get());
        std::vector<float> embedding(embedding_dim);

        const auto* last_embedding = llama_get_embeddings_ith(m_context.get(), static_cast<int32_t>(tokens.size()) - 1);
        if (last_embedding == nullptr)
        {
            std::cerr << "Embed error: Cannot get the embedding!" << '\n';
            return {};
        }
        std::copy(last_embedding, last_embedding + embedding_dim, embedding.begin());

        return embedding;
    }

    auto LlamaEngine::infer(std::string_view prompt, const SamplingParams& params) -> std::string {
        if (prompt.empty())
        {
            std::cerr << "Infer error: The prompt cannot be empty!" << '\n';
            return {};
        }

        if (!is_ready())
        {
            std::cerr << "Infer error: The engine is not initialized!" << '\n';
            return {};
        }

        // Serializes access to m_model/m_context: llama_context is not reentrant,
        // so concurrent embed()/infer() calls from different threads must not
        // interleave their llama_decode()/sampling calls on the same context.
        std::lock_guard<std::mutex> lock(m_mutex);

        llama_memory_clear(llama_get_memory(m_context.get()), true);

        // tokenization

        const auto* vocab = llama_model_get_vocab(m_model.get());
        std::vector<llama_token> tokens = tokenize(prompt, true);
        const auto tokens_count = static_cast<int32_t>(tokens.size());

        if (tokens_count <= 0)
        {
            std::cerr << "Infer error: Failed to tokenize the prompt!" << '\n';
            return {};
        }

        // prefill

        if (!decode_tokens(tokens, 0))
        {
            return {};
        }

        // sampler init

        using safe_sampler_ptr = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;

        auto chain_params = llama_sampler_chain_default_params();
        llama_sampler* raw_sampler = llama_sampler_chain_init(chain_params);
        safe_sampler_ptr sampler(raw_sampler, llama_sampler_free);

        // repetition penalty is an implementation detail, not exposed via SamplingParams
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_penalties(64, 1.1f, 0.0f, 0.0f));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_k(params.top_k));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_temp(params.temperature));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_p(params.top_p, 1));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

        // answer generation

        const int32_t max_tokens_in_answer = static_cast<int32_t>(params.max_tokens);
        int32_t generated_tokens_count = 0;
        int32_t generated_token_position = tokens_count;

        std::string answer;
        std::string buffer;
        buffer.resize(64);

        while(generated_tokens_count < max_tokens_in_answer) 
        {
            // inference

            llama_token token = llama_sampler_sample(sampler.get(), m_context.get(), -1);
            llama_sampler_accept(sampler.get(), token);

            if (token == llama_vocab_eos(vocab))
            {
                break;
            }

            int32_t bytes = llama_token_to_piece(vocab, token, buffer.data(), static_cast<int32_t>(buffer.size()), 0, true);

            if(bytes > 0)
            {
                answer.append(buffer.data(), bytes);
            }

            if (const auto stop_at = find_stop_sequence(answer, params.stop_sequences))
            {
                answer.erase(*stop_at);
                break;
            }

            if (!decode_tokens({token}, generated_token_position))
            {
                std::cerr << "Infer error: Cannot decode the generated token!" << '\n';
                break;
            }

            generated_token_position++;
            generated_tokens_count++;
        }

        return answer;
    }

    auto LlamaEngine::is_ready() const noexcept -> bool {
        // Lock-free on purpose: callers (e.g. a UI thread) must be able to poll
        // readiness without blocking on a mutex an in-flight infer() is holding.
        return m_ready.load(std::memory_order_acquire);
    }
} //namespace f1_pulse::ai