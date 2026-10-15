#include "LlamaEngine.hpp"
#include "IAIEngine.hpp"
#include "llama.h"
#include "textstream/Accumulator.hpp"

#include <algorithm>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <memory>
#include <mutex>
#include <cstdint>
#include <span>

namespace
{
    /// @brief
    enum class StopReason
    {
        EndOfGeneration,
        MaxTokens,
        StopSequence,
        Cancelled,
        DecodeFailed,
        PieceFailed
    };
} // namespace

namespace f1_pulse::ai
{
    using enum EngineErrorCode;

    void ModelCleaner::operator()(llama_model *model) const noexcept
    {
        if (model)
        {
            llama_model_free(model);
        }
    }

    void ContextCleaner::operator()(llama_context *context) const noexcept
    {
        if (context)
        {
            llama_free(context);
        }
    }

    auto LlamaEngine::tokenize(std::string_view text, bool add_special) const -> std::vector<llama_token>
    {
        const auto *vocab = llama_model_get_vocab(m_model.get());
        const auto text_size = static_cast<int32_t>(text.size());

        int32_t tokens_count = llama_tokenize(vocab, text.data(), text_size, nullptr, 0, add_special, false) * -1;

        if (tokens_count <= 0)
        {
            return {};
        }

        std::vector<llama_token> tokens(tokens_count);
        int32_t result = llama_tokenize(vocab, text.data(), text_size, tokens.data(), tokens_count, add_special, false);

        if (result < 0 || result != tokens_count)
        {
            return {};
        }

        return tokens;
    }

    auto LlamaEngine::decode_tokens(std::span<const llama_token> tokens, int32_t start_pos) const -> bool
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
            llama_batch &batch;
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
            return false;
        }

        return true;
    }

    auto LlamaEngine::init(const EngineConfig &config) -> std::expected<void, EngineError>
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        if (is_ready())
        {
            return std::unexpected(EngineError{
                AlreadyInitialized,
                "LlamaEngine is already initialized!"});
        }

        if (config.model_path.empty())
        {
            return std::unexpected(EngineError{
                ModelFileNotFound,
                "Model path is empty!"});
        }

        if (!std::filesystem::is_regular_file(config.model_path))
        {
            return std::unexpected(EngineError{
                ModelFileNotFound,
                "Model file not found or it's not a regular file!"});
        }

        static bool backend_init = []()
        {
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
            return std::unexpected(EngineError{
                ModelLoadFailed,
                "Failed to load GGUF model from " + config.model_path.string()});
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
            return std::unexpected(EngineError{
                ContextCreationFailed,
                "Failed to create llama_context for model: " + config.model_path.string()});
        }

        m_model = std::move(model);
        m_context = std::move(context);
        m_config = config;

        // Published last, after m_model/m_context are fully assigned, so a concurrent
        // is_ready() poll from another thread never observes "ready" with a partially
        // constructed engine.
        m_ready.store(true, std::memory_order_release);

        return {};
    }

    auto LlamaEngine::embed(std::string_view data) -> std::expected<std::vector<float>, EngineError>
    {
        if (data.empty())
        {
            return std::unexpected(EngineError{
                EmptyInput,
                "Embed error: The input data is empty!"});
        }

        if (!is_ready())
        {
            return std::unexpected(EngineError{
                NotInitialized,
                "Embed error: The engine is not initialized!"});
        }

        if (!m_config.enable_embeddings)
        {
            return std::unexpected(EngineError{
                EmbeddingsDisabled,
                "Embed error: Embeddings are disabled in the configuration!"});
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
            return std::unexpected(EngineError{
                TokenizationFailed,
                "Embed error: Failed to tokenize the input data!"});
        }

        const auto limit = std::min(
            static_cast<std::size_t>(llama_n_ctx(m_context.get())),
            static_cast<std::size_t>(llama_n_batch(m_context.get())));

        if (tokens.size() > limit)
        {
            return std::unexpected(EngineError{
                ContextOverflow,
                "Embed error: input (" + std::to_string(tokens.size()) + " tokens) exceeds limit (" + std::to_string(limit) + ")"});
        }

        // embedding

        if (!decode_tokens(tokens, 0))
        {
            return std::unexpected(EngineError{
                DecodeFailed,
                "Embed error: Failed to decode tokens!"});
        }

        const int32_t embedding_dim = llama_model_n_embd(m_model.get());
        if (embedding_dim <= 0)
        {
            return std::unexpected(EngineError{
                EmbeddingsExtractionFailed,
                "Embed error: Model reports invalid embedding dimension (engine or model inconsistency)!"});
        }

        std::vector<float> embedding(embedding_dim);
        const float *embeddings = nullptr;

        if (llama_pooling_type(m_context.get()) == LLAMA_POOLING_TYPE_NONE)
        {
            embeddings = llama_get_embeddings_ith(m_context.get(), static_cast<int32_t>(tokens.size()) - 1);
        }
        else
        { // pooled: one vector per sequence, seq id 0
            embeddings = llama_get_embeddings_seq(m_context.get(), 0);
        }

        if (embeddings == nullptr)
        {
            return std::unexpected(EngineError{
                EmbeddingsExtractionFailed,
                "Embed error: Cannot extract the embedding!"});
        }
        std::copy(embeddings, embeddings + embedding_dim, embedding.begin());

        return embedding;
    }

    auto LlamaEngine::infer(std::string_view prompt, const CallbackFunction &callback, std::stop_token stoken, const SamplingParams &params) -> void
    {
        if (stoken.stop_requested())
        {
            callback(Cancelled{});
            return;
        }

        if (prompt.empty())
        {
            callback(Failed{EngineError{
                EmptyInput,
                "Infer error: The prompt cannot be empty!"}});
            return;
        }

        if (!is_ready())
        {
            callback(Failed{EngineError{
                NotInitialized,
                "Infer error: The engine is not initialized!"}});
            return;
        }

        // Serializes access to m_model/m_context: llama_context is not reentrant,
        // so concurrent embed()/infer() calls from different threads must not
        // interleave their llama_decode()/sampling calls on the same context.
        std::lock_guard<std::mutex> lock(m_mutex);

        llama_memory_clear(llama_get_memory(m_context.get()), true);

        // tokenization

        const auto *vocab = llama_model_get_vocab(m_model.get());
        std::vector<llama_token> tokens = tokenize(prompt, true);
        const auto tokens_count = static_cast<int32_t>(tokens.size());

        if (tokens_count <= 0)
        {
            callback(Failed{EngineError{
                TokenizationFailed,
                "Infer error: Failed to tokenize the prompt!"}});
            return;
        }

        // overflow checking

        const auto n_ctx = static_cast<std::size_t>(llama_n_ctx(m_context.get()));
        const std::size_t prompt_tokens = tokens.size();
        const std::size_t requested = params.max_tokens;

        if ((prompt_tokens + requested) > n_ctx)
        {
            callback(Failed{EngineError{
                ContextOverflow,
                "Infer error: prompt (" + std::to_string(prompt_tokens) + " tokens) + max_tokens (" + std::to_string(requested) + ") exceeds context size (" + std::to_string(n_ctx) + ")"}});
            return;
        }

        // prefill, chunked by n_batch

        const auto n_batch = static_cast<std::size_t>(llama_n_batch(m_context.get()));
        for (std::size_t offset = 0; offset < prompt_tokens; offset += n_batch)
        {
            if (stoken.stop_requested())
            {
                callback(Cancelled{});
                return;
            }

            const std::size_t len = std::min(n_batch, prompt_tokens - offset);
            const std::span<const llama_token> chunk(tokens.data() + offset, len);

            if (!decode_tokens(chunk, static_cast<int32_t>(offset)))
            {
                callback(Failed{EngineError{
                    DecodeFailed,
                    "Infer error: Failed to decode prompt chunk at offset " + std::to_string(offset)}});
                return;
            }
        }

        // sampler init

        using safe_sampler_ptr = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;

        auto chain_params = llama_sampler_chain_default_params();
        llama_sampler *raw_sampler = llama_sampler_chain_init(chain_params);
        safe_sampler_ptr sampler(raw_sampler, llama_sampler_free);

        // repetition penalty is an implementation detail, not exposed via SamplingParams
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_penalties(64, 1.1f, 0.0f, 0.0f));

        if (params.temperature <= 0.0f)
        {
            llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());
        }
        else
        {
            llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_k(params.top_k));
            llama_sampler_chain_add(sampler.get(), llama_sampler_init_temp(params.temperature));
            llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_p(params.top_p, 1));
            llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
        }

        // answer generation

        const auto max_tokens_in_answer = static_cast<int32_t>(params.max_tokens);
        int32_t generated_tokens_count = 0;
        int32_t generated_token_position = tokens_count;

        textstream::Accumulator accumulator(params.stop_sequences);
        std::string piece(64, '\0');

        StopReason stop_reason{StopReason::MaxTokens};
        while (generated_tokens_count < max_tokens_in_answer)
        {
            // inference

            if (stoken.stop_requested())
            {
                stop_reason = StopReason::Cancelled;
                break;
            }

            llama_token token = llama_sampler_sample(sampler.get(), m_context.get(), -1);
            llama_sampler_accept(sampler.get(), token);

            if (llama_vocab_is_eog(vocab, token))
            {
                stop_reason = StopReason::EndOfGeneration;
                break;
            }

            int32_t bytes = llama_token_to_piece(vocab, token, piece.data(), static_cast<int32_t>(piece.size()), 0, false);

            if (bytes < 0)
            {
                piece.resize(static_cast<std::size_t>(-bytes));
                bytes = llama_token_to_piece(vocab, token, piece.data(), static_cast<int32_t>(piece.size()), 0, false);
            }

            if (bytes < 0)
            {
                stop_reason = StopReason::PieceFailed;
                break;
            }

            auto pushed = accumulator.push(std::string_view(piece.data(), static_cast<std::size_t>(bytes)));

            if (!pushed.ready.empty())
            {
                callback(TextChunk{
                    std::move(pushed.ready)});
            }

            if (pushed.stop_hit)
            {
                stop_reason = StopReason::StopSequence;
                break;
            }

            if (!decode_tokens(std::span<const llama_token>(&token, 1), generated_token_position))
            {
                stop_reason = StopReason::DecodeFailed;
                break;
            }

            generated_token_position++;
            generated_tokens_count++;
        }

        switch (stop_reason)
        {
        case StopReason::MaxTokens:
        case StopReason::EndOfGeneration:
        case StopReason::StopSequence:
        {
            auto tail = accumulator.flush();
            if (!tail.empty())
            {
                callback(TextChunk{
                    std::move(tail)});
            }
            callback(Done{});
            break;
        }
        case StopReason::Cancelled:
            callback(Cancelled{});
            break;
        case StopReason::DecodeFailed:
            callback(Failed{EngineError{
                DecodeFailed,
                "Infer error: Cannot decode the generated token!"}});
            break;
        case StopReason::PieceFailed:
            callback(Failed{EngineError{
                DecodeFailed,
                "Infer error: Cannot convert the generated token to text!"}});
            break;
        }
    }

    auto LlamaEngine::is_ready() const noexcept -> bool
    {
        // Lock-free on purpose: callers (e.g. a UI thread) must be able to poll
        // readiness without blocking on a mutex an in-flight infer() is holding.
        return m_ready.load(std::memory_order_acquire);
    }
} // namespace f1_pulse::ai