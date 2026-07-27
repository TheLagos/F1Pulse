#include "LlamaEngine.hpp"
#include "llama.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace f1_pulse::ai 
{
    LlamaEngine::~LlamaEngine() 
    {
        if (m_context)
        {
            llama_free(m_context);
        }

        if (m_model)
        {
            llama_model_free(m_model);
        }
    }

    auto LlamaEngine::init(const EngineConfig& config) -> bool {

        if (m_model || m_context)
        {
            std::cerr << "Error: Model is already initialized!" << '\n';
            return false;
        }

        static bool backend_init = []() {
            llama_backend_init();
            return true;
        }();

        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = (config.gpu_layers >= 0) ? config.gpu_layers : -1;

        auto model = llama_model_load_from_file(config.model_path.c_str(), model_params);
        if (model == nullptr)
        {
            std::cerr << "Error: Cannot initialize the model!" << '\n';
            return false;
        }

        auto context_params = llama_context_default_params();
        context_params.n_ctx = config.context_size;
        context_params.embeddings = true;

        auto context = llama_init_from_model(model, context_params);
        if (context == nullptr)
        {
            std::cerr << "Error: Cannot initialize the model's context!" << '\n';
            llama_model_free(model);
            return false;
        }
        
        m_model = model;
        m_context = context;

        return true;
    }

    auto LlamaEngine::embed(const std::string& data) -> std::vector<float> {
        if (data.empty())
        {
             std::cerr << "Embed error: The data is empty!" << '\n';
             return {};
        }
        
        llama_memory_clear(llama_get_memory(m_context), true);

        // tokenization

        auto vocab = llama_model_get_vocab(m_model);
        // ckecks buffer size and makes it positive
        int32_t tokens_count = llama_tokenize(vocab, data.c_str(), static_cast<int32_t>(data.length()), nullptr, 0, true, false) * -1;

        std::vector<llama_token> tokens(tokens_count);
        llama_tokenize(vocab, data.c_str(), static_cast<int32_t>(data.length()), tokens.data(), tokens.size(), true, false);

        // batching

        auto batch = llama_batch_init(tokens.size(), 0, 1);
        batch.n_tokens = tokens_count;

        for (int i = 0; i < tokens_count; ++i)
        {
            batch.token[i] = tokens[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (i == tokens_count - 1);
        }

        // embedding

        int32_t decode_status = llama_decode(m_context, batch);
        if(decode_status != 0)
        {
            std::cerr << "Embed error: Cannot decode the batch, error code - " << decode_status << "!" << '\n';
            llama_batch_free(batch);
            return {};
        }

        int32_t embedding_dim = llama_model_n_embd(m_model);
        std::vector<float> embedding(embedding_dim);

        auto last_embedding = llama_get_embeddings_ith(m_context, tokens_count - 1);
        if (last_embedding == nullptr)
        {
            std::cerr << "Embed error: Cannot get the embedding!" << '\n';
            llama_batch_free(batch);
            return {};
        }
        std::copy(last_embedding, last_embedding + embedding_dim, embedding.begin());

        llama_batch_free(batch);
        return embedding;
    }

    auto LlamaEngine::infer(const std::string& prompt) -> std::string {
        if (prompt.empty())
        {
            std::cerr << "Infer error: The prompt cannot be empty!" << '\n';
            return {};
        }

        llama_memory_clear(llama_get_memory(m_context), true);

        // tokenization

        auto vocab = llama_model_get_vocab(m_model);
        int32_t tokens_count = llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()), nullptr, 0, true, false) * -1;

        std::vector<llama_token> tokens(tokens_count);
        llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()), tokens.data(), tokens.size(), true, false);

        // batching

        auto batch = llama_batch_init(tokens_count, 0, 1);
        batch.n_tokens = tokens_count;
        for (int i = 0; i < tokens_count; ++i) 
        {
            batch.token[i] = tokens[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (i == tokens_count - 1);
        }

        // prefill

        int32_t decode_status = llama_decode(m_context, batch);
        if (decode_status != 0) 
        {
            std::cerr << "Embed error: Cannot decode the batch, error code - " << decode_status << "!" << '\n';
            llama_batch_free(batch);
            return {};
        }

        // sampler init

        using safe_sampler_ptr = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;

        auto chain_params = llama_sampler_chain_default_params();
        llama_sampler* raw_sampler = llama_sampler_chain_init(chain_params);
        safe_sampler_ptr sampler(raw_sampler, llama_sampler_free);

        llama_sampler_chain_add(sampler.get(), llama_sampler_init_penalties(64, 1.1f, 0.0f, 0.0f));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_k(20));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_temp(0.35f));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_p(0.7f, 1));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

        // answer generation

        int32_t max_tokens_in_answer = 1000;
        int32_t generated_tokens_count = 0;
        int32_t generated_token_position = tokens_count;

        std::string answer;
        std::string buffer;
        buffer.resize(64);

        while(generated_tokens_count < max_tokens_in_answer) 
        {
            // inference

            llama_token token = llama_sampler_sample(sampler.get(), m_context, -1);
            llama_sampler_accept(sampler.get(), token);

            if (token == llama_vocab_eos(vocab))
            {
                break;
            }

            int32_t bytes = llama_token_to_piece(vocab, token, &buffer[0], sizeof(buffer), 0, true);

            if(bytes > 0)
            {
                std::string_view word(buffer.data(), bytes);
                std::cout << word;
                answer.append(word);
            }

            batch.n_tokens = 0;
            batch.token[0] = token;
            batch.pos[0] = generated_token_position;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = true;
            batch.n_tokens = 1;

            llama_decode(m_context, batch);

            generated_token_position++;
            generated_tokens_count++;
        }

        llama_batch_free(batch);
        return answer;
    }

    auto LlamaEngine::is_ready() const noexcept -> bool {
        return (m_model) && (m_context);
    }
} //namespace f1_pulse::ai