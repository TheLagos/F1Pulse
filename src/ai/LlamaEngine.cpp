#include "LlamaEngine.hpp"
#include "llama.h"

#include <algorithm>
#include <iostream>
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
        return "ag";
    }

    auto LlamaEngine::is_ready() const noexcept -> bool {
        return (m_model) && (m_context);
    }
} //namespace f1_pulse::ai