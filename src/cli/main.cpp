#include "ai/LlamaEngine.hpp"

#include <condition_variable>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <variant>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace ai = f1_pulse::ai;

namespace
{
/// Thread-safe queue: the generation thread pushes events, the main thread pops them.
class EventQueue
{
  public:
    void push(ai::GenerationEvent event)
    {
        {
            std::lock_guard lock(m_mutex);
            m_events.push(std::move(event));
        }
        m_cv.notify_one();
    }

    auto pop() -> ai::GenerationEvent
    {
        std::unique_lock lock(m_mutex);
        m_cv.wait(lock, [this] { return !m_events.empty(); });
        auto event = std::move(m_events.front());
        m_events.pop();
        return event;
    }

  private:
    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::queue<ai::GenerationEvent> m_events;
};

/// Llama 3.x chat template. BOS is NOT written here: tokenize(add_special = true) adds it.
auto build_llama3_prompt(const std::string& user_message) -> std::string
{
    return "<|start_header_id|>system<|end_header_id|>\n\n"
           "You are Rookie, a concise and helpful assistant.<|eot_id|>"
           "<|start_header_id|>user<|end_header_id|>\n\n" +
           user_message +
           "<|eot_id|>"
           "<|start_header_id|>assistant<|end_header_id|>\n\n";
}

/// Prints events until a terminal one (Done / Cancelled / Failed) arrives.
auto print_events(EventQueue& queue) -> void
{
    while (true)
    {
        const auto event = queue.pop();

        if (const auto* chunk = std::get_if<ai::TextChunk>(&event))
        {
            std::cout << chunk->text << std::flush;
        }
        else if (std::holds_alternative<ai::Done>(event))
        {
            std::cout << "\n";
            return;
        }
        else if (std::holds_alternative<ai::Cancelled>(event))
        {
            std::cout << "\n[cancelled]\n";
            return;
        }
        else if (const auto* failed = std::get_if<ai::Failed>(&event))
        {
            std::cout << "\n[error] " << failed->error.message << "\n";
            return;
        }
    }
}

auto default_model_path() -> std::filesystem::path
{
    return std::filesystem::path(F1_DEFAULT_MODEL_DIR) / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf";
}
} // namespace

int main(int argc, char** argv)
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    ai::EngineConfig config;
    config.model_path = (argc > 1) ? std::filesystem::path(argv[1]) : default_model_path();
    config.context_size = 4096;
    config.gpu_layers = -1;

    ai::LlamaEngine engine;

    std::cout << "Loading model: " << config.model_path.string() << "\n";
    if (const auto result = engine.init(config); !result)
    {
        std::cerr << "Failed to load model: " << result.error().message << "\n";
        return 1;
    }
    std::cout << "Model loaded. Type 'exit' to quit.\n";

    const ai::SamplingParams params;
    std::string input;

    while (true)
    {
        std::cout << "\nYou >> ";

        if (!std::getline(std::cin, input) || input == "exit")
        {
            break;
        }

        if (input.empty())
        {
            continue;
        }

        EventQueue queue;
        const std::string prompt = build_llama3_prompt(input);

        // Declared last: destroyed first, so the thread is joined before queue/prompt die.
        std::jthread worker(
            [&](std::stop_token stoken) {
                engine.infer(prompt, [&queue](const ai::GenerationEvent& event) { queue.push(event); }, stoken, params);
            });

        std::cout << "Rookie >> ";
        print_events(queue);
    }

    return 0;
}