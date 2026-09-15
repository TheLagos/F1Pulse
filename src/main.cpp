#include "ai/LlamaEngine.hpp"
#include <iostream>
#include <filesystem>
#ifdef _WIN32
#include <windows.h>
#endif

int main() {
    f1_pulse::ai::LlamaEngine engine;
    std::string prompt;

#ifdef _WIN32
    char buf[MAX_PATH];
    GetModuleFileNameA(nullptr, buf, MAX_PATH);
    auto exe_dir = std::filesystem::path(buf).parent_path();
#else
    auto exe_dir = std::filesystem::canonical("/proc/self/exe").parent_path();
#endif
    auto model_path = exe_dir.parent_path() / "model" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf";

    engine.init({model_path.string(), 4096, -1});

    while (true)
    {
        std::cout << "Ask Rookie about something >> ";

        if(!std::getline(std::cin, prompt) || prompt == "exit")
        {
            break;
        }

        std::cout << "\nRookie >> ";
        std::cout << engine.infer(prompt) << std::endl;
    }

    return 0;
}