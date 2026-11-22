# F1 Pulse

A local, offline AI assistant for Formula 1 telemetry analysis, written in modern C++23 on top of [llama.cpp](https://github.com/ggerganov/llama.cpp).

The goal: ingest live and historical telemetry (OpenF1), keep it in an embedded database, and let a local language model answer questions about it with retrieval-augmented generation (RAG), through a Qt desktop interface. No cloud, everything runs on the user's machine.

## Status: work in progress (pre-0.1)

Working today:

- Local LLM inference through llama.cpp: model loading, batched prompt prefill, sampling (temperature, top-k, top-p, greedy), stop sequences
- Streaming generation through events (`TextChunk`, `Done`, `Cancelled`, `Failed`) with cooperative cancellation (`std::stop_token`)
- UTF-8-safe output: a token can end in the middle of a multi-byte character, so the `textstream` module buffers incomplete characters and stop-sequence prefixes
- Text embeddings through the same loaded model
- Error handling without exceptions (`std::expected` with typed error codes)
- In-memory vector store with cosine similarity (placeholder for a real database)
- Console chat with generation on a separate thread
- GPU acceleration through Vulkan (verified on Windows with a 12 GB GPU)
- Modular CMake build for Windows and Linux, unit tests for the streaming layer

Planned:

- [ ] Inference service: task queue, worker thread, cancellation of the current task (the last piece before 0.1)
- [ ] Telemetry ingestion from OpenF1
- [ ] Embedded database (SQLite) instead of the in-memory store
- [ ] RAG over telemetry
- [ ] Hardware-aware configuration
- [ ] Qt/QML dashboard
- [ ] Generic chat-template handling (currently Llama 3 only)

## Quick start

```sh
git clone --recurse-submodules <https://github.com/TheLagos/F1Pulse.git>
cd F1Pulse
```

Put a model into `model/` (default: Meta-Llama-3.1-8B-Instruct, Q4_K_M, about 5 GB), then:

```powershell
# Windows (PowerShell)
cmake --preset windows-msvc-cpu
cmake --build --preset windows-msvc-cpu
.\build\windows-msvc-cpu\src\cli\Release\f1_pulse.exe
```

```sh
# Linux (Ubuntu 24.04, WSL, Docker)
cmake --preset linux-gcc-cpu
cmake --build --preset linux-gcc-cpu
./build/linux-gcc-cpu/src/cli/f1_pulse
```

Prerequisites, Vulkan build, tests and troubleshooting: [BUILD.md](BUILD.md).

## Platform support

| Platform | Compiler | CPU | Vulkan |
|---|---|---|---|
| Windows 10/11 | MSVC (Visual Studio 2026) | verified | verified |
| Ubuntu 24.04 (WSL, Docker) | GCC 13 | verified | not verified |

## Architecture

```
cli          console chat (to be replaced by the Qt UI)
  |
services     inference service / orchestrator        (planned)
  |
ai           IAIEngine interface, LlamaEngine (llama.cpp wrapper)
  |    \
  |     textstream   UTF-8 and stop-sequence streaming buffer
  |
storage      vector store (in-memory)  ->  math (cosine similarity)
```

Dependencies point only downwards. Each module is a separate static library with its own `CMakeLists.txt`, and modules that do not need llama.cpp (`textstream`, `math`) build and test without it.

| Module | Responsibility |
|---|---|
| `src/ai` | `IAIEngine` interface and `LlamaEngine`: init, tokenize, embed, streaming infer |
| `src/textstream` | `Accumulator`: turns raw token bytes into safe-to-display text |
| `src/math` | Vector math |
| `src/storage` | Document vector storage |
| `src/cli` | Console application |

### Design notes

- **One model, one engine.** The same model serves generation and embeddings. `llama_context` is not reentrant, so access is serialized with a mutex.
- **Events instead of return values.** `infer()` returns nothing; every run ends with exactly one terminal event (`Done`, `Cancelled` or `Failed`). The callback runs on the thread that called `infer()`, so a UI must forward events to its own thread.
- **No exceptions across the engine boundary.** Errors are values: `std::expected<T, EngineError>` for one-shot calls, a `Failed` event for streaming.
- **Cooperative cancellation** through `std::stop_token`, checked between prompt chunks and between generated tokens.

## Known limitations

- The prompt template is hard-coded for Llama 3.
- Special-token markers are parsed in the whole prompt, including user text. Fine for a local console, but untrusted data must be filtered before it goes into prompts.
- The vector store is in memory and is lost on exit.
- Generation holds the engine lock, so embedding calls wait until it finishes.