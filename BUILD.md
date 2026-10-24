# Building F1 Pulse

| Platform | Compiler | Preset (CPU) | Preset (Vulkan) |
|---|---|---|---|
| Windows 10/11 | MSVC (Visual Studio 2026) | `windows-msvc-cpu` | `windows-msvc-vulkan` |
| Ubuntu 24.04 (also WSL, Docker) | GCC 13 | `linux-gcc-cpu` | `linux-gcc-vulkan` |

A build is three separate commands: **configure** (`cmake --preset`), **build** (`cmake --build --preset`), **test** (`ctest --preset`). Run them one per line, so a failed configure does not trigger a build.

## 1. Sources

```sh
git clone --recurse-submodules <https://github.com/TheLagos/F1Pulse.git>
cd F1Pulse
```

Already cloned without the submodule (llama.cpp)? Run `git submodule update --init --recursive`.

## 2. Model

Download a chat-tuned GGUF model from Hugging Face and put it into `model/`. The default is **Meta-Llama-3.1-8B-Instruct, Q4_K_M** (about 5 GB):

```
F1Pulse/model/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

`model/` is git-ignored. The chat prompt is written for the **Llama 3** template, so other model families may answer poorly.

## 3. Build and run

### Windows

Install Visual Studio 2026 with the **Desktop development with C++** workload (includes MSVC and CMake). Use *Developer PowerShell for Visual Studio*:

```powershell
cmake --preset windows-msvc-cpu
cmake --build --preset windows-msvc-cpu
ctest --preset windows-msvc-cpu
.\build\windows-msvc-cpu\src\cli\Release\f1_pulse.exe
```

### Linux (Ubuntu 24.04, WSL, Docker)

```sh
sudo apt update
sudo apt install build-essential cmake ninja-build
```

```sh
cmake --preset linux-gcc-cpu
cmake --build --preset linux-gcc-cpu
ctest --preset linux-gcc-cpu
./build/linux-gcc-cpu/src/cli/f1_pulse
```

### Run options

```
f1_pulse [path/to/model.gguf]
```

Without an argument, the program loads `model/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` from the project root. Type a question and press Enter; type `exit` to quit.

## GPU (Vulkan)

The backend is chosen at **build time**, so the Vulkan build is a separate build folder and a separate executable. With a Vulkan build, the layers set by `gpu_layers` are loaded into video memory and computed on the GPU, which makes generation much faster than on the CPU. The rest of the model stays in RAM. A CPU build ignores `gpu_layers`.

Requirements: a GPU with a Vulkan driver and the **Vulkan SDK including `glslc`**.

- Windows: install the LunarG Vulkan SDK, then reopen the shell.
- Ubuntu: `sudo apt install libvulkan-dev glslc` (not verified).

```powershell
cmake --preset windows-msvc-vulkan
cmake --build --preset windows-msvc-vulkan
```

The first Vulkan build is slow because hundreds of shaders are compiled.

To check that the GPU is used, run `f1_pulse.exe 2> run.log` and look in `run.log` for the detected Vulkan device and the number of layers offloaded to it.

**Windows path-length problem.** The Vulkan build creates very deep folders. MSBuild may fail with `The C compiler ... is not able to compile a simple test program` and `MSB6003 ... DirectoryNotFoundException` even though the compiler is fine. The fix is a short build folder. Create `CMakeUserPresets.json` in the project root (it is git-ignored). Its `version` must equal the one in `CMakePresets.json`:

```json
{
  "version": 3,
  "configurePresets": [
    { "name": "wvk", "inherits": "windows-msvc-vulkan", "binaryDir": "D:/b/vk" }
  ],
  "buildPresets": [
    { "name": "wvk", "inherits": "windows-msvc-vulkan", "configurePreset": "wvk" }
  ]
}
```

Then use `cmake --preset wvk` and `cmake --build --preset wvk`; the executable is in `D:\b\vk\src\cli\Release\`.

CUDA and Metal can be selected with `-DF1_GPU_BACKEND=CUDA|METAL`, but they have no presets and are untested.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Could not use disabled preset` | The preset belongs to another OS. Run `cmake --list-presets`. |
| `llama.h` or llama.cpp sources not found | Submodule is missing: `git submodule update --init --recursive`. |
| `cmake` not found on Windows | Use *Developer PowerShell for Visual Studio*. |
| Model not found | Put the `.gguf` into `model/` or pass its path as an argument. |
| Process dies on start (`Killed`) | Not enough RAM. In Docker/WSL2 raise the limit in `.wslconfig`, or use a smaller model. |
| Vulkan configure stops with a message about the SDK | Install the Vulkan SDK and make sure `glslc` is available. |
| Vulkan build fails with `MSB6003` / "compiler broken" | Path too long, see the section above. |
| Vulkan build runs out of GPU memory | Lower `gpu_layers` in `src/cli/main.cpp`. |
| Garbled characters in the console | Use a font with Cyrillic support. The program already switches the Windows console to UTF-8. |
| Odd behaviour after changing options | Delete `build/<preset>/` and configure again (CMake caches old values). |

## Other commands

| Command | Purpose |
|---|---|
| `cmake --list-presets` | presets available on this OS |
| `cmake --preset <name> --fresh` | configure ignoring the cache |
| `cmake --build --preset <name> --parallel 8` | build with 8 jobs |
| `cmake --build --preset <name> --verbose` | print full compiler commands |
| `ctest --preset <name> --output-on-failure` | show output of failing tests |

## Formatting and new modules

`.clang-format` defines the style. The scripts format only `src/`, never `third_party/`:

```sh
scripts/format.sh            # or --check
```

```powershell
.\scripts\format.ps1         # or -Check
```

To add a module, create `src/<name>/CMakeLists.txt`:

```cmake
f1_add_module(<name>
    SOURCES MyFile.cpp
    PRIVATE_DEPS f1::textstream)
```

and add `add_subdirectory(<name>)` to `src/CMakeLists.txt`. Unit tests in `src/<name>/tests/` are picked up automatically.