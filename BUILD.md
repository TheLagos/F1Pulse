# Building F1Pulse

Supported platforms:

| Platform | Compiler | Generator |
|---|---|---|
| Windows 10/11 | MSVC (Visual Studio 2022) | Visual Studio 17 2022 |
| Ubuntu 24.04 (incl. WSL) | GCC 13 | Ninja |

llama.cpp is a git submodule. Fetch it once after cloning:

```sh
git submodule update --init --recursive
```

## Prerequisites

### Windows

- Visual Studio 2022 with the **Desktop development with C++** workload (includes MSVC and CMake).
- Optional, for formatting: the **C++ Clang tools for Windows** component (provides `clang-format`).

Run the commands below from a **Developer PowerShell for VS 2022**.

### Ubuntu 24.04

```sh
sudo apt install build-essential cmake ninja-build clang-format
```

## Build

### Windows (MSVC, CPU)

```powershell
cmake --preset windows-msvc-cpu; cmake --build --preset windows-msvc-cpu
```

Executable: `build\windows-msvc-cpu\src\cli\Release\f1_pulse.exe`

### Ubuntu 24.04 (GCC 13, CPU)

```sh
cmake --preset linux-gcc-cpu && cmake --build --preset linux-gcc-cpu
```

Executable: `build/linux-gcc-cpu/src/cli/f1_pulse`

### Run

```sh
f1_pulse <path/to/model.gguf>
```

Without an argument, `f1_pulse` looks for `../model/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` relative to the directory of the executable.

## Choosing the GPU backend

The backend is selected with the `F1_GPU_BACKEND` cache variable: `CPU` (default), `VULKAN`, `CUDA` or `METAL` (Apple only).

Presets for CPU and Vulkan exist for both platforms:

| Preset | Backend |
|---|---|
| `windows-msvc-cpu`, `linux-gcc-cpu` | CPU |
| `windows-msvc-vulkan`, `linux-gcc-vulkan` | Vulkan |

Vulkan needs the Vulkan SDK including the `glslc` shader compiler; configuration stops with an explanatory error if it is missing.

- Windows: install the LunarG Vulkan SDK and reopen the shell so `VULKAN_SDK` is set.
- Ubuntu 24.04: `sudo apt install libvulkan-dev glslc`

For CUDA (no preset), pass the variable explicitly, e.g.:

```sh
cmake --preset linux-gcc-cpu -DF1_GPU_BACKEND=CUDA -B build/linux-gcc-cuda
```

Other options:

| Option | Default | Meaning |
|---|---|---|
| `F1_BUILD_TESTS` | `OFF` (`ON` in all presets) | Build unit tests |
| `F1_WARNINGS_AS_ERRORS` | `OFF` | `/WX` or `-Werror` for project code |

## Tests

Tests are enabled in all presets. After building:

```powershell
ctest --preset windows-msvc-cpu
```

```sh
ctest --preset linux-gcc-cpu
```

## Formatting

Style is defined in `.clang-format`. The scripts format only `src/` (`*.cpp`, `*.hpp`, `*.h`) and never touch `third_party/`.

```sh
scripts/format.sh           # format in place
scripts/format.sh --check   # change nothing, non-zero exit if a file is not formatted
```

```powershell
.\scripts\format.ps1          # format in place
.\scripts\format.ps1 -Check   # change nothing, non-zero exit if a file is not formatted
```

If PowerShell blocks the script, run it with `powershell -ExecutionPolicy Bypass -File scripts\format.ps1`.

## Adding a module

Create `src/<name>/CMakeLists.txt`:

```cmake
f1_add_module(<name>
    SOURCES MyFile.cpp
    PRIVATE_DEPS f1::textstream)
```

and add `add_subdirectory(<name>)` to `src/CMakeLists.txt`. This creates the static library `f1::<name>`. If `src/<name>/tests/CMakeLists.txt` exists, it is picked up when `F1_BUILD_TESTS` is `ON`.
