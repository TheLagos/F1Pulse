# Building F1 Pulse

| Platform | Compiler | Preset (CPU) | Preset (Vulkan) |
|---|---|---|---|
| Windows x64 | MSVC (Visual Studio 2026) | `windows-msvc-cpu` | `windows-msvc-vulkan` |
| Ubuntu 24.04 WSL2 (x86-64) | GCC 13 | `linux-gcc-cpu` | `linux-gcc-vulkan` |

## 1. Sources

```sh
git clone --recurse-submodules --branch feature/ai-engine https://github.com/TheLagos/F1Pulse.git
cd F1Pulse
```

Already cloned without the submodule (llama.cpp)? Run `git submodule update --init --recursive`.

## 2. Verify the CPU build

Both CPU presets build Release binaries with project tests enabled (`F1_BUILD_TESTS=ON`). CPU verification runs without GGUF weights, a GPU, or Qt. From the repository root, configure, build, and test in order. Run each command only after the previous one succeeds.

The presets use `build/windows-msvc-cpu` and `build/linux-gcc-cpu`, respectively. Give Windows, WSL2, and container builds separate build directories so each CMake cache records the correct compiler, generator, and paths.

CTest's `--no-tests=error` option fails when no tests are registered. Expect one CTest test, `textstream.accumulator`. With `-V`, its output shows seven doctest cases.

### Windows

Install Visual Studio 2026 with the **Desktop development with C++** workload, the x64 MSVC tools, a Windows SDK, and CMake/CTest 4.2 or newer. The project declares CMake 3.22 as its minimum. The Windows preset's `Visual Studio 18 2026` generator was added in CMake 4.2. [CMake generator reference](https://cmake.org/cmake/help/v4.2/generator/Visual%20Studio%2018%202026.html)

Tested on Windows 11 x64 with Visual Studio 2026 Insiders 18.13.0, MSVC 19.52.37032.0, Windows SDK 10.0.26100.0, and CMake/CTest 4.3.1-msvc1.

Open the x64 *Developer PowerShell for Visual Studio 2026*. Check that `cmake --help` lists `Visual Studio 18 2026`, then run the CPU verification commands.

```powershell
cmake --preset windows-msvc-cpu
cmake --build --preset windows-msvc-cpu
ctest --preset windows-msvc-cpu --no-tests=error -V
```

### Linux (Ubuntu 24.04 WSL2)

In Ubuntu, install GCC/G++ 13, CMake/CTest 3.28.x, and Ninja from the Ubuntu packages.

```sh
sudo apt update
sudo apt install build-essential cmake ninja-build
```

Tested on Ubuntu 24.04.4 LTS under WSL2 with GCC/G++ 13.3.0, CMake/CTest 3.28.3, and Ninja 1.11.1. Run the following from the repository root inside Ubuntu.

```sh
cmake --preset linux-gcc-cpu
cmake --build --preset linux-gcc-cpu
ctest --preset linux-gcc-cpu --no-tests=error -V
```

### GitHub Actions

The [CPU CI workflow](.github/workflows/cpu-ci.yml) runs on all branch pushes and on pull requests targeting `develop` or `main`. Pull request jobs build GitHub's test merge commit.

| Check | GitHub-hosted runner | CPU preset |
|---|---|---|
| `CPU / Windows MSVC` | `windows-2025-vs2026` | `windows-msvc-cpu` |
| `CPU / Linux GCC` | `ubuntu-24.04` | `linux-gcc-cpu` |

The jobs run independently and use tools preinstalled on their runners. Each verifies the event revision and the llama.cpp submodule against the checked-out gitlink, then configures, builds all default Release targets, and runs the CPU tests. CTest fails when the test selection is empty. Use the platform commands above to reproduce the build and test locally.

On Windows, CI uses `vswhere` to locate Visual Studio 2026 with C++ tools for CMake, then runs that installation's bundled CMake and CTest by absolute path. Both tools must be at least 4.2, and CMake must provide the `Visual Studio 18 2026` generator. If tool discovery fails, check the runner image and the tool paths and versions in **Verify Windows toolchain**. On Linux, check **Log Linux toolchain** and **Verify selected GCC compilers** for the installed tools and the C and C++ compiler paths selected by CMake. Both compilers must be GCC 13.

#### Merge gate

After both checks pass on GitHub, the repository owner requires them from GitHub Actions on `develop` and on `main` once it exists. Require a pull request, block force pushes, leave routine bypass disabled for all roles, and require zero human approvals. Enable up-to-date checks if compatible with the integration flow.

Verify that an up-to-date integration PR is eligible to merge with both checks passing. For the negative case, the owner creates a disposable branch from the CI-enabled candidate, makes one unit test fail, and opens a PR to the same protected branch. Confirm GitHub blocks the merge because of the failing required check. Record the PR and run links and the tested revisions for both cases. Close the disposable PR without merging it and delete its branch.

## 3. Run the console with a model

Download a chat-tuned GGUF model from Hugging Face and put it in `model/`. The default is **Meta-Llama-3.1-8B-Instruct, Q4_K_M** (about 5 GB).

```
F1Pulse/model/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

`model/` is git-ignored. The chat prompt is written for the **Llama 3** template, so other model families may answer poorly. From the repository root, run the CPU executable for your platform.

```powershell
.\build\windows-msvc-cpu\src\cli\Release\f1_pulse.exe
```

```sh
./build/linux-gcc-cpu/src/cli/f1_pulse
```

### Run options

```
f1_pulse [path/to/model.gguf]
```

Without an argument, the program loads `model/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` from the project root. Type a question and press Enter; type `exit` to quit.

## Optional GPU build (Vulkan)

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
| `cmake` or `ctest` missing or the wrong version on Windows | Open the VS 2026 Developer PowerShell. Run `Get-Command cmake,ctest` to check the resolved tools, then check both versions. |
| `Visual Studio 18 2026` missing from `cmake --help` | Use CMake 4.2 or newer. If CMake lists the generator but cannot find the instance, check the VS 2026 C++ workload and Windows SDK in Visual Studio Installer. |
| `gcc`, `g++`, `cmake`, `ctest`, or `ninja` missing in Ubuntu | Install `build-essential`, `cmake`, and `ninja-build` in the Ubuntu WSL2 distribution used for the build. |
| Model not found | Put the `.gguf` into `model/` or pass its path as an argument. |
| Process dies on start (`Killed`) | Not enough RAM. In Docker/WSL2 raise the limit in `.wslconfig`, or use a smaller model. |
| Vulkan configure stops with a message about the SDK | Install the Vulkan SDK and make sure `glslc` is available. |
| Vulkan build fails with `MSB6003` / "compiler broken" | Path too long, see the section above. |
| Vulkan build runs out of GPU memory | Lower `gpu_layers` in `src/cli/main.cpp`. |
| Garbled characters in the console | Use a font with Cyrillic support. The program already switches the Windows console to UTF-8. |
| Cache refers to a different build environment | Check which preset and build directory you selected. Use an independent directory for each environment. |

## Other commands

| Command | Purpose |
|---|---|
| `cmake --list-presets` | presets available on this OS |
| `cmake --preset <name> --fresh` | recreate the cache and `CMakeFiles/` in the selected build directory (CMake 3.24+) |
| `cmake --build --preset <name> --parallel 8` | build with 8 jobs |
| `cmake --build --preset <name> --verbose` | print full compiler commands |
| `ctest --preset <name> --no-tests=error --output-on-failure` | fail when no tests run and show output of failing tests |

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
