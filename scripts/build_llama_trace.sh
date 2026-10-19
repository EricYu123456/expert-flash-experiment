#!/usr/bin/env bash
set -euo pipefail
experiment_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
storage_dir=${EXPERIMENT_STORAGE:-/home/nfs_client/ericyu/expert_flash_experiment}
source_commit=927dada6c9143ba4c940b72004d2698fa5e4e930
source_dir="$storage_dir/sources/llama.cpp-927dada6c"
driver_dir="$storage_dir/sources/llama-route-trace-driver"
build_dir="$storage_dir/build/llama-route-trace"
mkdir -p "$driver_dir" "$build_dir" "$storage_dir/tmp"
export TMPDIR="$storage_dir/tmp"
if [[ ! -f "$source_dir/CMakeLists.txt" ]]; then
    source_archive="$storage_dir/sources/llama.cpp-$source_commit.tar.gz"
    curl -L --fail --retry 3 --output "$source_archive" \
        "https://github.com/ggml-org/llama.cpp/archive/$source_commit.tar.gz"
    mkdir -p "$source_dir"
    tar -xzf "$source_archive" --strip-components=1 -C "$source_dir"
    printf '%s\n' "$source_commit" > "$source_dir/EXPERIMENT_SOURCE_COMMIT.txt"
fi
[[ "$(cat "$source_dir/EXPERIMENT_SOURCE_COMMIT.txt")" == "$source_commit" ]]
cat > "$driver_dir/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.16)
project(llama_route_trace LANGUAGES C CXX)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE ON CACHE BOOL "" FORCE)
add_subdirectory("${LLAMA_SOURCE}" llama)
add_executable(llama-route-trace "${TRACE_SOURCE}")
target_link_libraries(llama-route-trace PRIVATE llama)
target_compile_features(llama-route-trace PRIVATE cxx_std_17)
add_executable(llama-local-quantize "${QUANTIZE_SOURCE}")
target_link_libraries(llama-local-quantize PRIVATE llama)
target_compile_features(llama-local-quantize PRIVATE cxx_std_17)
CMAKE
cmake -S "$driver_dir" -B "$build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DLLAMA_SOURCE="$source_dir" \
    -DLLAMA_BUILD_COMMIT="$source_commit" \
    -DTRACE_SOURCE="$experiment_dir/scripts/llama_route_trace.cpp" \
    -DQUANTIZE_SOURCE="$experiment_dir/scripts/llama_local_quantize.cpp"
cmake --build "$build_dir" --target llama-route-trace llama-local-quantize -j "${BUILD_JOBS:-8}"
printf 'Built %s\n' "$build_dir/llama-route-trace"
