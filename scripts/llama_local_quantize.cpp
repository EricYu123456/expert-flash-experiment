// Minimal local quantization driver using the public unmodified llama.cpp API.
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>

int main(int argc, char ** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "Usage: %s INPUT OUTPUT Q4_K_M|Q8_0 THREADS\n", argv[0]);
        return 2;
    }
    auto params = llama_model_quantize_default_params();
    if (!std::strcmp(argv[3], "Q4_K_M")) params.ftype = LLAMA_FTYPE_MOSTLY_Q4_K_M;
    else if (!std::strcmp(argv[3], "Q8_0")) params.ftype = LLAMA_FTYPE_MOSTLY_Q8_0;
    else return 2;
    params.nthread = std::atoi(argv[4]);
    if (params.nthread < 1 || params.nthread > 128) return 2;
    llama_backend_init();
    const int status = llama_model_quantize(argv[1], argv[2], &params);
    llama_backend_free();
    return status;
}
