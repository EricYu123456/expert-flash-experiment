// Local experimental harness; llama.cpp itself is kept unmodified.
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct trace_state {
    FILE * file = nullptr;
    bool enabled = false;
    bool failed = false;
    int token_index = -1;
    int position = -1;
    llama_token token_id = -1;
    int records = 0;
};

static bool trace_callback(ggml_tensor * tensor, bool ask, void * opaque) {
    auto & state = *static_cast<trace_state *>(opaque);
    int layer = -1;
    char trailing = '\0';
    const bool selected = state.enabled &&
        std::sscanf(tensor->name, "ffn_moe_topk-%d%c", &layer, &trailing) == 1;
    if (ask) {
        return selected;
    }
    if (!selected) {
        return true;
    }
    // Top-k is a view of the argsort output. Its row stride can include
    // all experts, so never assume that its data are densely packed.
    if (tensor->type != GGML_TYPE_I32 || tensor->ne[1] != 1 ||
            tensor->ne[2] != 1 || tensor->ne[3] != 1) {
        std::fprintf(stderr, "Unexpected routing tensor %s\n", tensor->name);
        state.failed = true;
        return false;
    }
    std::vector<unsigned char> data(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, data.data(), 0, data.size());
    std::fprintf(state.file,
        "{\"event\":\"routing\",\"phase\":\"decode\",\"token_index\":%d,"
        "\"position\":%d,\"token_id\":%d,\"layer\":%d,"
        "\"tensor_name\":\"%s\",\"expert_ids\":[",
        state.token_index, state.position, state.token_id, layer, tensor->name);
    for (int64_t rank = 0; rank < tensor->ne[0]; ++rank) {
        int32_t expert;
        std::memcpy(&expert, data.data() + rank * tensor->nb[0], sizeof(expert));
        std::fprintf(state.file, "%s%d", rank ? "," : "", expert);
    }
    std::fprintf(state.file, "]}\n");
    state.records++;
    return true;
}

static std::string token_piece(const llama_vocab * vocab, llama_token token) {
    std::vector<char> data(128);
    int length = llama_token_to_piece(vocab, token, data.data(), data.size(), 0, true);
    if (length < 0) {
        data.resize(-length);
        length = llama_token_to_piece(vocab, token, data.data(), data.size(), 0, true);
    }
    if (length < 0) throw std::runtime_error("token_to_piece failed");
    return std::string(data.data(), length);
}

int main(int argc, char ** argv) {
    if (argc != 8) {
        std::fprintf(stderr,
            "Usage: %s MODEL PROMPT_FILE TRACE_JSONL|- OUTPUT_TEXT STEPS THREADS SUMMARY_JSON\n",
            argv[0]);
        return 2;
    }
    try {
        const int steps = std::stoi(argv[5]);
        const int threads = std::stoi(argv[6]);
        if (steps < 1 || threads < 1 || threads > 128) throw std::runtime_error("invalid steps/threads");
        std::ifstream prompt_file(argv[2], std::ios::binary);
        if (!prompt_file) throw std::runtime_error("cannot open prompt");
        std::string prompt((std::istreambuf_iterator<char>(prompt_file)), {});
        std::ofstream output(argv[4], std::ios::binary);
        std::ofstream summary(argv[7]);
        if (!output || !summary) throw std::runtime_error("cannot open output");

        trace_state trace;
        const bool tracing = std::strcmp(argv[3], "-") != 0;
        if (tracing) {
            trace.file = std::fopen(argv[3], "w");
            if (!trace.file) throw std::runtime_error("cannot open trace");
        }

        llama_backend_init();
        ggml_backend_load_all();
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        llama_model * model = llama_model_load_from_file(argv[1], model_params);
        if (!model) throw std::runtime_error("model load failed");
        const auto * vocab = llama_model_get_vocab(model);
        int count = -llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0, true, true);
        if (count <= 0) throw std::runtime_error("empty prompt");
        std::vector<llama_token> tokens(count);
        if (llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), count, true, true) != count) {
            throw std::runtime_error("tokenization failed");
        }

        auto ctx_params = llama_context_default_params();
        ctx_params.n_ctx = std::max(512, count + steps + 8);
        ctx_params.n_batch = count;
        ctx_params.n_ubatch = count;
        ctx_params.n_threads = threads;
        ctx_params.n_threads_batch = threads;
        ctx_params.no_perf = false;
        if (tracing) {
            ctx_params.cb_eval = trace_callback;
            ctx_params.cb_eval_user_data = &trace;
        }
        llama_context * ctx = llama_init_from_model(model, ctx_params);
        if (!ctx) throw std::runtime_error("context init failed");
        auto batch = llama_batch_get_one(tokens.data(), tokens.size());
        if (llama_decode(ctx, batch)) throw std::runtime_error("prefill failed");
        llama_synchronize(ctx);
        auto * sampler = llama_sampler_init_greedy();
        const auto start = std::chrono::steady_clock::now();
        int evaluated = 0;
        for (int i = 0; i < steps; ++i) {
            llama_token token = llama_sampler_sample(sampler, ctx, -1);
            // Fixed token count: record EOS as an input too if it occurs.
            output << token_piece(vocab, token);
            trace.enabled = tracing;
            trace.token_index = i;
            trace.position = count + i;
            trace.token_id = token;
            batch = llama_batch_get_one(&token, 1);
            if (llama_decode(ctx, batch)) throw std::runtime_error("decode failed");
            llama_synchronize(ctx);
            if (trace.failed) throw std::runtime_error("routing callback failed");
            evaluated++;
        }
        const auto end = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(end - start).count();
        summary << "{\n  \"prompt_tokens\": " << count
                << ",\n  \"decoded_tokens\": " << evaluated
                << ",\n  \"layers\": " << llama_model_n_layer(model)
                << ",\n  \"threads\": " << threads
                << ",\n  \"trace_enabled\": " << (tracing ? "true" : "false")
                << ",\n  \"routing_records\": " << trace.records
                << ",\n  \"prompt_token_ids\": [";
        for (size_t i = 0; i < tokens.size(); ++i) {
            summary << (i ? "," : "") << tokens[i];
        }
        summary << "]"
                << ",\n  \"decode_seconds\": " << seconds
                << ",\n  \"decode_tokens_per_second\": " << evaluated / seconds << "\n}\n";
        llama_perf_context_print(ctx);
        llama_sampler_free(sampler);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        if (trace.file) std::fclose(trace.file);
        return 0;
    } catch (const std::exception & exception) {
        std::fprintf(stderr, "llama-route-trace: %s\n", exception.what());
        return 1;
    }
}
