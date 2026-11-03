// Independent sequences share a CPU decode batch; upstream llama.cpp is unchanged.
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
struct Job {
    std::string prompt_path, stem;
    std::vector<llama_token> prompt, generated;
    FILE * trace = nullptr;
    int records = 0;
    std::string text, prefix;
};
struct Trace {
    std::vector<Job> * jobs = nullptr;
    bool enabled = false, failed = false;
    int step = 0;
};
struct PrefillCache {
    std::vector<std::vector<uint8_t>> state;
    std::vector<llama_token> next;
};

static bool callback(ggml_tensor * tensor, bool ask, void * opaque) {
    auto & state = *static_cast<Trace *>(opaque);
    int layer = -1; char tail = '\0';
    bool selected = state.enabled && std::sscanf(tensor->name, "ffn_moe_topk-%d%c", &layer, &tail) == 1;
    if (ask) return selected;
    if (!selected) return true;
    auto & jobs = *state.jobs;
    if (tensor->type != GGML_TYPE_I32 || tensor->ne[1] != (int64_t) jobs.size() ||
        tensor->ne[2] != 1 || tensor->ne[3] != 1) {
        std::fprintf(stderr, "Unexpected top-k batch shape: %s [%lld,%lld,%lld,%lld]\n", tensor->name,
                     (long long)tensor->ne[0], (long long)tensor->ne[1],
                     (long long)tensor->ne[2], (long long)tensor->ne[3]);
        state.failed = true; return false;
    }
    std::vector<unsigned char> data(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, data.data(), 0, data.size());
    for (size_t j = 0; j < jobs.size(); ++j) {
        auto & job = jobs[j];
        std::fprintf(job.trace, "{\"event\":\"routing\",\"phase\":\"decode\",\"token_index\":%d,"
                     "\"position\":%d,\"token_id\":%d,\"layer\":%d,\"expert_ids\":[",
                     state.step, (int)job.prompt.size() + state.step, job.generated.back(), layer);
        for (int64_t k = 0; k < tensor->ne[0]; ++k) {
            int32_t expert;
            std::memcpy(&expert, data.data() + j * tensor->nb[1] + k * tensor->nb[0], sizeof(expert));
            std::fprintf(job.trace, "%s%d", k ? "," : "", expert);
        }
        std::fprintf(job.trace, "]}\n"); job.records++;
    }
    return true;
}

static std::string piece(const llama_vocab * vocab, llama_token token) {
    std::vector<char> buffer(128);
    int n = llama_token_to_piece(vocab, token, buffer.data(), buffer.size(), 0, true);
    if (n < 0) { buffer.resize(-n); n = llama_token_to_piece(vocab, token, buffer.data(), buffer.size(), 0, true); }
    if (n < 0) throw std::runtime_error("token_to_piece failed");
    return std::string(buffer.data(), n);
}
static void add(llama_batch & b, llama_token token, int pos, int seq, bool logits) {
    int i = b.n_tokens++;
    b.token[i] = token; b.pos[i] = pos; b.n_seq_id[i] = 1; b.seq_id[i][0] = seq; b.logits[i] = logits;
}
static void save_text(const std::string & path, const std::string & content) {
    std::ofstream f(path, std::ios::binary); if (!f) throw std::runtime_error("Cannot write " + path);
    f << content; if (!f) throw std::runtime_error("Write failed " + path);
}
static void ids(std::ostream & out, const std::vector<llama_token> & values) {
    out << '['; for (size_t i = 0; i < values.size(); ++i) out << (i ? "," : "") << values[i]; out << ']';
}

static void run(llama_model * model, std::vector<Job> & jobs, int steps, int threads, bool traced,
                PrefillCache * save = nullptr, const PrefillCache * restore = nullptr) {
    const auto * vocab = llama_model_get_vocab(model);
    int longest = 0;
    for (auto & job : jobs) {
        std::ifstream f(job.prompt_path, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot read prompt");
        std::string prompt((std::istreambuf_iterator<char>(f)), {});
        int n = -llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0, true, true);
        if (n <= 0) throw std::runtime_error("Empty prompt");
        job.prompt.resize(n);
        if (llama_tokenize(vocab, prompt.data(), prompt.size(), job.prompt.data(), n, true, true) != n)
            throw std::runtime_error("Tokenization failed");
        longest = std::max(longest, n);
        if (traced) {
            job.trace = std::fopen((job.stem + ".routing.jsonl").c_str(), "w");
            if (!job.trace) throw std::runtime_error("Cannot open trace");
        }
    }
    Trace trace; trace.jobs = &jobs;
    auto p = llama_context_default_params();
    p.n_seq_max = jobs.size();
    p.n_ctx = ((longest + steps + 255) / 256 * 256) * jobs.size();
    p.n_batch = p.n_ubatch = std::max(512, (int)jobs.size());
    p.n_threads = p.n_threads_batch = threads;
    p.kv_unified = false;
    p.no_perf = false;
    if (traced) { p.cb_eval = callback; p.cb_eval_user_data = &trace; }
    llama_context * ctx = llama_init_from_model(model, p);
    if (!ctx) throw std::runtime_error("Context init failed");
    auto batch = llama_batch_init(p.n_batch, 0, 1);
    auto * sampler = llama_sampler_init_greedy();
    std::vector<llama_token> next(jobs.size());
    const auto prefill_start = Clock::now();
    if (restore) {
        // Restore individual sequence states: the pinned whole-cache reader
        // clears the cache once per stream when kv_unified=false.
        for (size_t j=0;j<jobs.size();++j) {
            const auto & data=restore->state[j];
            if (llama_state_seq_set_data(ctx,data.data(),data.size(),j) != data.size())
                throw std::runtime_error("Prefill sequence state restore failed");
        }
        next=restore->next;
    } else for (size_t j = 0; j < jobs.size(); ++j) {
        const auto & tokens = jobs[j].prompt;
        for (size_t offset = 0; offset < tokens.size(); offset += p.n_batch) {
            batch.n_tokens = 0;
            int n = std::min((size_t)p.n_batch, tokens.size() - offset);
            for (int k = 0; k < n; ++k) add(batch, tokens[offset+k], offset+k, j, offset+k+1 == tokens.size());
            if (llama_decode(ctx, batch)) throw std::runtime_error("Prefill failed");
        }
        next[j] = llama_sampler_sample(sampler, ctx, -1);
    }
    llama_synchronize(ctx);
    if (save) {
        save->state.resize(jobs.size());
        for (size_t j=0;j<jobs.size();++j) {
            auto & data=save->state[j];
            data.resize(llama_state_seq_get_size(ctx,j));
            if (llama_state_seq_get_data(ctx,data.data(),data.size(),j) != data.size())
                throw std::runtime_error("Prefill sequence state save failed");
        }
        save->next=next;
    }
    const double prefill = std::chrono::duration<double>(Clock::now() - prefill_start).count();
    const auto start = Clock::now();
    for (int step = 0; step < steps; ++step) {
        batch.n_tokens = 0; trace.enabled = traced; trace.step = step;
        for (size_t j = 0; j < jobs.size(); ++j) {
            auto & job = jobs[j];
            job.generated.push_back(next[j]); job.text += piece(vocab, next[j]);
            if (step < 32) job.prefix = job.text;
            add(batch, next[j], job.prompt.size() + step, j, true);
        }
        if (llama_decode(ctx, batch)) throw std::runtime_error("Decode failed");
        llama_synchronize(ctx);
        if (trace.failed) throw std::runtime_error("Routing callback failed");
        for (size_t j = 0; j < jobs.size(); ++j) next[j] = llama_sampler_sample(sampler, ctx, j);
    }
    const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
    for (auto & job : jobs) {
        if (job.trace) { if (std::fclose(job.trace)) throw std::runtime_error("Trace write failed"); job.trace = nullptr; }
        const std::string label = job.stem + (traced ? ".traced" : ".baseline");
        save_text(label + ".output.txt", job.text); save_text(label + ".prefix32.txt", job.prefix);
        std::ofstream summary(label + ".summary.json");
        summary << "{\"prompt_tokens\":" << job.prompt.size() << ",\"decoded_tokens\":" << steps
                << ",\"threads\":" << threads << ",\"sequence_batch\":" << jobs.size()
                << ",\"trace_enabled\":" << (traced ? "true" : "false") << ",\"routing_records\":" << job.records
                << ",\"prefill_state_reused\":" << (restore ? "true" : "false")
                << ",\"batch_decode_seconds\":" << seconds << ",\"batch_prefill_seconds\":" << prefill
                << ",\"prompt_token_ids\":"; ids(summary, job.prompt);
        summary << ",\"generated_token_ids\":"; ids(summary, job.generated); summary << "}\n";
        if (!summary) throw std::runtime_error("Summary write failed");
        std::cout << "Completed " << label << " prompt_tokens=" << job.prompt.size()
                  << " records=" << job.records << " batch_decode_seconds=" << seconds << std::endl;
    }
    llama_perf_context_print(ctx); llama_sampler_free(sampler); llama_batch_free(batch); llama_free(ctx);
}

int main(int argc, char ** argv) {
    if (argc != 7) { std::fprintf(stderr, "Usage: %s MODEL JOBS_TSV STEPS THREADS BATCH_SIZE trace|baseline|both|both-cache\n", argv[0]); return 2; }
    try {
        int steps=std::stoi(argv[3]), threads=std::stoi(argv[4]), batch_size=std::stoi(argv[5]);
        bool traced = std::string(argv[6]) == "trace";
        bool cache = std::string(argv[6]) == "both-cache";
        bool both = std::string(argv[6]) == "both" || cache;
        if (steps<1 || threads<1 || batch_size<1 || batch_size>32 ||
            (!traced && !both && std::string(argv[6]) != "baseline")) throw std::runtime_error("Invalid arguments");
        std::ifstream list(argv[2]); if (!list) throw std::runtime_error("Cannot read jobs");
        std::vector<Job> all; std::string line;
        while (std::getline(list,line)) {
            auto tab=line.find('\t'); if(tab==std::string::npos) throw std::runtime_error("Invalid TSV");
            Job j; j.prompt_path=line.substr(0,tab); j.stem=line.substr(tab+1); all.push_back(j);
        }
        if (all.empty()) throw std::runtime_error("No jobs");
        llama_backend_init(); ggml_backend_load_all();
        auto mp=llama_model_default_params(); mp.n_gpu_layers=0;
        auto * model=llama_model_load_from_file(argv[1],mp); if(!model) throw std::runtime_error("Model load failed");
        for(size_t i=0;i<all.size();i+=batch_size) {
            std::vector<Job> group(all.begin()+i,all.begin()+std::min(all.size(),i+batch_size));
            PrefillCache prefill;
            run(model,group,steps,threads,traced || both,cache ? &prefill : nullptr);
            if (both) {
                std::vector<Job> baseline(all.begin()+i,all.begin()+std::min(all.size(),i+batch_size));
                run(model,baseline,steps,threads,false,nullptr,cache ? &prefill : nullptr);
            }
            std::cout << "BATCH_COMPLETE " << i << " " << group.size() << std::endl;
        }
        llama_model_free(model); llama_backend_free(); return 0;
    } catch(const std::exception & e) { std::fprintf(stderr,"llama-corpus-trace: %s\n",e.what()); return 1; }
}
