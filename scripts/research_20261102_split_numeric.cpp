// Independent sequences share a CPU decode batch; upstream llama.cpp is unchanged.
#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <map>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
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


// Standalone numerical audit derived from the existing corpus collector.
// Upstream llama source, libraries, old runner and old binary remain unchanged.
// All weight block scales and quantized activation blocks are retained.
static std::string audit_model, audit_variant;
static bool numeric_enabled=false, overwrite_enabled=false;
static bool overwrite_full=false;
static gguf_context * original_gguf=nullptr;
static const char * original_weight_mapping=nullptr;
static size_t original_weight_size=0;
struct ErrorStats {
    uint64_t count=0, bitwise_different=0, nonfinite=0, max_ulp=0;
    double max_abs=0, max_relative=0, squared_error=0, squared_reference=0;
    void add(float value, float reference) {
        ++count;
        uint32_t a,b; std::memcpy(&a,&value,4); std::memcpy(&b,&reference,4);
        bitwise_different += a!=b;
        if (!std::isfinite(value) || !std::isfinite(reference)) {++nonfinite;return;}
        double error=std::abs(double(value)-double(reference));
        max_abs=std::max(max_abs,error);
        max_relative=std::max(max_relative,error/std::max(std::abs(double(reference)),1e-30));
        squared_error+=error*error; squared_reference+=double(reference)*reference;
        auto order=[](uint32_t x){return x&0x80000000U?~x:x|0x80000000U;};
        uint64_t x=order(a),y=order(b);max_ulp=std::max(max_ulp,x>y?x-y:y-x);
    }
    void json(std::ostream &o) const {
        o << "{\"count\":"<<count<<",\"bitwise_different\":"<<bitwise_different
          <<",\"nonfinite\":"<<nonfinite<<",\"max_abs\":"<<max_abs
          <<",\"max_relative_unclipped\":"<<max_relative<<",\"max_ulp\":"<<max_ulp
          <<",\"relative_l2\":"<<std::sqrt(squared_error/std::max(squared_reference,1e-300))<<"}";
    }
};
struct NumericStats {
    ErrorStats full_vs_dst, split2_vs_full, split2_vs_dst, split3_vs_full;
    uint64_t calls=0, dot_rows=0;
    std::string weight_name, weight_type, activation_dot_type;
    int64_t columns=0, output_rows=0, selected=0, tokens=0;
};
static std::map<std::string, NumericStats> numeric_stats;
static bool target_numeric(const ggml_tensor *t,int &layer,std::string &projection) {
    if(t->op!=GGML_OP_MUL_MAT_ID || !t->src[0])return false;
    char p[64];int end=0;
    if(std::sscanf(t->src[0]->name,"blk.%d.ffn_%63[^.]%n",&layer,p,&end)!=2)return false;
    projection=p;
    return audit_model=="granite4tiny" ? layer<3&&(projection=="gate_exps"||projection=="up_exps")
          : audit_model=="qwen3"&&layer<2&&projection=="down_exps";
}
static void audit_tensor(ggml_tensor *dst,int layer,const std::string &projection) {
    const ggml_tensor *w=dst->src[0],*x=dst->src[1],*ids=dst->src[2];
    if(!w->data || !x->data || !ids->data || !dst->data || x->type!=GGML_TYPE_F32 ||
       ids->type!=GGML_TYPE_I32 || dst->type!=GGML_TYPE_F32 || w->ne[3]!=1 || x->ne[3]!=1)
        throw std::runtime_error("Unexpected numeric tensor storage/type");
    const int C=w->ne[0],R=w->ne[1],first=C==1536?1024:C==768?512:0;
    if(!first)throw std::runtime_error("Unexpected split columns");
    const auto *wt=ggml_get_type_traits_cpu(w->type);
    const auto dot_type=wt->vec_dot_type;
    const auto *at=ggml_get_type_traits_cpu(dot_type);
    if(!wt->vec_dot||!at->from_float || C%ggml_blck_size(w->type) ||
       first%ggml_blck_size(w->type)||first%ggml_blck_size(dot_type))
        throw std::runtime_error("Missing CPU quantized dot/block alignment");
    const std::string key=std::to_string(layer)+"/"+projection;
    auto &s=numeric_stats[key];++s.calls;s.weight_name=w->name;s.weight_type=ggml_type_name(w->type);
    s.activation_dot_type=ggml_type_name(dot_type);s.columns=C;s.output_rows=R;s.selected=ids->ne[0];s.tokens=ids->ne[1];
    std::vector<uint8_t> quant(ggml_row_size(dot_type,C));
    const int64_t tensor_id=gguf_find_tensor(original_gguf,w->name);
    if(tensor_id<0 || gguf_get_tensor_type(original_gguf,tensor_id)!=w->type)
        throw std::runtime_error("Original GGUF tensor mismatch");
    const size_t tensor_offset=gguf_get_data_offset(original_gguf)+gguf_get_tensor_offset(original_gguf,tensor_id);
    const size_t original_row_bytes=ggml_row_size(w->type,C);
    if(tensor_offset+original_row_bytes*R*w->ne[2]>original_weight_size)
        throw std::runtime_error("Original GGUF tensor outside file");
    const char *original_weight=original_weight_mapping+tensor_offset;
    const size_t woff=ggml_row_size(w->type,first),xoff=ggml_row_size(dot_type,first);
    for(int64_t token=0;token<ids->ne[1];++token)for(int64_t selected=0;selected<ids->ne[0];++selected){
        int32_t expert;std::memcpy(&expert,(char*)ids->data+token*ids->nb[1]+selected*ids->nb[0],4);
        if(expert<0||expert>=w->ne[2])throw std::runtime_error("Invalid expert ID");
        const float *input=(float*)((char*)x->data+token*x->nb[2]+(selected%x->ne[1])*x->nb[1]);
        // Quantize the original full input first; split original blocks without requantization.
        at->from_float(input,quant.data(),C);
        for(int row=0;row<R;++row){
            // CPU extra buffer types may repack src0->data. Use original immutable
            // GGUF blocks, not a false reinterpretation of the repacked buffer.
            const char *weight=original_weight+(expert*R+row)*original_row_bytes;
            float *original=(float*)((char*)dst->data+token*dst->nb[2]+selected*dst->nb[1]+row*dst->nb[0]);
            float full,left,right;
            wt->vec_dot(C,&full,0,weight,0,quant.data(),0,1);
            wt->vec_dot(first,&left,0,weight,0,quant.data(),0,1);
            wt->vec_dot(C-first,&right,0,weight+woff,0,quant.data()+xoff,0,1);
            const float split=left+right;
            s.full_vs_dst.add(full,*original);s.split2_vs_full.add(split,full);s.split2_vs_dst.add(split,*original);
            if(C==1536){
                float total=0;
                for(int part=0;part<3;++part){float partial;
                    wt->vec_dot(512,&partial,0,weight+ggml_row_size(w->type,512)*part,0,
                                quant.data()+ggml_row_size(dot_type,512)*part,0,1);
                    total+=partial;
                }
                s.split3_vs_full.add(total,full);
            }
            ++s.dot_rows;
            if(overwrite_enabled)*original=overwrite_full?full:split;
        }
    }
}
static void write_numeric(const std::string &path){
    std::ofstream out(path);out<<std::setprecision(17)<<"{\"model\":\""<<audit_model
        <<"\",\"full_model_values_modified\":"<<(overwrite_enabled?"true":"false")
        <<",\"full_dot_overwrite_control\":"<<(overwrite_full?"true":"false")
        <<",\"native_aif_equivalence_claimed\":false,\"weights_read_from\":\"original_readonly_GGUF_not_repacked_CPU_buffer\",\"phases\":[";
    bool comma=false;for(const auto &entry:numeric_stats){const auto &s=entry.second;if(comma)out<<',';comma=true;
        out<<"{\"phase\":\""<<entry.first<<"\",\"weight_name\":\""<<s.weight_name
           <<"\",\"weight_type\":\""<<s.weight_type<<"\",\"activation_dot_type\":\""<<s.activation_dot_type
           <<"\",\"columns\":"<<s.columns<<",\"output_rows\":"<<s.output_rows
           <<",\"selected_experts\":"<<s.selected<<",\"tokens_per_batch\":"<<s.tokens
           <<",\"callbacks\":"<<s.calls<<",\"dot_rows\":"<<s.dot_rows<<",\"full_vs_dst\":";s.full_vs_dst.json(out);
        out<<",\"split2_vs_full\":";s.split2_vs_full.json(out);out<<",\"split2_vs_dst\":";s.split2_vs_dst.json(out);
        out<<",\"split3_vs_full\":";s.split3_vs_full.json(out);out<<'}';
    }out<<"]}\n";if(!out)throw std::runtime_error("Numeric report write failed");
}
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
    int numeric_layer=-1;std::string projection;
    if(state.enabled && numeric_enabled && target_numeric(tensor,numeric_layer,projection)) {
        if(ask)return true;
        try {audit_tensor(tensor,numeric_layer,projection);} catch(const std::exception &e) {
            std::fprintf(stderr,"Numeric callback: %s\n",e.what());state.failed=true;return false;
        }
        return true;
    }
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
            job.trace = std::fopen((job.stem + "." + audit_variant + ".routing.jsonl").c_str(), "w");
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
        const std::string label = job.stem + "." + audit_variant;
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


int main(int argc,char **argv){
    if(argc!=9){std::fprintf(stderr,"Usage: %s MODEL JOBS_TSV STEPS THREADS BATCH model_alias REPORT_JSON control-capture|overwrite|overwrite-full\n",argv[0]);return 2;}
    try {
        int steps=std::stoi(argv[3]),threads=std::stoi(argv[4]),batch=std::stoi(argv[5]);
        audit_model=argv[6];const std::string mode=argv[8];overwrite_full=mode=="overwrite-full";
        overwrite_enabled=mode=="overwrite"||overwrite_full;
        if(steps<1||threads<1||batch<1||batch>2||(audit_model!="granite4tiny"&&audit_model!="qwen3")||
            (mode!="control-capture"&&mode!="overwrite"&&mode!="overwrite-full"))throw std::runtime_error("Invalid parameters");
        std::ifstream list(argv[2]);std::vector<Job> all;std::string line;
        while(std::getline(list,line)){auto tab=line.find('\t');if(tab==std::string::npos)throw std::runtime_error("Invalid TSV");
            Job j;j.prompt_path=line.substr(0,tab);j.stem=line.substr(tab+1);all.push_back(j);}
        if(all.empty())throw std::runtime_error("No prompts");
        gguf_init_params gp={true,nullptr};original_gguf=gguf_init_from_file(argv[1],gp);
        if(!original_gguf)throw std::runtime_error("GGUF metadata loading failed");
        const int fd=open(argv[1],O_RDONLY);struct stat info;
        if(fd<0||fstat(fd,&info))throw std::runtime_error("GGUF readonly open failed");
        original_weight_size=info.st_size;
        auto *mapping=mmap(nullptr,original_weight_size,PROT_READ,MAP_SHARED,fd,0);close(fd);
        if(mapping==MAP_FAILED)throw std::runtime_error("GGUF readonly mmap failed");
        original_weight_mapping=(const char*)mapping;
        llama_backend_init();ggml_backend_load_all();auto mp=llama_model_default_params();mp.n_gpu_layers=0;
        auto *model=llama_model_load_from_file(argv[1],mp);if(!model)throw std::runtime_error("Model loading failed");
        for(size_t i=0;i<all.size();i+=batch){
            for(int variant=0;variant<(overwrite_enabled?1:2);++variant){
                numeric_enabled=overwrite_enabled||variant==1;audit_variant=overwrite_full?"overwrite-full":overwrite_enabled?"overwrite":variant==0?"control":"capture";
                std::vector<Job> jobs(all.begin()+i,all.begin()+std::min(all.size(),i+batch));
                run(model,jobs,steps,threads,true);
            }
        }
        write_numeric(argv[7]);llama_model_free(model);llama_backend_free();
        munmap((void*)original_weight_mapping,original_weight_size);gguf_free(original_gguf);return 0;
    }catch(const std::exception &e){std::fprintf(stderr,"split-numeric: %s\n",e.what());return 1;}
}
