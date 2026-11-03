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



// Functional raw-W8A8 operator reference. This intentionally changes GGUF
// quantization and is not a new quantization method or native AiF format claim.
static std::string audit_model,audit_variant,report_directory;
static bool numeric_enabled=true;
static gguf_context *original_gguf=nullptr;
static const char *original_weight_mapping=nullptr;
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

static uint64_t hash_bytes(uint64_t h,const void *data,size_t n){
    const auto *p=(const unsigned char*)data;
    for(size_t i=0;i<n;++i){h^=p[i];h*=1099511628211ULL;}return h;
}
static constexpr uint64_t HASH_INIT=14695981039346656037ULL;
static float quantize_vector(const float *input,int C,int8_t *output){
    float amax=0;
    for(int k=0;k<C;++k){if(!std::isfinite(input[k]))throw std::runtime_error("Nonfinite quantization input");
        amax=std::max(amax,std::abs(input[k]));}
    const float scale=amax==0?1.0f:amax/127.0f;
    if(!(scale>0)||!std::isfinite(scale))throw std::runtime_error("Invalid/underflowed quantization scale");
    for(int k=0;k<C;++k){
        // FP32 division, then nearest with halfway ties away from zero.
        const float normalized=input[k]/scale;
        const int q=std::clamp((int)std::round(normalized),-127,127);
        output[k]=(int8_t)q;
    }return scale;
}
struct ExpertCache {
    std::string weight_name,source_type,scale_file,created_variant;
    int expert=0,C=0,R=0;size_t source_offset=0;
    std::vector<int8_t> weights;std::vector<float> row_scales;
    uint64_t weights_fnv64=HASH_INIT,scales_fnv64=HASH_INIT;
};
static std::map<std::string,ExpertCache> weight_cache;
static const ExpertCache &get_expert(const ggml_tensor *w,int expert){
    const std::string key=std::string(w->name)+"/"+std::to_string(expert);
    auto found=weight_cache.find(key);if(found!=weight_cache.end())return found->second;
    const int64_t tid=gguf_find_tensor(original_gguf,w->name);
    if(tid<0||gguf_get_tensor_type(original_gguf,tid)!=w->type)throw std::runtime_error("Original GGUF tensor mismatch");
    const auto *traits=ggml_get_type_traits(w->type);
    if(!traits->to_float)throw std::runtime_error("Missing original GGUF dequantizer");
    ExpertCache e;e.weight_name=w->name;e.source_type=ggml_type_name(w->type);e.expert=expert;
    e.C=w->ne[0];e.R=w->ne[1];e.created_variant=audit_variant;
    const size_t row_bytes=ggml_row_size(w->type,e.C);
    e.source_offset=gguf_get_data_offset(original_gguf)+gguf_get_tensor_offset(original_gguf,tid)+expert*e.R*row_bytes;
    if(e.source_offset+e.R*row_bytes>original_weight_size)throw std::runtime_error("Original GGUF range invalid");
    e.weights.resize((size_t)e.R*e.C);e.row_scales.resize(e.R);std::vector<float> row(e.C);
    for(int r=0;r<e.R;++r){
        traits->to_float(original_weight_mapping+e.source_offset+r*row_bytes,row.data(),e.C);
        e.row_scales[r]=quantize_vector(row.data(),e.C,e.weights.data()+(size_t)r*e.C);
    }
    e.weights_fnv64=hash_bytes(HASH_INIT,e.weights.data(),e.weights.size());
    e.scales_fnv64=hash_bytes(HASH_INIT,e.row_scales.data(),e.row_scales.size()*4);
    e.scale_file=report_directory+"/scales/"+e.weight_name+".expert"+std::to_string(expert)+".f32";
    std::ofstream scales(e.scale_file,std::ios::binary);
    scales.write((const char*)e.row_scales.data(),e.row_scales.size()*4);
    if(!scales)throw std::runtime_error("Scale cache write failed");
    return weight_cache.emplace(key,std::move(e)).first->second;
}
struct RawStats {
    ErrorStats raw_vs_original,split2_vs_full,split3_vs_full;
    uint64_t callbacks=0,dot_rows=0,integer_mismatches=0,scaled_bit_mismatches=0;
    uint64_t input_vectors=0,input_fnv64=HASH_INIT,output_fnv64=HASH_INIT;
    int64_t max_abs_full_integer=0,max_abs_partial2=0,max_abs_partial3=0;
    int C=0,R=0,K=0,tokens=0;std::string weight_type;
};
static std::map<std::string,RawStats> raw_stats;
static bool target_numeric(const ggml_tensor *t,int &layer,std::string &projection) {
    if(t->op!=GGML_OP_MUL_MAT_ID || !t->src[0])return false;
    char p[64];int end=0;
    if(std::sscanf(t->src[0]->name,"blk.%d.ffn_%63[^.]%n",&layer,p,&end)!=2)return false;
    projection=p;
    return audit_model=="granite4tiny" ? layer<3&&(projection=="gate_exps"||projection=="up_exps")
          : audit_model=="qwen3"&&layer<2&&projection=="down_exps";
}

static float scaled_output(int64_t integer_sum,float row_scale,float input_scale){
    // Same operations and rounding in both modes; compiler uses no fast math/FMA.
    const float scale_product=row_scale*input_scale;
    const float integer_as_float=(float)integer_sum;
    return integer_as_float*scale_product;
}
static void audit_tensor(ggml_tensor *dst,int layer,const std::string &projection){
    const auto *w=dst->src[0],*x=dst->src[1],*ids=dst->src[2];
    if(!x->data||!ids->data||!dst->data||x->type!=GGML_TYPE_F32||ids->type!=GGML_TYPE_I32||
       dst->type!=GGML_TYPE_F32||x->ne[3]!=1||w->ne[3]!=1)throw std::runtime_error("Unsupported input tensor");
    const int C=w->ne[0],R=w->ne[1],cut=C==1536?1024:C==768?512:0;
    if(!cut)throw std::runtime_error("Unexpected operator columns");
    auto &s=raw_stats[audit_variant+"/"+std::to_string(layer)+"/"+projection];++s.callbacks;
    s.C=C;s.R=R;s.K=ids->ne[0];s.tokens=ids->ne[1];s.weight_type=ggml_type_name(w->type);
    const size_t vectors=x->ne[1]*x->ne[2];
    std::vector<int8_t> input_quant(vectors*C);std::vector<float> input_scales(vectors);
    for(int64_t token=0;token<x->ne[2];++token)for(int64_t vector=0;vector<x->ne[1];++vector){
        const auto *original=(const float*)((const char*)x->data+token*x->nb[2]+vector*x->nb[1]);
        const size_t index=token*x->ne[1]+vector;
        input_scales[index]=quantize_vector(original,C,input_quant.data()+index*C);++s.input_vectors;
        s.input_fnv64=hash_bytes(s.input_fnv64,input_quant.data()+index*C,C);
        s.input_fnv64=hash_bytes(s.input_fnv64,&input_scales[index],4);
    }
    for(int64_t token=0;token<ids->ne[1];++token)for(int64_t selected=0;selected<ids->ne[0];++selected){
        int32_t expert;std::memcpy(&expert,(const char*)ids->data+token*ids->nb[1]+selected*ids->nb[0],4);
        if(expert<0||expert>=w->ne[2])throw std::runtime_error("Invalid expert ID");
        const auto &cache=get_expert(w,expert);
        const size_t input_index=token*x->ne[1]+selected%x->ne[1];
        const int8_t *input=input_quant.data()+input_index*C;
        const float input_scale=input_scales[input_index];
        for(int r=0;r<R;++r){
            const int8_t *weight=cache.weights.data()+(size_t)r*C;
            int64_t full=0;for(int k=0;k<C;++k)full+=(int32_t)weight[k]*(int32_t)input[k];
            int32_t left=0,right=0;
            for(int k=0;k<cut;++k)left+=(int32_t)weight[k]*(int32_t)input[k];
            for(int k=cut;k<C;++k)right+=(int32_t)weight[k]*(int32_t)input[k];
            const int64_t split2=(int64_t)left+(int64_t)right;
            int64_t split3=0;
            if(C==1536)for(int p=0;p<3;++p){int32_t partial=0;
                for(int k=p*512;k<(p+1)*512;++k)partial+=(int32_t)weight[k]*(int32_t)input[k];
                split3+=(int64_t)partial;s.max_abs_partial3=std::max(s.max_abs_partial3,std::abs((int64_t)partial));}
            s.integer_mismatches+=(full!=split2)||(C==1536&&full!=split3);
            s.max_abs_full_integer=std::max(s.max_abs_full_integer,std::abs(full));
            s.max_abs_partial2=std::max(s.max_abs_partial2,std::max(std::abs((int64_t)left),std::abs((int64_t)right)));
            const float result_full=scaled_output(full,cache.row_scales[r],input_scale);
            const float result_split=scaled_output(split2,cache.row_scales[r],input_scale);
            const float result_split3=scaled_output(split3,cache.row_scales[r],input_scale);
            s.split2_vs_full.add(result_split,result_full);
            if(C==1536)s.split3_vs_full.add(result_split3,result_full);
            uint32_t fa,fb;std::memcpy(&fa,&result_full,4);std::memcpy(&fb,&result_split,4);
            s.scaled_bit_mismatches+=fa!=fb;
            float *original=(float*)((char*)dst->data+token*dst->nb[2]+selected*dst->nb[1]+r*dst->nb[0]);
            s.raw_vs_original.add(result_full,*original);
            // Functional split uses 3x512 for G4 and 512+256 for Qwen.
            const float replacement=audit_variant=="raw-full"?result_full:C==1536?result_split3:result_split;
            *original=replacement;s.output_fnv64=hash_bytes(s.output_fnv64,&replacement,4);++s.dot_rows;
        }
    }
}
static void write_numeric(const std::string &path){
    std::ofstream out(path);out<<std::setprecision(17)<<"{\"model\":\""<<audit_model
        <<"\",\"full_model_quantization\":false,\"scope\":\"target_decode_GEMVs_only_original_GGUF_prefill_and_other_operators\","
        <<"\"native_aif_equivalence_claimed\":false,\"quantization\":{\"signed_range\":[-127,127],"
        <<"\"scale\":\"FP32 maxabs/127; allzero scale=1\",\"rounding\":\"FP32 division then nearest halfway-away-from-zero then clip\","
        <<"\"weights\":\"original GGUF to_float row then fixed per-row quantization shared cache\","
        <<"\"activation\":\"whole original vector quantized once per GEMV; partials slice the same bytes\","
        <<"\"output\":\"I64 integer sum; convert once to F32; multiply once by FP32(rowScale*inputScale)\"},\"phases\":[";
    bool comma=false;for(const auto &entry:raw_stats){const auto &s=entry.second;if(comma)out<<',';comma=true;
        out<<"{\"phase\":\""<<entry.first<<"\",\"source_weight_type\":\""<<s.weight_type
           <<"\",\"columns\":"<<s.C<<",\"output_rows\":"<<s.R<<",\"selected_experts\":"<<s.K
           <<",\"tokens_per_batch\":"<<s.tokens<<",\"callbacks\":"<<s.callbacks<<",\"dot_rows\":"<<s.dot_rows
           <<",\"integer_mismatches\":"<<s.integer_mismatches<<",\"scaled_bit_mismatches\":"<<s.scaled_bit_mismatches
           <<",\"max_abs_full_integer\":"<<s.max_abs_full_integer<<",\"max_abs_partial2\":"<<s.max_abs_partial2
           <<",\"max_abs_partial3\":"<<s.max_abs_partial3<<",\"quantized_input_vectors\":"<<s.input_vectors
           <<",\"input_fnv1a64\":\""<<s.input_fnv64<<"\",\"output_fnv1a64\":\""<<s.output_fnv64<<"\",\"raw_vs_original\":";
        s.raw_vs_original.json(out);out<<",\"split2_vs_full\":";s.split2_vs_full.json(out);out<<",\"split3_vs_full\":";
        s.split3_vs_full.json(out);out<<'}';
    }out<<"],\"weight_cache\":[";comma=false;
    for(const auto &entry:weight_cache){const auto &e=entry.second;if(comma)out<<',';comma=true;
        out<<"{\"key\":\""<<entry.first<<"\",\"source_type\":\""<<e.source_type<<"\",\"columns\":"<<e.C
           <<",\"rows\":"<<e.R<<",\"source_byte_offset\":"<<e.source_offset<<",\"raw_i8_bytes\":"<<e.weights.size()
           <<",\"scale_bytes\":"<<e.row_scales.size()*4<<",\"scales_f32le_path\":\""<<e.scale_file
           <<"\",\"created_variant\":\""<<e.created_variant<<"\",\"weights_fnv1a64\":\""<<e.weights_fnv64
           <<"\",\"scales_fnv1a64\":\""<<e.scales_fnv64<<"\"}";
    }out<<"]}\n";if(!out)throw std::runtime_error("Report write failed");
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
    if(argc!=8){std::fprintf(stderr,"Usage: %s MODEL JOBS_TSV STEPS THREADS BATCH model_alias REPORT_JSON\n",argv[0]);return 2;}
    try {
        int steps=std::stoi(argv[3]),threads=std::stoi(argv[4]),batch=std::stoi(argv[5]);audit_model=argv[6];
        std::string report=argv[7];report_directory=report.substr(0,report.find_last_of('/'));
        if(steps<1||threads<1||batch<1||batch>2||(audit_model!="granite4tiny"&&audit_model!="qwen3"))throw std::runtime_error("Invalid parameters");
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
        if(mapping==MAP_FAILED)throw std::runtime_error("GGUF readonly mmap failed");original_weight_mapping=(const char*)mapping;
        llama_backend_init();ggml_backend_load_all();auto mp=llama_model_default_params();mp.n_gpu_layers=0;
        auto *model=llama_model_load_from_file(argv[1],mp);if(!model)throw std::runtime_error("Model loading failed");
        for(size_t i=0;i<all.size();i+=batch){
            size_t full_cache_entries=0;
            for(const std::string &variant:{std::string("raw-full"),std::string("raw-split")}){
                audit_variant=variant;std::vector<Job> jobs(all.begin()+i,all.begin()+std::min(all.size(),i+batch));
                run(model,jobs,steps,threads,true);
                if(variant=="raw-full")full_cache_entries=weight_cache.size();
                else if(weight_cache.size()!=full_cache_entries)throw std::runtime_error("Split selected new expert cache entries");
            }
        }
        write_numeric(argv[7]);llama_model_free(model);llama_backend_free();munmap((void*)original_weight_mapping,original_weight_size);
        gguf_free(original_gguf);return 0;
    }catch(const std::exception &e){std::fprintf(stderr,"raw-w8a8-numeric: %s\n",e.what());return 1;}
}
