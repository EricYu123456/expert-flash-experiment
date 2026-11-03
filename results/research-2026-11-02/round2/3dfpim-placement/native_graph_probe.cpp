#include "3dfpim.h"
#include "model.h"
#include "tensors.h"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

// Existing compiler frontend only. Aligned vector-tile aliasing supplies the
// missing public concat(Vector,Vector) implementation without a new opcode.
static Vector alias_tiles(Model model, std::vector<Vector> parts) {
    auto impl = new VectorImpl(model.unwrap(), 128 * parts.size());
    for (unsigned j=0; j<parts.size(); ++j)
        impl->setTile(j, parts[j].unwrap()->getTile(0));
    return Vector(impl);
}
static Vector extract_tile(Model model, Vector x, unsigned j) {
    auto impl = new VectorImpl(model.unwrap(), 128);
    impl->setTile(0, x.unwrap()->getTile(j));
    return Vector(impl);
}
int main(int argc, char** argv) {
    if (argc != 6) return 2;
    const std::string name(argv[1]), policy(argv[3]), partition(argv[4]);
    const unsigned seed=std::strtoul(argv[5],nullptr,10);
    std::srand(seed);
    std::ifstream in(argv[2]);
    unsigned experts; in >> experts;
    if (!in || experts != 64) return 2;
    std::vector<unsigned> route; unsigned e;
    while (in >> e) { if (e>=64) return 2; route.push_back(e); }
    std::sort(route.begin(),route.end());
    Model model=Model::create(name);
    struct Weights { ConstantMatrix g,u,d; };
    std::vector<Weights> weights;
    for (unsigned i=0;i<64;++i) {
        weights.push_back({
            ConstantMatrix::create(model,"G"+std::to_string(i),1536,512),
            ConstantMatrix::create(model,"U"+std::to_string(i),1536,512),
            ConstantMatrix::create(model,"D"+std::to_string(i),512,1536)});
    }
    auto input=InputVector::create(model,"common_x",1536);
    Vector x(input);
    std::vector<Vector> outputs(64);
    for (unsigned i : route) {
        Vector gate=weights[i].g*x, up=weights[i].u*x;
        Vector z=(gate*sig(gate))*up;
        // Nontrivial compile-time route coefficients use existing MULI.
        outputs[i]=(float(i+1)/65.0f)*(weights[i].d*z);
        if (policy=="private") {
            auto out=OutputVector::create(model,"expert_out"+std::to_string(i),1536);
            out=outputs[i];
        }
    }
    if (policy=="aggregate") {
        // Mapping is fixed offline; only ALU reduction order changes by route.
        // Same local-then-global reduction rule is applied to every layout.
        std::vector<int> tile(1280);
        const char* map_path=std::getenv("FPIM_TILE_MAP");
        if (map_path) { std::ifstream f(map_path); for(int& t:tile) f>>t; }
        else if(partition=="random") {
            std::vector<unsigned> order(1280);
            for(unsigned n=0;n<1280;++n) order[n]=n;
            std::srand(seed); std::random_shuffle(order.begin(),order.end());
            for(unsigned r=0;r<1280;++r) tile[order[r]]=r/24;
            std::srand(seed); // Compiler's own shuffle must see the same seed.
        } else for(unsigned n=0;n<1280;++n) tile[n]=n/24;
        std::vector<Vector> rows;
        for (unsigned j=0;j<12;++j) {
            std::map<int,std::vector<unsigned>> groups;
            for(unsigned i:route) groups[tile[i*20+8+j]].push_back(i);
            std::vector<std::pair<unsigned,Vector>> partials;
            for(auto& group:groups) {
                Vector sum=extract_tile(model,outputs[group.second[0]],j);
                for(unsigned k=1;k<group.second.size();++k)
                    sum=sum+extract_tile(model,outputs[group.second[k]],j);
                partials.push_back({group.second.back(),sum});
            }
            // Native affinity places a newly ready ALU on its last-assigned
            // operand. Increasing semantic expert index keeps each local
            // reduction local and global reduction moves exactly once/group.
            std::sort(partials.begin(),partials.end(),
                      [](const std::pair<unsigned,Vector>& a,const std::pair<unsigned,Vector>& b){return a.first<b.first;});
            Vector sum=partials[0].second;
            for(unsigned k=1;k<partials.size();++k) sum=sum+partials[k].second;
            rows.push_back(sum);
        }
        auto out=OutputVector::create(model,"moe_sum",1536);
        out=alias_tiles(model,rows);
    } else if(policy!="private") return 2;
    CompilerOptions options;
    if(partition=="random") options.gp_=CompilerOptions::GP_RANDOM;
    else if(partition=="col") options.gp_=CompilerOptions::GP_COL_MAJOR;
    else options.gp_=CompilerOptions::GP_ROW_MAJOR;
    model.compile(options); model.destroy();
}
