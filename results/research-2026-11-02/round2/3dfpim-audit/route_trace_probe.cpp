#include "3dfpim.h"

#include <cstdlib>
#include <iostream>
#include <string>

// Research probe only: invokes the unmodified official trace compiler.
// Both weight matrices exist in every invocation. The route changes operations,
// never the matrix inventory or its construction order. No numerical execution.
int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: probe model-name route-mask input-mode partition-mode\n";
        return 2;
    }
    const std::string model_name(argv[1]);
    const unsigned route = std::strtoul(argv[2], nullptr, 10);
    const std::string input_mode(argv[3]);
    const std::string partition_mode(argv[4]);
    if (route < 1 || route > 3 ||
        (input_mode != "shared" && input_mode != "private")) {
        return 2;
    }
    Model model = Model::create(model_name);
    ConstantMatrix weights[2] = {
        ConstantMatrix::create(model, "expert0", 512, 512),
        ConstantMatrix::create(model, "expert1", 512, 512)
    };
    InputVector inputs[2] = {
        InputVector::create(model, "x0", 512),
        InputVector::create(model, "x1", 512)
    };
    for (unsigned expert = 0; expert < 2; ++expert) {
        if ((route & (1U << expert)) == 0) continue;
        const unsigned input_id = input_mode == "shared" ? 0 : expert;
        Vector result = weights[expert] * Vector(inputs[input_id]);
        auto output = OutputVector::create(
            model, "out" + std::to_string(expert), 512);
        output = result;
    }
    CompilerOptions options;
    if (partition_mode == "row") options.gp_ = CompilerOptions::GP_ROW_MAJOR;
    else if (partition_mode == "col") options.gp_ = CompilerOptions::GP_COL_MAJOR;
    else if (partition_mode == "random") options.gp_ = CompilerOptions::GP_RANDOM;
    else return 2;
    model.compile(options);
    model.destroy();
    return 0;
}
