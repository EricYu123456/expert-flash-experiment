// Native host-only I64 reconstruction microbenchmark, not an AiF measurement.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

extern "C" void recover_i32(const int32_t* low, const int32_t* high,
                            const int32_t* known, int64_t* recovered,
                            int rows, int experts, int known_shards) {
    for (int expert = 0; expert < experts; ++expert) {
        for (int row = 0; row < rows; ++row) {
            const size_t j = size_t(expert) * rows + row;
            int64_t value = int64_t(low[j]) + int64_t(high[j]) * 256;
            for (int shard = 0; shard < known_shards; ++shard)
                value -= known[(size_t(expert) * known_shards + shard) * rows + row];
            recovered[j] = value;
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    const int rows = std::atoi(argv[1]);
    const int experts = std::atoi(argv[2]);
    const int known_shards = std::atoi(argv[3]);
    if (rows < 1 || experts < 1 || known_shards < 1) return 2;
    const size_t n = size_t(rows) * experts;
    std::vector<int32_t> low(n), high(n), known(n * known_shards);
    std::vector<int64_t> recovered(n);
    for (size_t j = 0; j < n; ++j) { low[j] = int(j % 1009); high[j] = int(j % 13) - 6; }
    for (size_t j = 0; j < known.size(); ++j) known[j] = int(j % 10007) - 5003;
    auto decode = [&]() {
        recover_i32(low.data(), high.data(), known.data(), recovered.data(), rows, experts, known_shards);
        asm volatile("" : : "g"(recovered.data()) : "memory");
    };
    for (int i = 0; i < 100; ++i) decode();
    std::vector<double> sample;
    constexpr int repetitions = 200;
    for (int run = 0; run < 31; ++run) {
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < repetitions; ++i) {
            known[(size_t(i) * 7919 + run) % known.size()] ^= 1;
            decode();
        }
        auto end = std::chrono::steady_clock::now();
        sample.push_back(std::chrono::duration<double, std::nano>(end-start).count()/repetitions);
    }
    std::sort(sample.begin(), sample.end());
    std::cout << "{\"rows\":" << rows << ",\"experts\":" << experts
              << ",\"known_shards\":" << known_shards
              << ",\"median_ns\":" << sample[15] << ",\"p95_ns\":" << sample[29]
              << ",\"checksum\":" << recovered[n/2] << "}\n";
}
