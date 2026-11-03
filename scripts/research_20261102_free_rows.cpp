// Train-only immutable row placement. A 2x2 exchange preserves every expert's
// shard multiset AND every chip's stored row/page/wave multiset.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

extern "C" double optimize_rows(int32_t *rows, const int32_t *routes,
    const int64_t *weights, int experts, int events, int active, int columns,
    int common, int iterations, uint64_t seed, double start_temperature,
    int64_t *stats) {
  using A = std::array<int64_t, 16>;
  std::vector<A> wave(events), input(events), output(events);
  std::vector<std::vector<int>> incident(experts);
  std::vector<std::vector<uint8_t>> member(experts,
                                         std::vector<uint8_t>(events));
  const auto w = [columns](int64_t r) { return (r * columns + 65535) / 65536; };
  for (int t = 0; t < events; ++t) {
    for (int k = 0; k < active; ++k) {
      int e = routes[t * active + k];
      incident[e].push_back(t);
      member[e][t] = 1;
      for (int c = 0; c < 16; ++c) {
        int64_t r = rows[e * 16 + c];
        wave[t][c] += w(r);
        input[t][c] += (r > 0);
        output[t][c] += r * 4;
      }
    }
  }
  int64_t host_input = columns * (common ? 1 : active);
  const auto cost = [&](int t) {
    int64_t max_wave = 0, max_input = 0, max_output = 0, total_output = 0;
    for (int c = 0; c < 16; ++c) {
      max_wave = std::max(max_wave, wave[t][c]);
      total_output += output[t][c];
    }
    for (int c = 0; c < 8; ++c) {
      int64_t a = common ? (input[t][c] > 0) : input[t][c];
      int64_t b = common ? (input[t][c + 8] > 0) : input[t][c + 8];
      max_input = std::max(max_input, (a + b) * columns);
      max_output = std::max(max_output, output[t][c] + output[t][c + 8]);
    }
    return max_wave * 28000 +
      std::max((host_input + 7) / 8, (max_input + 1) / 2) +
      std::max((total_output + 7) / 8, (max_output + 1) / 2);
  };
  std::vector<int64_t> costs(events);
  int64_t total = 0, count = 0;
  for (int t = 0; t < events; ++t) {
    costs[t] = cost(t);
    total += costs[t] * weights[t]; count += weights[t];
  }
  int64_t best = total;
  std::vector<int32_t> best_rows(rows, rows + experts * 16);
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> uniform(0, 1);
  std::vector<int> affected, candidates;
  std::vector<int64_t> proposed;
  int valid = 0, accepted = 0, improving = 0;
  for (int iteration = 0; iteration < iterations; ++iteration) {
    int a = rng() % experts, b = rng() % experts;
    if (a == b) continue;
    int c = rng() % 16;
    int ra = rows[a * 16 + c], rb = rows[b * 16 + c];
    if (ra == rb) continue;
    candidates.clear();
    for (int d = 0; d < 16; ++d)
      if (rows[a * 16 + d] == rb && rows[b * 16 + d] == ra)
        candidates.push_back(d);
    if (candidates.empty()) continue;
    int d = candidates[rng() % candidates.size()];
    ++valid;
    int64_t dw = w(rb) - w(ra), di = (rb > 0) - (ra > 0);
    int64_t dr = int64_t(rb - ra) * 4;
    affected.clear();
    for (int t : incident[a]) if (!member[b][t]) affected.push_back(t);
    for (int t : incident[b]) if (!member[a][t]) affected.push_back(t);
    proposed.resize(affected.size());
    int64_t delta = 0;
    for (size_t j = 0; j < affected.size(); ++j) {
      int t = affected[j], sign = member[a][t] ? 1 : -1;
      wave[t][c] += sign * dw; wave[t][d] -= sign * dw;
      input[t][c] += sign * di; input[t][d] -= sign * di;
      output[t][c] += sign * dr; output[t][d] -= sign * dr;
      proposed[j] = cost(t);
      delta += (proposed[j] - costs[t]) * weights[t];
    }
    double temperature = start_temperature *
      std::pow(0.001, double(iteration) / std::max(1, iterations - 1));
    bool accept = delta <= 0 || (temperature > 0 &&
      uniform(rng) < std::exp(-double(delta) / count / temperature));
    if (accept) {
      ++accepted; improving += delta < 0;
      total += delta;
      std::swap(rows[a * 16 + c], rows[a * 16 + d]);
      std::swap(rows[b * 16 + c], rows[b * 16 + d]);
      for (size_t j = 0; j < affected.size(); ++j) costs[affected[j]] = proposed[j];
      if (total < best) {
        best = total;
        std::copy(rows, rows + experts * 16, best_rows.begin());
      }
    } else {
      for (int t : affected) {
        int sign = member[a][t] ? 1 : -1;
        wave[t][c] -= sign * dw; wave[t][d] += sign * dw;
        input[t][c] -= sign * di; input[t][d] += sign * di;
        output[t][c] -= sign * dr; output[t][d] += sign * dr;
      }
    }
  }
  std::copy(best_rows.begin(), best_rows.end(), rows);
  stats[0] = valid; stats[1] = accepted; stats[2] = improving;
  return double(best) / count;
}
