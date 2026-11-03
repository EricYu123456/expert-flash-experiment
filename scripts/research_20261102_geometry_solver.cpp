// Fixed whole-row geometries; only expert labels are swapped during training.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

extern "C" double optimize_geometry(int32_t *rows, const int32_t *routes,
    const int64_t *weights, int experts, int events, int active, int columns,
    int common, int iterations, uint64_t seed, double temperature_ns,
    int64_t *stats) {
  using A = std::array<int64_t, 16>;
  const auto nw = [columns](int64_t r) { return (r * columns + 65535) / 65536; };
  const auto nc = [&](int64_t r) { return (nw(r) + 191) / 192; };
  std::vector<A> wave(events), input(events), output(events);
  std::vector<int64_t> command(events);
  std::vector<std::vector<int>> incidence(experts);
  std::vector<std::vector<uint8_t>> member(experts,
                                          std::vector<uint8_t>(events));
  for (int t = 0; t < events; ++t) {
    for (int k = 0; k < active; ++k) {
      int e = routes[t * active + k];
      incidence[e].push_back(t); member[e][t] = 1;
      for (int c = 0; c < 16; ++c) {
        int r = rows[e * 16 + c];
        wave[t][c] += nw(r); input[t][c] += (r > 0);
        output[t][c] += int64_t(r) * 4;
        command[t] += nc(r);
      }
    }
  }
  const int64_t host_in = columns * (common ? 1 : active);
  const auto cost = [&](int t) {
    int64_t peak = 0, in = 0, out = 0, total_out = 0;
    for (int c = 0; c < 16; ++c) {
      peak = std::max(peak, wave[t][c]); total_out += output[t][c];
    }
    for (int c = 0; c < 8; ++c) {
      int64_t a = common ? (input[t][c] > 0) : input[t][c];
      int64_t b = common ? (input[t][c + 8] > 0) : input[t][c + 8];
      in = std::max(in, (a + b) * columns);
      out = std::max(out, output[t][c] + output[t][c + 8]);
    }
    return peak * 28000 + command[t] * 250 + 2000 +
        std::max((host_in + 7) / 8, (in + 1) / 2) +
        std::max((total_out + 7) / 8, (out + 1) / 2);
  };
  std::vector<int64_t> costs(events);
  int64_t total = 0, count = 0;
  for (int t = 0; t < events; ++t) {
    costs[t] = cost(t); total += costs[t] * weights[t]; count += weights[t];
  }
  int64_t best = total;
  std::vector<int32_t> best_rows(rows, rows + experts * 16);
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> uniform(0, 1);
  std::vector<int> affected;
  std::vector<int64_t> proposed;
  int accepted = 0, valid = 0;
  for (int it = 0; it < iterations; ++it) {
    int a = rng() % experts, b = rng() % experts;
    if (a == b || std::equal(rows + a*16, rows + (a+1)*16, rows + b*16)) continue;
    ++valid;
    A dw{}, di{}, dout{};
    int64_t dc = 0;
    for (int c = 0; c < 16; ++c) {
      int ra = rows[a*16+c], rb = rows[b*16+c];
      dw[c] = nw(rb) - nw(ra); di[c] = (rb > 0) - (ra > 0);
      dout[c] = int64_t(rb-ra) * 4; dc += nc(rb) - nc(ra);
    }
    affected.clear();
    for (int t : incidence[a]) if (!member[b][t]) affected.push_back(t);
    for (int t : incidence[b]) if (!member[a][t]) affected.push_back(t);
    proposed.resize(affected.size());
    int64_t delta = 0;
    for (size_t j = 0; j < affected.size(); ++j) {
      int t = affected[j], sign = member[a][t] ? 1 : -1;
      for (int c = 0; c < 16; ++c) {
        wave[t][c] += sign*dw[c]; input[t][c] += sign*di[c];
        output[t][c] += sign*dout[c];
      }
      command[t] += sign*dc;
      proposed[j] = cost(t); delta += (proposed[j]-costs[t])*weights[t];
    }
    double temperature = temperature_ns * std::pow(0.001, double(it)/std::max(1,iterations-1));
    bool accept = delta <= 0 || (temperature > 0 &&
        uniform(rng) < std::exp(-double(delta)/count/temperature));
    if (accept) {
      ++accepted; total += delta;
      for (int c = 0; c < 16; ++c) std::swap(rows[a*16+c],rows[b*16+c]);
      for (size_t j = 0; j < affected.size(); ++j) costs[affected[j]] = proposed[j];
      if (total < best) {
        best = total; std::copy(rows,rows+experts*16,best_rows.begin());
      }
    } else {
      for (int t : affected) {
        int sign = member[a][t] ? 1 : -1;
        for (int c = 0; c < 16; ++c) {
          wave[t][c] -= sign*dw[c]; input[t][c] -= sign*di[c];
          output[t][c] -= sign*dout[c];
        }
        command[t] -= sign*dc;
      }
    }
  }
  std::copy(best_rows.begin(), best_rows.end(), rows);
  stats[0] = valid; stats[1] = accepted;
  return double(best)/count;
}
