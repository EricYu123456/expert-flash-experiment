// Train-only coordinate descent over immutable complete-row placement profiles.
#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

using A = std::array<int64_t, 16>;
struct Profile { A wave{}, input{}, output{}; int64_t commands = 0; };

static Profile make_profile(const int32_t *rows, int columns) {
  Profile p;
  for (int c = 0; c < 16; ++c) {
    p.wave[c] = (int64_t(rows[c]) * columns + 65535) / 65536;
    p.input[c] = rows[c] > 0;
    p.output[c] = int64_t(rows[c]) * 4;
    p.commands += (p.wave[c] + 191) / 192;
  }
  return p;
}

extern "C" double optimize_mixed_profiles(int32_t *rows,
    const int32_t *profile_rows, int profiles, const int32_t *routes,
    const int64_t *weights, int experts, int events, int active, int columns,
    int common, int max_passes, uint64_t seed, int64_t *stats) {
  std::vector<Profile> menu(profiles), current(experts);
  for (int p = 0; p < profiles; ++p)
    menu[p] = make_profile(profile_rows + p * 16, columns);
  std::vector<A> wave(events), input(events), output(events);
  std::vector<int64_t> commands(events), cost(events);
  std::vector<std::vector<int>> incidence(experts);
  for (int e = 0; e < experts; ++e) current[e] = make_profile(rows + e * 16, columns);
  for (int t = 0; t < events; ++t) {
    for (int k = 0; k < active; ++k) {
      int e = routes[t * active + k];
      incidence[e].push_back(t);
      for (int c = 0; c < 16; ++c) {
        wave[t][c] += current[e].wave[c];
        input[t][c] += current[e].input[c];
        output[t][c] += current[e].output[c];
      }
      commands[t] += current[e].commands;
    }
  }
  int64_t host_input = columns * (common ? 1LL : active);
  const auto value = [&](const A &w, const A &i, const A &o, int64_t cmd) {
    int64_t peak = 0, in = 0, out = 0, total_out = 0;
    for (int c = 0; c < 16; ++c) {
      peak = std::max(peak, w[c]);
      total_out += o[c];
    }
    for (int c = 0; c < 8; ++c) {
      int64_t a = common ? (i[c] > 0) : i[c];
      int64_t b = common ? (i[c + 8] > 0) : i[c + 8];
      in = std::max(in, (a + b) * columns);
      out = std::max(out, o[c] + o[c + 8]);
    }
    // For normal reads pages <= 4*waves, so pages*2560 <= waves*10240:
    // the 28000 ns/wave term always dominates the existing bandwidth ceiling.
    return peak * 28000 + cmd * 250 + 2000 +
        std::max((host_input + 7) / 8, (in + 1) / 2) +
        std::max((total_out + 7) / 8, (out + 1) / 2);
  };
  int64_t objective = 0, count = 0;
  for (int t = 0; t < events; ++t) {
    cost[t] = value(wave[t], input[t], output[t], commands[t]);
    objective += cost[t] * weights[t]; count += weights[t];
  }
  int64_t initial = objective, accepted = 0, evaluations = 0;
  std::vector<int> order(experts);
  std::iota(order.begin(), order.end(), 0);
  std::mt19937_64 rng(seed);
  int pass = 0; bool converged = false;
  for (; pass < max_passes; ++pass) {
    std::shuffle(order.begin(), order.end(), rng);
    int changes = 0;
    for (int e : order) {
      int best_profile = -1; int64_t best_delta = 0;
      for (int p = 0; p < profiles; ++p) {
        if (std::equal(rows + e * 16, rows + (e + 1) * 16, profile_rows + p * 16)) continue;
        ++evaluations;
        A dw{}, di{}, dout{};
        for (int c = 0; c < 16; ++c) {
          dw[c] = menu[p].wave[c] - current[e].wave[c];
          di[c] = menu[p].input[c] - current[e].input[c];
          dout[c] = menu[p].output[c] - current[e].output[c];
        }
        int64_t dc = menu[p].commands - current[e].commands, delta = 0;
        for (int t : incidence[e]) {
          A w = wave[t], i = input[t], o = output[t];
          for (int c = 0; c < 16; ++c) { w[c] += dw[c]; i[c] += di[c]; o[c] += dout[c]; }
          delta += (value(w, i, o, commands[t] + dc) - cost[t]) * weights[t];
        }
        if (delta < best_delta) { best_delta = delta; best_profile = p; }
      }
      if (best_profile < 0) continue;
      const Profile &next = menu[best_profile];
      for (int t : incidence[e]) {
        for (int c = 0; c < 16; ++c) {
          wave[t][c] += next.wave[c] - current[e].wave[c];
          input[t][c] += next.input[c] - current[e].input[c];
          output[t][c] += next.output[c] - current[e].output[c];
        }
        commands[t] += next.commands - current[e].commands;
        cost[t] = value(wave[t], input[t], output[t], commands[t]);
      }
      current[e] = next;
      std::copy(profile_rows + best_profile * 16, profile_rows + (best_profile + 1) * 16,
                rows + e * 16);
      objective += best_delta; ++accepted; ++changes;
    }
    if (!changes) { converged = true; ++pass; break; }
  }
  stats[0] = pass; stats[1] = accepted; stats[2] = evaluations;
  stats[3] = initial; stats[4] = objective; stats[5] = converged;
  return double(objective) / count;
}
