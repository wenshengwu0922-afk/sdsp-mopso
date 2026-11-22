// Shared operators: real-coded decoding with precedence repair (Section 3.1), leader encoding (Eq. 18), and the
// PPX crossover and swap mutation used by the GA-based baselines.
#pragma once
#include <algorithm>
#include <map>
#include <numeric>
#include <queue>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "evaluator.hpp"

namespace ssdsp {

using Rng = std::mt19937;
using Params = std::map<std::string, double>;

inline double par(const Params& p, const std::string& k) {
    auto it = p.find(k);
    if (it == p.end()) throw std::invalid_argument("missing parameter " + k);
    return it->second;
}
// Portable random numbers: only the raw mt19937 output (identical on every platform) is used, so a given
// seed reproduces the same run with MinGW, MSVC or GCC. std::uniform_*_distribution / std::shuffle are
// implementation-defined and are therefore not used.
inline double U01(Rng& g) { return g() * (1.0 / 4294967296.0); }                  // [0, 1)
inline int Uint(Rng& g, int lo, int hi) { return lo + (int)(g() % (uint32_t)(hi - lo + 1)); }
template <class T> inline void shuffle_vec(std::vector<T>& v, Rng& g) {          // Fisher-Yates
    for (int i = (int)v.size() - 1; i > 0; --i) std::swap(v[i], v[Uint(g, 0, i)]);
}

// Precedence repair (Section 3.1): whenever a part appears before one of its predecessors defined by the
// precedence matrix A, the two parts are swapped; the scan is repeated until no violation remains. If pos != nullptr,
// the position values are swapped as well, so that the position vector X_p stays consistent with the sequence.
inline void repair(std::vector<int>& seq, const Problem& P, std::vector<double>* pos = nullptr) {
    const int n = (int)seq.size();
    bool swapped;
    do {
        swapped = false;
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                if (P.prec[seq[j]][seq[i]]) {
                    if (pos) std::swap((*pos)[seq[i] - 1], (*pos)[seq[j] - 1]);
                    std::swap(seq[i], seq[j]);
                    swapped = true;
                }
    } while (swapped);
}

// DECODE-REPAIR(X_p, A) (Section 3.1): argsort of the position vector X_p in [0,1]^N gives a tentative permutation
// S_p, which is then repaired; the repaired S_p is always feasible.
inline std::vector<int> decode_repair(std::vector<double>& pos, const Problem& P) {
    std::vector<int> seq(P.N);
    std::iota(seq.begin(), seq.end(), 1);
    std::stable_sort(seq.begin(), seq.end(), [&](int a, int b) { return pos[a - 1] < pos[b - 1]; });
    repair(seq, P, &pos);
    return seq;
}

// Eq. (18): position vector X^a of an archive member S^a, x^a_{s_j} = (j - 0.5)/N, j = 1..N (k = j - 1 below);
// decoding X^a returns S^a.
inline std::vector<double> encode(const std::vector<int>& seq) {
    const int n = (int)seq.size();
    std::vector<double> pos(n);
    for (int k = 0; k < n; ++k) pos[seq[k] - 1] = (k + 0.5) / n;
    return pos;
}

// Uniformly shuffled permutation made feasible by precedence repair (initial GA populations, random search).
inline std::vector<int> random_feasible(const Problem& P, Rng& g) {
    std::vector<int> s(P.N);
    std::iota(s.begin(), s.end(), 1);
    shuffle_vec(s, g);
    repair(s, P);
    return s;
}

// Swap mutation of the GA baselines: swap two random positions, then repair.
inline void swap_mutation(std::vector<int>& seq, const Problem& P, Rng& g) {
    const int n = (int)seq.size();
    const int i = Uint(g, 0, n - 1);
    int j = Uint(g, 0, n - 1);
    while (i == j) j = Uint(g, 0, n - 1);
    std::swap(seq[i], seq[j]);
    repair(seq, P);
}

// Precedence-preserving crossover (PPX): each gene is taken from the parent selected by a random mask,
// choosing that parent's earliest part whose predecessors are all placed. Children are always feasible.
inline std::vector<int> ppx_child(const std::vector<int>& p1, const std::vector<int>& p2, const Problem& P, Rng& g) {
    const int n = P.N;
    std::vector<int> pos1(n + 1), pos2(n + 1), indeg(n + 1, 0);
    for (int i = 0; i < n; ++i) { pos1[p1[i]] = i; pos2[p2[i]] = i; }
    for (int u = 1; u <= n; ++u)
        for (int v = 1; v <= n; ++v)
            if (P.prec[u][v]) ++indeg[v];
    using Node = std::pair<int, int>;   // (position in parent, part)
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> q1, q2;
    for (int t = 1; t <= n; ++t)
        if (indeg[t] == 0) { q1.push({pos1[t], t}); q2.push({pos2[t], t}); }
    std::vector<char> used(n + 1, 0);
    auto pop = [&](auto& q) {
        while (!q.empty() && used[q.top().second]) q.pop();
        if (q.empty()) return -1;
        const int t = q.top().second;
        q.pop();
        return t;
    };
    std::vector<int> child;
    child.reserve(n);
    for (int k = 0; k < n; ++k) {
        const int pick = (U01(g) < 0.5) ? pop(q1) : pop(q2);   // both queues hold the same ready parts
        if (pick < 0) throw std::runtime_error("PPX failed: cyclic precedence graph");
        child.push_back(pick); used[pick] = 1;
        for (int v = 1; v <= n; ++v)
            if (P.prec[pick][v] && --indeg[v] == 0) { q1.push({pos1[v], v}); q2.push({pos2[v], v}); }
    }
    return child;
}

// Pareto dominance between the representative objective vectors (f1, f2) at k_rep of two sequences
// (personal-best update, Section 3.3; NSGA-II sorting).
inline bool dom_eval(const Eval& a, const Eval& b) { return dominates(a.f1, a.f2, b.f1, b.f2); }

}  // namespace ssdsp
