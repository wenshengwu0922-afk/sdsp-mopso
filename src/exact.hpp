// Exact Pareto front over all feasible stopping states (Section 4.1, Table 8). Both objectives (Eqs. 2-3) depend
// only on the set of removed parts, and the feasible removed sets are exactly the order ideals of the precedence
// relation, so the front over all (sequence, stopping point) pairs is obtained by enumerating the ideals
// (stapler: 415 states, 18 Pareto points; ceiling fan: 24,839 states, 31 Pareto points). Its HV is HV*.
#pragma once
#include <cstdint>
#include <unordered_set>
#include <vector>

#include "evaluator.hpp"

namespace ssdsp {

struct ExactPoint { double f1, f2; int k; };

inline std::vector<ExactPoint> exact_front(const Problem& P, size_t* n_states = nullptr) {
    const int N = P.N;
    if (N > 63) throw std::runtime_error("exact enumeration supports N <= 63");
    std::vector<uint64_t> predmask(N + 1, 0);
    for (int a = 1; a <= N; ++a)
        for (int b = 1; b <= N; ++b)
            if (P.prec[a][b]) predmask[b] |= (uint64_t(1) << (a - 1));
    std::unordered_set<uint64_t> seen{0};
    std::vector<uint64_t> stack{0};
    std::vector<ExactPoint> pts;
    while (!stack.empty()) {
        const uint64_t S = stack.back(); stack.pop_back();
        if (S) {
            double f1 = 0, f2 = 0; int k = 0;
            for (int i = 1; i <= N; ++i) {
                const bool in = (S >> (i - 1)) & 1;
                if (in) { ++k; f2 += P.g[i]; } else f2 += P.gbulk[i];
                for (const auto& [v, c] : P.adj[i])          // liaison cost counted once (from its smaller end)
                    if (i < v && (in || ((S >> (v - 1)) & 1))) f1 += c;
            }
            pts.push_back({f1, f2, k});                       // one feasible stopping state
        }
        for (int i = 1; i <= N; ++i) {
            const uint64_t bit = uint64_t(1) << (i - 1);
            if (!(S & bit) && (predmask[i] & ~S) == 0 && seen.insert(S | bit).second) stack.push_back(S | bit);
        }
    }
    if (n_states) *n_states = pts.size();
    std::sort(pts.begin(), pts.end(), [](const ExactPoint& a, const ExactPoint& b) {
        return a.f1 != b.f1 ? a.f1 < b.f1 : a.f2 > b.f2; });
    std::vector<ExactPoint> front;                         // non-dominated states = exact Pareto front
    for (const auto& p : pts)
        if (front.empty() || p.f2 > front.back().f2 + 1e-9) front.push_back(p);
    return front;
}

}  // namespace ssdsp
