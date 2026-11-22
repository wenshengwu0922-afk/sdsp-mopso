// NSGA-II baseline (Deb et al. 2002), with the same stopping-point evaluation (Algorithm 1) and external archive FA
// as MOPSO (Section 4.1); sorting on the representative objective vectors (k_rep, Eq. 15): fast non-dominated
// sorting, crowding distance, crowded binary tournament, PPX, swap mutation. Parameters (Table 11): p_c (pc),
// p_m (pm); population 100.
#pragma once
#include "common.hpp"

namespace ssdsp {

namespace ns2 {
struct Ind { std::vector<int> seq; Eval ev; int rank = 0; double cd = 0; };

inline std::vector<std::vector<int>> nd_sort(std::vector<Ind>& Pp) {
    const int n = (int)Pp.size();
    std::vector<std::vector<int>> S(n), F(1);
    std::vector<int> cnt(n, 0);
    for (int p = 0; p < n; ++p) {
        for (int q = 0; q < n; ++q) {
            if (p == q) continue;
            if (dom_eval(Pp[p].ev, Pp[q].ev)) S[p].push_back(q);
            else if (dom_eval(Pp[q].ev, Pp[p].ev)) ++cnt[p];
        }
        if (cnt[p] == 0) { Pp[p].rank = 0; F[0].push_back(p); }
    }
    for (int i = 0; !F[i].empty(); ++i) {
        std::vector<int> nx;
        for (int p : F[i]) for (int q : S[p]) if (--cnt[q] == 0) { Pp[q].rank = i + 1; nx.push_back(q); }
        F.push_back(nx);
    }
    F.pop_back();
    return F;
}

inline void crowding(std::vector<Ind>& Pp, std::vector<int> f) {
    const double INF = 1e18;
    for (int i : f) Pp[i].cd = 0;
    if (f.size() <= 2) { for (int i : f) Pp[i].cd = INF; return; }
    for (int obj = 0; obj < 2; ++obj) {
        auto val = [&](int i) { return obj == 0 ? Pp[i].ev.f1 : Pp[i].ev.f2; };
        std::stable_sort(f.begin(), f.end(), [&](int a, int b) { return val(a) < val(b); });
        Pp[f.front()].cd = Pp[f.back()].cd = INF;
        const double rg = std::max(1e-12, val(f.back()) - val(f.front()));
        for (size_t k = 1; k + 1 < f.size(); ++k)
            if (Pp[f[k]].cd < INF) Pp[f[k]].cd += (val(f[k + 1]) - val(f[k - 1])) / rg;
    }
}

inline void rank_and_crowd(std::vector<Ind>& Pp) { for (auto& f : nd_sort(Pp)) crowding(Pp, f); }

inline int tournament(const std::vector<Ind>& Pp, Rng& g) {
    const int a = Uint(g, 0, (int)Pp.size() - 1);
    const int b = Uint(g, 0, (int)Pp.size() - 1);
    if (Pp[a].rank != Pp[b].rank) return Pp[a].rank < Pp[b].rank ? a : b;
    return Pp[a].cd > Pp[b].cd ? a : b;
}
}  // namespace ns2

inline void run_nsga2(Evaluator& E, Rng& g, const Params& prm) {
    using namespace ns2;
    const Problem& P = E.P;
    const int NP = 100;
    const double pc = par(prm, "pc"), pm = par(prm, "pm");

    std::vector<Ind> pop(NP);
    for (auto& ind : pop) { ind.seq = random_feasible(P, g); ind.ev = E.evaluate(ind.seq); }
    rank_and_crowd(pop);
    for (;;) {   // terminates by BudgetExhausted
        const uint64_t e_before = E.evaluations();
        std::vector<Ind> comb = pop;
        while ((int)comb.size() < 2 * NP) {
            const int pa = tournament(pop, g);
            const int pb = tournament(pop, g);
            const bool cross = U01(g) < pc;
            for (int c = 0; c < 2 && (int)comb.size() < 2 * NP; ++c) {
                Ind child;
                child.seq = cross ? ppx_child(pop[pa].seq, pop[pb].seq, P, g) : pop[c == 0 ? pa : pb].seq;
                if (U01(g) < pm) swap_mutation(child.seq, P, g);
                child.ev = E.evaluate(child.seq);                             // every offspring: one evaluation
                comb.push_back(std::move(child));
            }
        }
        // environmental selection: whole fronts, last front truncated by crowding distance
        auto fronts = nd_sort(comb);
        std::vector<Ind> next;
        next.reserve(NP);
        for (auto& f : fronts) {
            if ((int)(next.size() + f.size()) <= NP) { for (int i : f) next.push_back(comb[i]); continue; }
            crowding(comb, f);
            std::stable_sort(f.begin(), f.end(), [&](int a, int b) { return comb[a].cd > comb[b].cd; });
            for (int i = 0; (int)next.size() < NP; ++i) next.push_back(comb[f[i]]);
            break;
        }
        pop.swap(next);
        rank_and_crowd(pop);
        if (E.evaluations() == e_before) throw std::runtime_error("no evaluation in an iteration");
    }
}

}  // namespace ssdsp
