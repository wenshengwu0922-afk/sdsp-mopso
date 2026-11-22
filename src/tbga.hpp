// TBGA baseline (team-based GA, Li et al. 2020), adapted to two objectives (Section 4.1), with the same
// stopping-point evaluation (Algorithm 1) and external archive FA as MOPSO: layered feasible generation,
// three-parent PPX, duplicate control by Hamming distance, MHM / FBOO local search scored by the exclusive HV
// contribution w.r.t. the archive, and team competition. Fitness of an individual = HV_pt (Eq. 14) of its
// representative stopping point (Eq. 15). Parameters (Table 11): team_ratio = T_R, delta = delta,
// nm_ratio = N_R; population 100.
#pragma once
#include "common.hpp"

namespace ssdsp {

namespace tb {
struct Ind { std::vector<int> seq; Eval ev; };

inline int shd(const std::vector<int>& a, const std::vector<int>& b) {
    int d = 0;
    for (size_t i = 0; i < a.size(); ++i) d += (a[i] != b[i]);
    return d;
}

// layers of the precedence graph (longest predecessor chain); shuffling inside each layer is always feasible
inline std::vector<std::vector<int>> layers(const Problem& P) {
    const int n = P.N;
    std::vector<int> indeg(n + 1, 0), lev(n + 1, 0);
    for (int i = 1; i <= n; ++i) for (int j = 1; j <= n; ++j) if (P.prec[i][j]) ++indeg[j];
    std::vector<int> q;
    for (int i = 1; i <= n; ++i) if (!indeg[i]) q.push_back(i);
    for (size_t h = 0; h < q.size(); ++h) {
        const int u = q[h];
        for (int v = 1; v <= n; ++v)
            if (P.prec[u][v]) { lev[v] = std::max(lev[v], lev[u] + 1); if (--indeg[v] == 0) q.push_back(v); }
    }
    const int L = *std::max_element(lev.begin() + 1, lev.end());
    std::vector<std::vector<int>> out(L + 1);
    for (int i = 1; i <= n; ++i) out[lev[i]].push_back(i);
    return out;
}

inline std::vector<int> ffsg(const std::vector<std::vector<int>>& lay, Rng& g) {
    std::vector<int> s;
    for (auto l : lay) { shuffle_vec(l, g); s.insert(s.end(), l.begin(), l.end()); }
    return s;
}

// three-parent PPX: each gene from a random parent's earliest unused part (feasible by construction)
inline std::vector<int> ppx3(const std::vector<int>* par3[3], int n, Rng& g) {
    std::vector<int> child, ptr(3, 0);
    std::vector<char> used(n + 1, 0);
    for (int k = 0; k < n; ++k) {
        const int src = Uint(g, 0, 2);
        int chosen = -1;
        for (int s = 0; s < 3 && chosen < 0; ++s) {
            const int id = (src + s) % 3;
            const auto& p = *par3[id];
            while (ptr[id] < n && used[p[ptr[id]]]) ++ptr[id];
            if (ptr[id] < n) chosen = p[ptr[id]];
        }
        used[chosen] = 1; child.push_back(chosen);
    }
    return child;
}

inline bool converged(const std::vector<Ind>& tm, double delta) {
    double best = tm[0].ev.hv, sum = 0;
    for (const auto& x : tm) { best = std::max(best, x.ev.hv); sum += x.ev.hv; }
    return std::fabs(best - sum / tm.size()) < delta;
}
}  // namespace tb

inline void run_tbga(Evaluator& E, Rng& g, const Params& prm) {
    using namespace tb;
    const Problem& P = E.P;
    const int NP = 100, n = P.N;
    // Table 11: T_R = team size / population size, delta = team convergence threshold, N_R = MHM moves / N
    const double team_ratio = par(prm, "team_ratio"), delta = par(prm, "delta"), nm_ratio = par(prm, "nm_ratio");
    const int team = std::max(2, std::min((int)std::lround(NP * team_ratio), NP));
    const int NT = std::max(1, NP / team);
    const int NM = std::max(1, (int)std::lround(n * nm_ratio));   // round(N_R * N) MHM moves (Table 11 note)
    const auto lay = layers(P);
    auto by_hv = [](const Ind& a, const Ind& b) { return a.ev.hv > b.ev.hv; };
    auto contrib = [&](const Ind& x) { return E.exclusive_contribution(x.ev.f1, x.ev.f2, x.seq); };

    auto unique_pop = [&](int cnt) {
        std::vector<Ind> pop;
        for (int guard = 0; (int)pop.size() < cnt && guard < cnt * 50; ++guard) {
            std::vector<int> s = ffsg(lay, g);
            bool dup = false;
            for (const auto& o : pop) if (shd(s, o.seq) < 3) { dup = true; break; }
            if (!dup) pop.push_back({s, E.evaluate(s)});
        }
        while ((int)pop.size() < cnt) { std::vector<int> s = ffsg(lay, g); pop.push_back({s, E.evaluate(s)}); }
        return pop;
    };

    // MHM: move one part to the best feasible insertion position (scored by exclusive HV contribution)
    auto mhm = [&](Ind& x) {
        for (int it = 0; it < NM; ++it) {
            const int mp = Uint(g, 0, n - 1), node = x.seq[mp];
            std::vector<int> rem;
            for (int i = 0; i < n; ++i) if (i != mp) rem.push_back(x.seq[i]);
            int lo = 0, hi = n - 1;
            for (int i = 0; i < n - 1; ++i) {
                if (P.prec[rem[i]][node]) lo = std::max(lo, i + 1);
                if (P.prec[node][rem[i]]) hi = std::min(hi, i);
            }
            if (lo > hi) continue;
            Ind best; double bsc = -1;
            for (int ins = lo; ins <= hi; ++ins) {
                Ind c; c.seq = rem; c.seq.insert(c.seq.begin() + ins, node);
                c.ev = (c.seq == x.seq) ? x.ev : E.evaluate(c.seq);
                const double sc = contrib(c);
                if (sc > bsc) { bsc = sc; best = std::move(c); }
            }
            x = std::move(best);
        }
    };
    // FBOO: forward and backward passes over adjacent swappable pairs; keep a swap if the contribution
    // does not decrease
    auto fboo = [&](Ind& x) {
        auto try_swap = [&](int i) {
            if (P.prec[x.seq[i]][x.seq[i + 1]] || P.prec[x.seq[i + 1]][x.seq[i]]) return;
            const double before = contrib(x);
            Ind y = x; std::swap(y.seq[i], y.seq[i + 1]);
            y.ev = E.evaluate(y.seq);
            if (contrib(y) >= before) x = std::move(y);
        };
        for (int i = 0; i + 1 < n; ++i) try_swap(i);
        for (int i = n - 2; i >= 0; --i) try_swap(i);
    };

    std::vector<std::vector<Ind>> teams(NT);
    {
        auto all = unique_pop(NP);
        for (int i = 0; i < NP; ++i) teams[std::min(i / team, NT - 1)].push_back(all[i]);
    }
    for (;;) {   // terminates by BudgetExhausted
        const uint64_t e_before = E.evaluations();
        for (auto& tm : teams) {
            const int sz = (int)tm.size();
            if (sz < 3) continue;
            std::stable_sort(tm.begin(), tm.end(), by_hv);
            if (!converged(tm, delta)) {
                const int b1 = std::max(1, sz * 15 / 100), b2 = std::max(b1 + 1, sz * 30 / 100),
                          b3 = std::max(b2 + 1, sz * 45 / 100);
                std::vector<Ind> ch;
                for (int c = 0; c < sz; ++c) {
                    const int i1 = Uint(g, 0, b1 - 1);
                    const int i2 = Uint(g, b1, b2 - 1);
                    const int i3 = Uint(g, b2, b3 - 1);
                    const std::vector<int>* par3[3] = {&tm[i1].seq, &tm[i2].seq, &tm[i3].seq};
                    Ind k; k.seq = ppx3(par3, n, g);
                    k.ev = E.evaluate(k.seq);                                   // every offspring: one evaluation
                    ch.push_back(std::move(k));
                }
                for (auto& c : ch) tm.push_back(std::move(c));
            }
            std::stable_sort(tm.begin(), tm.end(), by_hv);
            const int cm = Uint(g, 0, std::max(0, (int)tm.size() / 2 - 1));
            mhm(tm[cm]);
            fboo(tm[cm]);
            std::stable_sort(tm.begin(), tm.end(), by_hv);
            if ((int)tm.size() > team) tm.resize(team);
        }
        bool all_conv = true;
        for (const auto& tm : teams) if (!converged(tm, delta)) { all_conv = false; break; }
        if (all_conv && NT >= 2) {   // team competition: of the two most similar teams, replace the weaker one
            int wi = -1, wj = -1, best_d = 1 << 30;
            for (int i = 0; i < NT; ++i)
                for (int j = i + 1; j < NT; ++j) {
                    const int dd = shd(teams[i][0].seq, teams[j][0].seq);
                    if (dd < best_d) { best_d = dd; wi = i; wj = j; }
                }
            const int loser = teams[wi][0].ev.hv < teams[wj][0].ev.hv ? wi : wj;
            teams[loser] = unique_pop(team);
            std::stable_sort(teams[loser].begin(), teams[loser].end(), by_hv);
        }
        if (E.evaluations() == e_before) throw std::runtime_error("no evaluation in an iteration");
    }
}

}  // namespace ssdsp
