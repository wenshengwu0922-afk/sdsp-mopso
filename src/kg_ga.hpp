// K&G-GA baseline (Kongar and Gupta 2006), adapted to two objectives (Section 4.1), with the same stopping-point
// evaluation (Algorithm 1) and external archive FA as MOPSO. Fitness = exclusive HV contribution of an individual's
// representative point (k_rep, Eq. 15) within the population; roulette parent and survivor selection;
// precedence-preserving crossover (PPX); swap mutation. Parameters (Table 11): crossover probability p_c (pc),
// mutation probability p_m (pm); population 100. Every offspring costs one evaluation; surviving parents keep their
// known objective values (no re-evaluation).
#pragma once
#include "common.hpp"

namespace ssdsp {

namespace kg {
struct Ind { std::vector<int> seq; Eval ev; };

// exclusive HV contribution of each member (dominated members get 0), normalized space, reference 1.1
inline std::vector<double> contributions(const std::vector<Ind>& pop, const Problem& P) {
    const int n = (int)pop.size();
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    auto X = [&](int i) { return P.sigma1(pop[i].ev.f1); };
    auto Y = [&](int i) { return P.sigma2(pop[i].ev.f2); };
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return X(a) != X(b) ? X(a) < X(b) : Y(a) < Y(b); });
    std::vector<int> front;
    double ybest = 1e300;
    for (int i : idx) if (Y(i) < ybest - 1e-12) { front.push_back(i); ybest = Y(i); }
    std::vector<double> c(n, 0.0);
    const double R = Evaluator::HV_REF;
    for (int m = 0; m < (int)front.size(); ++m) {
        const double xr = m + 1 < (int)front.size() ? X(front[m + 1]) : R;
        const double yl = m > 0 ? Y(front[m - 1]) : R;
        c[front[m]] = std::max(0.0, (xr - X(front[m])) * (yl - Y(front[m])));
    }
    return c;
}

inline int roulette(const std::vector<double>& c, Rng& g) {
    double tot = 0;
    for (double x : c) tot += x + 1e-9;           // +1e-9: dominated members keep a tiny probability
    double r = U01(g) * tot;
    for (int i = 0; i < (int)c.size(); ++i) { r -= c[i] + 1e-9; if (r <= 0) return i; }
    return (int)c.size() - 1;
}
}  // namespace kg

inline void run_kg_ga(Evaluator& E, Rng& g, const Params& prm) {
    using namespace kg;
    const Problem& P = E.P;
    const int NP = 100;
    const double pc = par(prm, "pc"), pm = par(prm, "pm");

    std::vector<Ind> pop(NP);
    for (auto& ind : pop) { ind.seq = random_feasible(P, g); ind.ev = E.evaluate(ind.seq); }
    std::vector<double> contrib = contributions(pop, P);
    for (;;) {   // terminates by BudgetExhausted
        const uint64_t e_before = E.evaluations();
        std::vector<Ind> comb = pop;               // parents keep their objective values
        while ((int)comb.size() < 2 * NP) {
            const int pa = roulette(contrib, g);
            int pb = Uint(g, 0, NP - 1);
            while (pb == pa) pb = Uint(g, 0, NP - 1);
            const bool cross = U01(g) < pc;          // one crossover decision per parent pair
            for (int c = 0; c < 2 && (int)comb.size() < 2 * NP; ++c) {
                Ind child;
                child.seq = cross ? ppx_child(pop[pa].seq, pop[pb].seq, P, g) : pop[c == 0 ? pa : pb].seq;
                if (U01(g) < pm) swap_mutation(child.seq, P, g);
                child.ev = E.evaluate(child.seq);                             // every offspring: one evaluation
                comb.push_back(std::move(child));
            }
        }
        const std::vector<double> cc = contributions(comb, P);
        std::vector<Ind> next;
        next.reserve(NP);
        for (int i = 0; i < NP; ++i) next.push_back(comb[roulette(cc, g)]);
        pop.swap(next);
        contrib = contributions(pop, P);
        if (E.evaluations() == e_before) throw std::runtime_error("no evaluation in an iteration");
    }
}

}  // namespace ssdsp
