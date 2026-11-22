// PSO baseline (Zhong et al. 2011), adapted to the stopping-point problem (Section 4.1): the same decoding
// (Section 3.1), the same stopping-point evaluation (Algorithm 1) and the same external archive FA as MOPSO, so that
// only the generation of new sequences differs. Fitness of a particle = HV_pt (Eq. 14) of its representative
// stopping point k_rep (Eq. 15); personal best replaced if the fitness increases; single global best = particle
// with the largest personal-best fitness. Parameters (Table 11): w, c1, c2; swarm size 100.
#pragma once
#include "common.hpp"

namespace ssdsp {

inline void run_pso(Evaluator& E, Rng& g, const Params& prm) {
    const Problem& P = E.P;
    const int SW = 100, N = P.N;
    const double w = par(prm, "w"), c1 = par(prm, "c1"), c2 = par(prm, "c2");

    std::vector<std::vector<double>> x(SW, std::vector<double>(N)), v(SW, std::vector<double>(N, 0.0)), pbest;
    std::vector<std::vector<int>> seq(SW);
    std::vector<double> fit(SW), pfit;
    for (int i = 0; i < SW; ++i) {
        for (auto& xi : x[i]) xi = U01(g);
        seq[i] = decode_repair(x[i], P);
        fit[i] = E.evaluate(seq[i]).hv;
    }
    pbest = x; pfit = fit;
    int gi = (int)(std::max_element(pfit.begin(), pfit.end()) - pfit.begin());
    for (;;) {   // terminates by BudgetExhausted
        const uint64_t e_before = E.evaluations();
        const std::vector<double> gbest = pbest[gi];
        for (int i = 0; i < SW; ++i) {
            for (int j = 0; j < N; ++j) {   // Eqs. (19)-(20) with the global best as leader
                const double r1 = U01(g), r2 = U01(g);
                v[i][j] = w * v[i][j] + c1 * r1 * (pbest[i][j] - x[i][j]) + c2 * r2 * (gbest[j] - x[i][j]);
                x[i][j] = std::clamp(x[i][j] + v[i][j], 0.0, 1.0);
            }
            seq[i] = decode_repair(x[i], P);
            fit[i] = E.evaluate(seq[i]).hv;
            if (fit[i] > pfit[i]) { pfit[i] = fit[i]; pbest[i] = x[i]; }
        }
        gi = (int)(std::max_element(pfit.begin(), pfit.end()) - pfit.begin());
        if (E.evaluations() == e_before) throw std::runtime_error("no evaluation in an iteration");
    }
}

}  // namespace ssdsp
