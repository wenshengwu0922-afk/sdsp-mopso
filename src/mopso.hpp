// Proposed method (Section 3): archive-guided MOPSO combined with the stopping-point Pareto evaluation (rule B),
// Algorithm 2. Leader selection from the archive follows Coello Coello et al. (2004). Parameters (Table 11):
// inertia weight w (omega), cognitive and social coefficients c1, c2; swarm size P = 100.
//   - Section 3.1: each particle p carries a position vector X_p in [0,1]^N decoded into a feasible sequence S_p;
//   - Section 2.3 / Algorithm 1: one evaluation scans all N stopping points of S_p, every k in K^ND(S_p) enters
//     the external archive FA (Eq. 17), and the particle receives the objective vector of k_rep(S_p) (Eq. 15);
//   - Section 3.2: the global leader of every particle is a member of FA drawn uniformly at random, encoded by
//     Eq. (18); no swarm stratification and no mutation;
//   - Section 3.3: velocity and position updates of Eqs. (19)-(20); the personal best is replaced if the new
//     representative point dominates it, kept if it is dominated, and replaced with probability 0.5 if the two
//     are mutually non-dominated.
//
// Ablation switches (Section 4.4, Table 15; main.cpp), not part of the proposed method:
//   --leader archive (proposed) | swarm_nd (random non-dominated member of the swarm: "swarm leader") |
//            layer_hv (member of the swarm with the largest HV_pt: "gbest"; with Q = 10: "layered leader")
//   --pbest  dominance (proposed) | hv (replaced only if HV_pt increases: "HV_pt pbest")
//   optional parameters: pm (swap-mutation probability, "mutation" variant with pm = 0.2; default 0), Q (number of
//   layers for the swarm-based leaders, default 1), beta1 (layer criterion d = beta1*sigma1 + (1 - beta1)*sigma2,
//   default 0.5).
#pragma once
#include "common.hpp"

namespace ssdsp {

enum class LeaderRule { Archive, SwarmND, LayerHV };
enum class PbestRule { Dominance, HV };
struct MopsoOptions { LeaderRule leader = LeaderRule::Archive; PbestRule pbest = PbestRule::Dominance; };

inline void run_mopso(Evaluator& E, Rng& g, const Params& prm, const MopsoOptions& opt = {}) {
    const Problem& P = E.P;
    const int SW = 100, N = P.N, SWAPS = 2;   // SW = swarm size P
    const double w = par(prm, "w"), c1 = par(prm, "c1"), c2 = par(prm, "c2");
    const double pm = par(prm, "pm"), beta1 = par(prm, "beta1");
    const int Q = (int)par(prm, "Q");
    if (Q < 1 || Q > SW) throw std::invalid_argument("Q must be in [1, 100]");
    if (beta1 < 0.0 || beta1 > 1.0) throw std::invalid_argument("beta1 must be in [0, 1]");

    // X_p, V_p, pBest_p, S_p, representative evaluation of S_p and of pBest_p
    struct Particle { std::vector<double> x, v, pbest; std::vector<int> seq; Eval cur, pb; int layer = 0; };
    std::vector<Particle> sw(SW);

    // Step 1 (FA = {}, e = 0, A) is done by Evaluator / Problem.
    // Steps 2-4 (Algorithm 2, lines 2-6): X_p ~ U(0,1)^N, V_p = 0, DECODE-REPAIR, EVALUATE (FA filled inside
    // E.evaluate by Eq. 17), pBest_p <- X_p.
    for (auto& p : sw) {
        p.x.resize(N); p.v.assign(N, 0.0);
        for (auto& xi : p.x) xi = U01(g);
        p.seq = decode_repair(p.x, P);
        p.cur = E.evaluate(p.seq);
        p.pbest = p.x; p.pb = p.cur;
    }
    std::vector<int> order(SW);
    std::vector<double> dval(SW), Lbuf;
    std::vector<std::vector<double>> layer_leader(Q);
    for (;;) {   // generation t; Step 8: ends when e reaches E_max (BudgetExhausted thrown by E.evaluate)
        const uint64_t e_before = E.evaluations();

        if (opt.leader != LeaderRule::Archive) {   // ablation only: leaders taken from the swarm (copied first)
            for (int i = 0; i < SW; ++i)
                dval[i] = beta1 * P.sigma1(sw[i].cur.f1) + (1.0 - beta1) * P.sigma2(sw[i].cur.f2);
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return dval[a] < dval[b]; });
            for (int r = 0; r < SW; ++r) sw[order[r]].layer = std::min(r * Q / SW, Q - 1);
            std::vector<std::vector<int>> members(Q);
            for (int i = 0; i < SW; ++i) members[sw[i].layer].push_back(i);
            for (int q = 0; q < Q; ++q) {
                if (opt.leader == LeaderRule::SwarmND) {
                    std::vector<int> nd;
                    for (int i : members[q]) {
                        bool dom = false;
                        for (int j : members[q]) if (dom_eval(sw[j].cur, sw[i].cur)) { dom = true; break; }
                        if (!dom) nd.push_back(i);
                    }
                    layer_leader[q] = sw[nd[Uint(g, 0, (int)nd.size() - 1)]].x;
                } else {
                    int b = members[q][0];
                    for (int i : members[q]) if (sw[i].cur.hv > sw[b].cur.hv) b = i;
                    layer_leader[q] = sw[b].x;
                }
            }
        }
        for (int i = 0; i < SW; ++i) {
            Particle& p = sw[i];
            // Step 5 (line 9): leader_p <- X^a of a member (S^a, k^a) drawn uniformly from FA (Eq. 18)
            if (opt.leader == LeaderRule::Archive) {
                const auto& A = E.archive();
                Lbuf = encode(A[Uint(g, 0, (int)A.size() - 1)].seq);
            } else {
                Lbuf = layer_leader[p.layer];
            }
            // Step 6 (line 10): Eqs. (19)-(20), xi1, xi2 ~ U(0,1); X_p clamped to [0,1]^N
            for (int j = 0; j < N; ++j) {
                const double xi1 = U01(g), xi2 = U01(g);
                p.v[j] = w * p.v[j] + c1 * xi1 * (p.pbest[j] - p.x[j]) + c2 * xi2 * (Lbuf[j] - p.x[j]);
                p.x[j] = std::clamp(p.x[j] + p.v[j], 0.0, 1.0);
            }
            if (pm > 0.0 && U01(g) < pm)   // ablation only: swap-based mutation of two random pairs of entries
                for (int m = 0; m < SWAPS; ++m) {
                    const int a = Uint(g, 0, N - 1);
                    const int b = Uint(g, 0, N - 1);
                    std::swap(p.x[a], p.x[b]);
                }
            // Step 6-7 (lines 10-12): DECODE-REPAIR and EVALUATE (Algorithm 1); FA <- ND(FA U {(S_p,k)}) (Eq. 17)
            p.seq = decode_repair(p.x, P);
            p.cur = E.evaluate(p.seq);
            // Step 7 (line 13): personal best by Pareto dominance (Section 3.3)
            if (opt.pbest == PbestRule::Dominance) {
                const bool newDom = dom_eval(p.cur, p.pb), oldDom = dom_eval(p.pb, p.cur);
                if (newDom || (!oldDom && U01(g) < 0.5)) { p.pbest = p.x; p.pb = p.cur; }
            } else if (p.cur.hv > p.pb.hv) {               // ablation only
                p.pbest = p.x; p.pb = p.cur;
            }
        }
        if (E.evaluations() == e_before) throw std::runtime_error("no evaluation in an iteration");
    }
}

}  // namespace ssdsp
