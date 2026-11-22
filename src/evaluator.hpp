// Stopping-point Pareto evaluation (Section 2.3, Algorithm 1), external archive FA (Eq. 17), evaluation budget
// E_max with counter e, and the performance metrics HV(FA), HV* and NFE* (Section 4.1).
// One sequence evaluation = one call of evaluate() = one call of Algorithm 1: all N stopping points of the
// sequence are evaluated in a single incremental pass.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "problem.hpp"

namespace ssdsp {

struct BudgetExhausted {};

struct Eval {          // representative stopping point k_rep(S) (Eq. 15) and its objective vector and HV_pt
    int k_rep = 0;
    double f1 = 0, f2 = 0, hv = 0;
};

struct ArchivePoint {  // non-dominated (S, k) pair: f1(S,k), f2(S,k), k, S
    double f1, f2;
    int k;
    std::vector<int> seq;
};

// Rule B (proposed, stopping-point Pareto evaluation): every k in K^ND(S) enters FA (Eqs. 16-17).
// Rule A (ablation, Section 4.4): a single stopping point per sequence, only k_rep(S) enters FA.
enum class Rule { B, A };

// Pareto dominance (S,k') < (S,k): f1' <= f1 and f2' >= f2, at least one strict (min f1, max f2; Eq. 1)
inline bool dominates(double a1, double a2, double b1, double b2) {
    constexpr double eps = 1e-12;
    return a1 <= b1 + eps && a2 >= b2 - eps && (a1 < b1 - eps || a2 > b2 + eps);
}

class Evaluator {
public:
    static constexpr double HV_REF = 1.1;   // reference point (1.1, 1.1) of the HV(FA) metric (fixed; Section 4.1)

    Evaluator(const Problem& p, uint64_t budget, Rule rule, int checkpoints)
        : P(p), budget_(budget), rule_(rule) {
        step_ = checkpoints > 0 ? std::max<uint64_t>(1, budget / (uint64_t)checkpoints) : 0;
        f1_.resize(P.N + 1); f2_.resize(P.N + 1);
    }

    const Problem& P;
    uint64_t evaluations() const { return e_; }
    uint64_t budget() const { return budget_; }
    const std::vector<ArchivePoint>& archive() const { return fa_; }
    const std::vector<std::pair<uint64_t, double>>& trace() const { return trace_; }
    // Target HV* (HV of the exact Pareto front, Table 8): nfe_to_target() is NFE*, the evaluation at which HV(FA)
    // first equals HV* (0 if never reached; written as -1 in the output).
    void set_target(double t) { target_ = t; }
    uint64_t nfe_to_target() const { return hit_; }

    // Algorithm 1: EVALUATE(S). Throws BudgetExhausted when e has already reached E_max (Step 8 of Algorithm 2).
    Eval evaluate(const std::vector<int>& S) {
        if (e_ >= budget_) throw BudgetExhausted{};
        ++e_;                                                                   // line 11: e <- e + 1
        const int N = P.N;
        removed_.assign(N + 1, 0);
        double f1 = 0, G = 0, B = 0;          // line 1: R = {} (removed_), f1 = 0, G = 0 (3R1T), B = sum g_i^bulk
        for (int i = 1; i <= N; ++i) B += P.gbulk[i];
        Eval best; best.hv = -1e300;
        for (int k = 1; k <= N; ++k) {
            const int i = S[k - 1];                                             // line 3: i <- s_k
            for (const auto& [v, c] : P.adj[i]) if (!removed_[v]) f1 += c;     // line 4: Eqs. (2), (4)
            removed_[i] = 1;
            G += P.g[i]; B -= P.gbulk[i];                                       // line 5: Eqs. (8), (9)
            f1_[k] = f1; f2_[k] = G + B;                                        // line 6: Eq. (3)
            const double hv = P.hv_point(f1_[k], f2_[k]);                       // line 7: Eqs. (12)-(14)
            if (hv > best.hv) best = {k, f1_[k], f2_[k], hv};                   // line 9, Eq. (15): first maximum
        }
        bool changed = false;
        if (rule_ == Rule::A) {
            changed = insert({best.f1, best.f2, best.k_rep, S});
        } else {                                                                // line 10, Eqs. (16)-(17)
            for (int k = 1; k <= N; ++k) {
                bool dom = false;
                for (int j = 1; j <= N && !dom; ++j)
                    if (j != k && dominates(f1_[j], f2_[j], f1_[k], f2_[k])) dom = true;
                if (!dom) changed |= insert({f1_[k], f2_[k], k, S});
            }
        }
        if (changed && target_ > 0 && !hit_ && hv() >= target_ - 1e-9) hit_ = e_;
        if (step_ && (e_ % step_ == 0 || e_ == budget_)) trace_.push_back({e_, hv()});
        return best;
    }

    // HV(FA): hypervolume of the archive in the normalized space (sigma1, sigma2), reference point (1.1, 1.1).
    double hv() const { return hv_of(fa_); }

    double hv_of(const std::vector<ArchivePoint>& F) const {
        std::vector<std::pair<double, double>> pts;
        pts.reserve(F.size());
        for (const auto& a : F) {
            const double x = a.f1 / P.f1max, y = 1.0 - a.f2 / P.f2max;
            if (x < HV_REF && y < HV_REF) pts.push_back({x, y});
        }
        std::sort(pts.begin(), pts.end());
        double area = 0, ybest = HV_REF;
        for (const auto& [x, y] : pts)
            if (y < ybest - 1e-12) { area += (HV_REF - x) * (ybest - y); ybest = y; }
        return area;
    }

    // Exclusive HV contribution of a candidate (f1, f2) w.r.t. the current archive (used only by the MHM/FBOO
    // local search of the TBGA baseline).
    // Archive points stemming from the candidate's own sequence are ignored, so that the contribution is
    // measured against the archive without the candidate (as in the original TBGA adaptation).
    double exclusive_contribution(double f1, double f2, const std::vector<int>& own_seq) const {
        const double x = f1 / P.f1max, y = 1.0 - f2 / P.f2max;
        double right_x = HV_REF, upper_y = HV_REF;
        for (const auto& a : fa_) {
            if (a.seq == own_seq) continue;
            const double ax = a.f1 / P.f1max, ay = 1.0 - a.f2 / P.f2max;
            if (ax <= x + 1e-12 && ay <= y + 1e-12) return 0.0;      // candidate dominated (or duplicate)
            if (ax > x + 1e-12 && ay < y - 1e-12) right_x = std::min(right_x, ax);   // right neighbour
            else if (ax < x - 1e-12 && ay > y + 1e-12) upper_y = std::min(upper_y, ay); // left neighbour
            // otherwise a is dominated by the candidate and leaves the front
        }
        return std::max(0.0, (right_x - x) * (upper_y - y));
    }

private:
    // FA <- ND(FA U {(S,k)}) (Eq. 17): add (S,k) unless it is dominated by (or duplicates) a member, and remove the
    // members it dominates. Returns true if FA changed.
    bool insert(ArchivePoint p) {
        for (const auto& a : fa_) {
            if (std::fabs(a.f1 - p.f1) < 1e-12 && std::fabs(a.f2 - p.f2) < 1e-12) return false;
            if (dominates(a.f1, a.f2, p.f1, p.f2)) return false;
        }
        fa_.erase(std::remove_if(fa_.begin(), fa_.end(),
                                 [&](const ArchivePoint& a) { return dominates(p.f1, p.f2, a.f1, a.f2); }),
                  fa_.end());
        fa_.push_back(std::move(p));
        return true;
    }

    uint64_t budget_, e_ = 0, step_ = 0, hit_ = 0;
    double target_ = -1.0;
    Rule rule_;
    std::vector<ArchivePoint> fa_;
    std::vector<std::pair<uint64_t, double>> trace_;
    std::vector<char> removed_;
    std::vector<double> f1_, f2_;
};

}  // namespace ssdsp
