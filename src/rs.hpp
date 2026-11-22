// Random search (Section 4.1): uniformly shuffled permutations made feasible by precedence repair, each evaluated
// once by Algorithm 1 (rule B archive). No parameters. Used only as a reference level for the difficulty of the
// instances.
#pragma once
#include "common.hpp"

namespace ssdsp {

inline void run_rs(Evaluator& E, Rng& g, const Params&) {
    for (;;) E.evaluate(random_feasible(E.P, g));   // terminates by BudgetExhausted; one evaluation per sample
}

}  // namespace ssdsp
