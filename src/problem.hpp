// Instance data and model of the stopping-point selective disassembly sequence planning problem (Section 2).
//
// Notation (Table 6 of the manuscript):
//   N            number of parts; parts are indexed i = 1..N
//   tau_l        disassembly time of liaison l = rating x 4 s (Das and Naik 2002 ratings; Marconi et al. 2018)
//   C_l          labor cost of removing liaison l = tau_l * W_b / 3600                         Eq. (4)
//   o_i, r_i     virgin / recycled material carbon of part i = OCF_m * w_i, RCF_m * w_i       Eqs. (5)-(6)
//   phi_i        process carbon of part i = PCF_q * w_i                                       Eq. (7)
//   g_i          carbon recovery of part i under its 3R1T strategy                            Eq. (8)
//   g_i^bulk     carbon recovery of part i through bulk recycling = eta_m * (o_i - r_i)        Eq. (9)
//   f1max, f2max disassembly cost and carbon recovery of complete disassembly                 Eqs. (10)-(11)
//   sigma1/2     normalized cost and carbon objective (minimization form)                     Eqs. (12)-(13)
//   HV_pt        single-point HV with reference coordinate rho                                Eq. (14)
//   A            precedence matrix derived from the priority graph (transitive closure)
#pragma once
#include <array>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <sstream>

namespace ssdsp {

enum Material { IRON, STEEL, ALU, SUS, CU, MAT_N };   // iron, steel, aluminium, stainless steel, copper
enum Process { STAMP, EXTRUDE, MACHINE, PROC_N };     // stamping, extrusion, machining
enum Strategy { RU, RM, RC, T };                      // 3R1T: reuse, remanufacture, recycle, trash

// Carbon factors (kgCO2e/kg) and bulk-recycling efficiencies (Section 2.1-2.2)
inline std::array<double, MAT_N> OCF = {3.27, 3.27, 9.85, 3.562, 3.65};      // Table 2, virgin material (OCF_m)
inline std::array<double, MAT_N> RCF = {0.657, 0.678, 0.844, 2.831, 0.697};  // Table 2, recycled material (RCF_m)
inline std::array<double, PROC_N> PCF = {0.662, 0.912, 0.262};               // Table 3, manufacturing process (PCF_q)
inline std::array<double, MAT_N> ETA = {0.47, 0.47, 0.51, 0.50, 0.34};       // Table 7, bulk-recycling efficiency (eta_m)
constexpr double BASE_TIME = 4.0;   // base time (s) per rating unit: tau_l = rating x 4 s (Marconi et al. 2018)
constexpr double WAGE = 196.0;      // base wage W_b (NT$/h)

// ---- Section 4.8 (component damage): a damaged part is downgraded from RU/RM to RC, and each of its liaisons is
// removed by the destructive action that corresponds to its joining method (Table 20). Ratings: non-destructive
// joining methods from Table 1, destructive actions from Table 5 of Das and Naik (2002).
inline std::vector<int>& destr_set() { static std::vector<int> v; return v; }   // ids of the damaged parts
inline bool destroyed(int id) { for (int x : destr_set()) if (x == id) return true; return false; }
inline double destructive_rating(double r) {
    struct M { double a, b; };
    static const M m[] = {{2.2, 1.8},    // integrally threaded part -> saw cutting
                          {2.1, 1.8},    // nut & bolt               -> shearing cut
                          {1.8, 1.6},    // nail w/o head pin, interference fit -> hammering
                          {1.6, 1.4},    // cylindrical snapfit, seam -> levered pulling
                          {1.4, 1.2},    // screw/bolt standard head -> drilling
                          {1.3, 1.4},    // cantilever snapfit       -> levered pulling
                          {1.2, 1.0},    // socket and plug          -> axial pulling
                          {2.0, 2.0}};   // rivet/staple (already destructive) -> unchanged
    for (auto& x : m) if (std::abs(x.a - r) < 1e-9) return x.b;
    throw std::runtime_error("no destructive rating");
}
// ---- Sections 4.7-4.8 only: model data overridden through environment variables. Without them the program uses
// the manuscript values above. SSDSP_OCF / SSDSP_RCF / SSDSP_ETA: 5 values (iron, steel, aluminium, stainless
// steel, copper); SSDSP_PCF: 3 values (stamping, extrusion, machining); SSDSP_DESTR: ids of the damaged parts.
template <class A> inline void env_override(const char* key, A& arr) {
    if (const char* v = std::getenv(key)) { std::stringstream ss(v); std::string t; size_t i = 0;
        while (std::getline(ss, t, ',') && i < arr.size()) arr[i++] = std::stod(t); }
}
inline void apply_env_overrides() {
    static bool done = false; if (done) return; done = true;
    env_override("SSDSP_OCF", OCF); env_override("SSDSP_RCF", RCF); env_override("SSDSP_PCF", PCF);
    env_override("SSDSP_ETA", ETA);
    if (const char* v = std::getenv("SSDSP_DESTR")) { std::stringstream ss(v); std::string t;
        while (std::getline(ss, t, ',')) if (!t.empty()) destr_set().push_back(std::stoi(t)); }
}
struct Part { int id; Material m; double w; Process q; Strategy s; };   // i, m_i, w_i (kg), q_i, strategy(i)
struct Liaison { int a, b; double rating; };                            // liaison l between parts a and b
struct Instance {
    std::string name;
    std::vector<Part> parts;               // ids 1..N
    std::vector<Liaison> liaisons;         // set L
    std::vector<std::pair<int, int>> rule; // direct precedence constraints (a, b): a must be removed before b
};

// Ratings of the joining methods (Table 1 of Das and Naik 2002), used in Tables 5 and 10:
//   2.2 integrally threaded part   2.1 nut & bolt          2.0 rivet / staple (destructive)
//   1.8 interference fit / nail w/o head pin               1.6 seam / crimp, cylindrical snapfit
//   1.4 screw/bolt standard head   1.3 cantilever snapfit  1.2 socket and plug
// Stapler (Section 2.1): parts (id, material, weight in kg, process, 3R1T strategy) of Table 4, liaisons
// (part a, part b, rating) of Table 5, and the direct precedence constraints of the priority graph in Fig. 1(c).
inline Instance stapler() {
    return {"stapler",
            {{1, IRON, 0.0150, STAMP, RC},             // Steel cover
             {2, IRON, 0.0080, STAMP, RU},             // Bracket spring
             {3, IRON, 0.0025, STAMP, T},              // Rivet 1
             {4, STEEL, 0.0150, STAMP, RM},            // Steel top
             {5, STEEL, 0.0060, STAMP, RU},            // Pivot spring
             {6, ALU, 0.0040, EXTRUDE, RM},            // Slide foot
             {7, IRON, 0.0050, STAMP, RU},             // Staple spring
             {8, STEEL, 0.0200, EXTRUDE, RM},          // Bottom track
             {9, IRON, 0.0030, EXTRUDE, RU},           // Guide rod
             {10, IRON, 0.0025, STAMP, T},             // Rivet 2
             {11, STEEL, 0.0030, STAMP, RM},           // Impact plate
             {12, STEEL, 0.0950, STAMP, RC},           // Base
             {13, IRON, 0.0020, STAMP, RU},            // Spring
             {14, STEEL, 0.0025, STAMP, RC},           // Rivet bottom
             {15, STEEL, 0.0025, STAMP, RC},           // Fastener piece
             {16, IRON, 0.0025, STAMP, T},             // Rivet 3
             {17, IRON, 0.0025, STAMP, T},             // Rivet 4
             {18, STEEL, 0.0021, MACHINE, RC}},         // Pivot rod
            {{2, 3, 2.0},  {8, 10, 2.0},  {10, 11, 2.0}, {12, 16, 2.0}, {15, 16, 2.0}, {12, 17, 2.0}, {15, 17, 2.0},
             {12, 14, 2.0},{13, 14, 2.0}, {4, 5, 1.3},   {5, 6, 1.8},   {5, 7, 1.3},   {6, 8, 1.2},   {6, 9, 1.8},
             {7, 9, 1.8},  {8, 18, 1.8},  {11, 12, 1.8}, {12, 13, 1.3}, {12, 15, 1.6}, {1, 2, 1.8},   {1, 4, 1.8},
             {1, 18, 1.8}, {4, 18, 1.8},  {6, 7, 1.8},   {12, 18, 1.8}, {8, 12, 1.8}},
            {{3, 2},  {3, 1},  {18, 1}, {18, 4}, {18, 8}, {18, 12}, {10, 11}, {11, 12}, {14, 13}, {13, 12},
             {16, 15},{15, 12},{17, 15},{4, 5},  {4, 8},  {5, 6},   {5, 7},   {8, 6},   {8, 7},   {6, 9},
             {7, 9},  {2, 1},  {1, 4},  {6, 7},  {12, 8}, {10, 8},  {16, 12}, {17, 12}, {14, 12}}};
}

// Ceiling fan (Section 4.1): parts of Table 9, liaisons of Table 10 (all non-destructive), and the direct
// precedence constraints of the disassembly priority graph in Fig. 3(b).
inline Instance ceiling_fan() {
    return {"fan",
            {{1, IRON, 0.1610, MACHINE, RC},           // Balance hanger
             {2, IRON, 0.0260, STAMP, RC},             // Hemisphere hook
             {3, ALU, 0.0810, STAMP, RU},              // Decorative cover
             {4, IRON, 0.1270, EXTRUDE, RU},           // Hanging tube
             {5, SUS, 0.0020, MACHINE, T},             // Fork pin 1
             {6, SUS, 0.0020, MACHINE, T},             // Fork pin 2
             {7, ALU, 0.4070, STAMP, RU},              // Exterior cover 1
             {8, SUS, 0.0050, MACHINE, RC},            // Screws 1
             {9, SUS, 0.0050, MACHINE, RC},            // Screws 2
             {10, IRON, 0.3050, STAMP, RU},            // Motor shell 1
             {11, SUS, 0.0050, MACHINE, RC},           // Screws 3
             {12, SUS, 0.0050, MACHINE, RC},           // Screws 4
             {13, CU, 0.4070, EXTRUDE, RC},            // Stator 1
             {14, IRON, 0.3890, STAMP, RM},            // Stator 2
             {15, SUS, 0.0040, MACHINE, T},            // Nuts 1
             {16, SUS, 0.0040, MACHINE, T},            // Nuts 2
             {17, IRON, 0.3050, STAMP, RU},            // Motor shell 2
             {18, ALU, 0.2010, STAMP, RU},             // Exterior cover 2
             {19, SUS, 0.0050, MACHINE, RC},           // Screws 5
             {20, SUS, 0.0050, MACHINE, RC},           // Screws 6
             {21, SUS, 0.0050, MACHINE, RC},           // Screws 7
             {22, ALU, 0.1680, STAMP, RU},             // Fan blade 1
             {23, ALU, 0.1680, STAMP, RU},             // Fan blade 2
             {24, ALU, 0.1680, STAMP, RU},             // Fan blade 3
             {25, ALU, 0.0700, STAMP, RC},             // Switch cover 1
             {26, ALU, 0.0610, STAMP, RC}},             // Switch cover 2
            {{14, 13, 2.2}, {15, 7, 2.1},  {15, 18, 2.1}, {16, 7, 2.1},  {16, 18, 2.1}, {5, 1, 1.8},   {5, 2, 1.8},
             {5, 3, 1.8},   {5, 4, 1.8},   {6, 4, 1.8},   {10, 13, 1.6}, {17, 14, 1.6}, {8, 10, 1.4},  {8, 14, 1.4},
             {9, 10, 1.4},  {9, 14, 1.4},  {11, 7, 1.4},  {11, 18, 1.4}, {12, 7, 1.4},  {12, 18, 1.4}, {19, 22, 1.4},
             {19, 23, 1.4}, {19, 24, 1.4}, {20, 22, 1.4}, {20, 23, 1.4}, {20, 24, 1.4}, {21, 22, 1.4}, {21, 23, 1.4},
             {21, 24, 1.4}, {18, 17, 1.3}, {25, 17, 1.3}, {26, 25, 1.3}, {4, 7, 1.2}},
            {{5, 1},  {5, 2},  {5, 3},  {5, 4},  {6, 4},  {11, 7}, {11, 18}, {12, 7}, {12, 18}, {15, 7},
             {15, 18},{16, 7}, {16, 18},{19, 22},{19, 23},{19, 24},{20, 22}, {20, 23},{20, 24}, {21, 22},
             {21, 23},{21, 24},{26, 25},{4, 7},  {25, 17},{7, 8},  {7, 9},   {18, 17},{8, 10},  {8, 14},
             {9, 10}, {9, 14}, {17, 14},{10, 13},{14, 13}}};
}

inline Instance instance_by_name(const std::string& n) {
    if (n == "stapler") return stapler();
    if (n == "fan") return ceiling_fan();
    throw std::invalid_argument("unknown case: " + n + " (use stapler or fan)");
}

// Model quantities pre-computed once (Step 1 of Algorithm 2) and shared read-only by all runs.
struct Problem {
    std::string name;
    int N = 0;
    std::vector<double> g, gbulk;                       // g_i (Eq. 8) and g_i^bulk (Eq. 9), index 1..N
    std::vector<std::vector<std::pair<int, double>>> adj; // L(i): part i -> (v_l, C_l) for every liaison l of i
    std::vector<std::vector<char>> prec;                // precedence matrix A: prec[a][b] = 1 if a precedes b
    double f1max = 0, f2max = 0;                        // Eqs. (10), (11)
    double rho = 1.1;                                   // reference coordinate rho of HV_pt (Eq. 14; Section 4.5)

    explicit Problem(const Instance& in) : name(in.name), N((int)in.parts.size()) {
        apply_env_overrides();
        g.assign(N + 1, 0.0); gbulk.assign(N + 1, 0.0); adj.assign(N + 1, {});
        for (const auto& p : in.parts) {
            if (p.id < 1 || p.id > N) throw std::runtime_error("part ids must be 1..N");
            const double o = OCF[p.m] * p.w, r = RCF[p.m] * p.w, phi = PCF[p.q] * p.w;   // Eqs. (5)-(7)
            Strategy st = p.s;
            if (destroyed(p.id) && (st == RU || st == RM)) st = RC;                     // Section 4.8: damaged part RU/RM -> RC
            switch (st) {
                case RU: g[p.id] = o + phi; break;     // Eq. (8): reuse avoids material and process
                case RM: g[p.id] = o; break;           //          remanufacture avoids the virgin material
                case RC: g[p.id] = o - r; break;       //          recycle avoids o_i - r_i
                case T:  g[p.id] = 0.0; break;         //          trash: no individual recovery
            }
            gbulk[p.id] = ETA[p.m] * (o - r);                                           // Eq. (9)
            f2max += g[p.id];                                                           // Eq. (11)
        }
        for (const auto& l : in.liaisons) {
            const double rt = (destroyed(l.a) || destroyed(l.b)) ? destructive_rating(l.rating) : l.rating;
            const double c = rt * BASE_TIME * WAGE / 3600.0;                             // Eq. (4), tau_l = rt * 4 s
            adj[l.a].push_back({l.b, c}); adj[l.b].push_back({l.a, c});
            f1max += c;                                                                 // Eq. (10)
        }
        prec.assign(N + 1, std::vector<char>(N + 1, 0));
        for (const auto& r : in.rule) prec[r.first][r.second] = 1;
        for (int k = 1; k <= N; ++k)                                                    // Floyd-Warshall transitive closure -> A
            for (int i = 1; i <= N; ++i)
                if (prec[i][k])
                    for (int j = 1; j <= N; ++j)
                        if (prec[k][j]) prec[i][j] = 1;
    }
    double sigma1(double f1) const { return f1 / f1max; }            // Eq. (12)
    double sigma2(double f2) const { return 1.0 - f2 / f2max; }      // Eq. (13)
    double hv_point(double f1, double f2) const {                    // Eq. (14)
        return (rho - sigma1(f1)) * (rho - sigma2(f2));
    }
};

}  // namespace ssdsp
