// ssdsp: experiment driver for the stopping-point selective disassembly study (rule B unless --rule A).
//
//   ssdsp --alg MOPSO|PSO|KG_GA|NSGA2|TBGA|RS --case stapler|fan --budget 100000 --seeds 1001-1030
//         [--grid "w=0,0.2;c1=1,2"] [--set "w=0,c1=5,c2=3"]
//         [--rule B|A] [--leader archive|swarm_nd|layer_hv] [--pbest dominance|hv] [--tag full] [--rho 1.1]
//         [--trace 100] [--front] [--threads 8] --out results/name
//   ssdsp --exact --case fan --out analysis/exact_fan      (exact front over all stopping states)
//
// Output files: <out>_runs.csv (one row per run), <out>_trace.csv (HV of FA at every budget/trace
// evaluations, if --trace > 0), <out>_front.csv (final archives, if --front). The column nfe_opt of
// <out>_runs.csv is the evaluation at which the archive first reached the exact optimum HV* (-1: never).
// One evaluation = one sequence evaluated at all N stopping points (Algorithm 1); the budget E_max (--budget) counts
// these evaluations (Section 4.1). Every run uses its own mt19937(seed); results do not depend on --threads.
//
// Algorithm names: MOPSO (proposed, Section 3), PSO (Zhong et al. 2011), KG_GA (Kongar and Gupta 2006), NSGA2 (Deb et
// al. 2002), TBGA (Li et al. 2020), RS (random search). --rule A, --leader and --pbest are the ablation switches of
// Section 4.4; --rho is the reference coordinate rho of Eq. (14) (Section 4.5).
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#ifndef SSDSP_NO_THREADS
#include <thread>
#endif

#include "kg_ga.hpp"
#include "mopso.hpp"
#include "nsga2.hpp"
#include "exact.hpp"
#include "pso.hpp"
#include "rs.hpp"
#include "tbga.hpp"

using namespace ssdsp;

static const std::map<std::string, std::vector<std::string>> PARAM_NAMES = {
    {"MOPSO", {"w", "c1", "c2", "pm", "Q", "beta1"}}, {"PSO", {"w", "c1", "c2"}}, {"KG_GA", {"pc", "pm"}},
    {"NSGA2", {"pc", "pm"}}, {"TBGA", {"team_ratio", "delta", "nm_ratio"}}, {"RS", {}}};
// optional parameters (ablation switches) and their defaults; all other parameters are required
static const std::map<std::string, Params> PARAM_DEFAULTS = {{"MOPSO", {{"pm", 0.0}, {"Q", 1.0}, {"beta1", 0.5}}}};

static std::vector<std::string> split(const std::string& s, char d) {
    std::vector<std::string> out; std::string cur; std::stringstream ss(s);
    while (std::getline(ss, cur, d)) if (!cur.empty()) out.push_back(cur);
    return out;
}

struct Task { Params prm; unsigned seed; };
struct Result { Params prm; unsigned seed; double hv; size_t front; uint64_t nfe; long long nfe_opt; double sec;
                std::vector<std::pair<uint64_t, double>> trace; std::vector<ArchivePoint> fa; };

int main(int argc, char** argv) {
    std::map<std::string, std::string> a = {{"rule", "B"}, {"leader", "archive"}, {"pbest", "dominance"}, {"tag", "-"},
                                            {"trace", "0"}, {"threads", "1"}, {"rho", "1.1"}};
    bool want_front = false, want_exact = false;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "--front") { want_front = true; continue; }
        if (k == "--exact") { want_exact = true; continue; }
        if (k.rfind("--", 0) != 0 || i + 1 >= argc) { std::cerr << "bad argument: " << k << "\n"; return 2; }
        a[k.substr(2)] = argv[++i];
    }
    if (want_exact) {
        if (!a.count("case") || !a.count("out")) { std::cerr << "--exact needs --case and --out\n"; return 2; }
        Problem prob(instance_by_name(a["case"]));
        size_t ns = 0;
        const auto F = exact_front(prob, &ns);
        std::vector<ArchivePoint> fa;
        for (const auto& p : F) fa.push_back({p.f1, p.f2, p.k, {}});
        Evaluator E(prob, 1, Rule::B, 0);
        std::ofstream fo(a["out"] + ".csv");
        if (!fo) { std::cerr << "cannot write " << a["out"] << ".csv\n"; return 3; }
        fo << std::setprecision(15) << "case,N,f1max,f2max,states,cost,carbon,stopping_point\n";
        for (const auto& p : F)
            fo << prob.name << "," << prob.N << "," << prob.f1max << "," << prob.f2max << "," << ns << ","
               << p.f1 << "," << p.f2 << "," << p.k << "\n";
        std::cout << prob.name << ": " << ns << " stopping states, exact front " << F.size()
                  << " points, HV = " << std::setprecision(6) << std::fixed << E.hv_of(fa) << " -> " << a["out"] << ".csv\n";
        return 0;
    }
    for (const char* req : {"alg", "case", "budget", "seeds", "out"})
        if (!a.count(req)) { std::cerr << "missing --" << req << "\n"; return 2; }
    const std::string alg = a["alg"];
    if (!PARAM_NAMES.count(alg)) { std::cerr << "unknown --alg (MOPSO, PSO, KG_GA, NSGA2, TBGA, RS)\n"; return 2; }
    const auto& names = PARAM_NAMES.at(alg);

    Problem prob(instance_by_name(a["case"]));
    prob.rho = std::stod(a["rho"]);
    const uint64_t budget = std::stoull(a["budget"]);
    if (a["rule"] != "A" && a["rule"] != "B") { std::cerr << "--rule must be A or B\n"; return 2; }
    const Rule rule = a["rule"] == "A" ? Rule::A : Rule::B;
    MopsoOptions mopt;
    if (a["leader"] == "archive") mopt.leader = LeaderRule::Archive;
    else if (a["leader"] == "swarm_nd") mopt.leader = LeaderRule::SwarmND;
    else if (a["leader"] == "layer_hv") mopt.leader = LeaderRule::LayerHV;
    else { std::cerr << "--leader must be archive, swarm_nd or layer_hv\n"; return 2; }
    if (a["pbest"] == "dominance") mopt.pbest = PbestRule::Dominance;
    else if (a["pbest"] == "hv") mopt.pbest = PbestRule::HV;
    else { std::cerr << "--pbest must be dominance or hv\n"; return 2; }
    const int ckpt = std::stoi(a["trace"]);

    // seeds: "1001-1030" or "1001,1005,1010"
    std::vector<unsigned> seeds;
    for (auto& part : split(a["seeds"], ',')) {
        auto dash = part.find('-');
        if (dash == std::string::npos) seeds.push_back((unsigned)std::stoul(part));
        else for (unsigned s = std::stoul(part.substr(0, dash)); s <= std::stoul(part.substr(dash + 1)); ++s) seeds.push_back(s);
    }
    // parameter sets: --set gives the base values, --grid gives the factors to cross (full factorial)
    Params base;
    if (PARAM_DEFAULTS.count(alg)) base = PARAM_DEFAULTS.at(alg);
    if (a.count("set")) for (auto& kv : split(a["set"], ',')) { auto e = kv.find('='); base[kv.substr(0, e)] = std::stod(kv.substr(e + 1)); }
    std::vector<Params> sets = {base};
    if (a.count("grid"))
        for (auto& factor : split(a["grid"], ';')) {
            auto e = factor.find('=');
            const std::string key = factor.substr(0, e);
            std::vector<Params> nx;
            for (auto& p : sets) for (auto& v : split(factor.substr(e + 1), ',')) { Params q = p; q[key] = std::stod(v); nx.push_back(q); }
            sets.swap(nx);
        }
    for (auto& p : sets) {
        for (auto& n : names) if (!p.count(n)) { std::cerr << "parameter " << n << " not given (--set or --grid)\n"; return 2; }
        for (auto& kv : p)
            if (std::find(names.begin(), names.end(), kv.first) == names.end()) {
                std::cerr << "parameter " << kv.first << " is not used by " << alg << "\n"; return 2;
            }
    }

    // HV* of the exact Pareto front (enumeration of all feasible stopping states, Table 8) as target for NFE*
    double hv_star = -1.0;
    if (prob.N <= 30) {
        const auto F = exact_front(prob);
        std::vector<ArchivePoint> fa;
        for (const auto& p : F) fa.push_back({p.f1, p.f2, p.k, {}});
        hv_star = Evaluator(prob, 1, Rule::B, 0).hv_of(fa);
    }
    std::vector<Task> tasks;
    for (auto& p : sets) for (unsigned s : seeds) tasks.push_back({p, s});
    std::vector<Result> res(tasks.size());
    std::atomic<size_t> next{0};
    std::atomic<bool> failed{false};
    std::mutex io;
    const auto t_start = std::chrono::steady_clock::now();

    auto worker = [&]() {
        for (size_t t; (t = next++) < tasks.size();) {
            const Task& tk = tasks[t];
            Evaluator E(prob, budget, rule, ckpt);
            E.set_target(hv_star);
            Rng g(tk.seed);
            const auto t0 = std::chrono::steady_clock::now();
            try {
                if (alg == "MOPSO") run_mopso(E, g, tk.prm, mopt);
                else if (alg == "PSO") run_pso(E, g, tk.prm);
                else if (alg == "KG_GA") run_kg_ga(E, g, tk.prm);
                else if (alg == "NSGA2") run_nsga2(E, g, tk.prm);
                else if (alg == "TBGA") run_tbga(E, g, tk.prm);
                else run_rs(E, g, tk.prm);
            } catch (const BudgetExhausted&) {
            } catch (const std::exception& ex) {
                std::lock_guard<std::mutex> lk(io);
                std::cerr << "\nERROR " << alg << " seed " << tk.seed << ": " << ex.what() << "\n";
                failed = true;
            }
            if (!failed && E.evaluations() != budget) {   // every run must use exactly E_max evaluations
                std::lock_guard<std::mutex> lk(io);
                std::cerr << "\nERROR " << alg << " seed " << tk.seed << ": used " << E.evaluations() << " of "
                          << budget << " evaluations\n";
                failed = true;
            }
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            res[t] = {tk.prm, tk.seed, E.hv(), E.archive().size(), E.evaluations(),
                      E.nfe_to_target() ? (long long)E.nfe_to_target() : -1LL, sec, E.trace(),
                      want_front ? E.archive() : std::vector<ArchivePoint>{}};
            if ((t + 1) % 50 == 0 || t + 1 == tasks.size()) {
                std::lock_guard<std::mutex> lk(io);
                const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
                std::cerr << "\r" << alg << " " << a["case"] << " " << a["tag"] << ": " << (t + 1) << "/" << tasks.size()
                          << " runs, " << std::fixed << std::setprecision(0) << el << " s" << std::flush;
            }
        }
    };
#ifdef SSDSP_NO_THREADS   // for MinGW builds without std::thread support (win32 thread model)
    worker();
#else
    int nth = std::max(1, std::stoi(a["threads"]));
    std::vector<std::thread> pool;
    for (int i = 0; i < nth; ++i) pool.emplace_back(worker);
    for (auto& th : pool) th.join();
#endif
    std::cerr << "\n";
    if (failed) { std::cerr << "run(s) failed; no output written\n"; return 4; }

    // ---- output: every file is written under a temporary name and renamed when complete; <out>_runs.csv is
    // renamed last, so its existence means that the whole call finished (used by the scripts to resume).
    const std::string out = a["out"];
    std::string head = "alg,case,N,budget,rule,leader,pbest,tag,rho";
    for (auto& n : names) head += "," + n;
    auto prefix = [&](const Result& r) {
        std::ostringstream s; s << std::setprecision(10);
        s << alg << "," << prob.name << "," << prob.N << "," << budget << "," << a["rule"] << "," << a["leader"] << ","
          << a["pbest"] << "," << a["tag"] << "," << prob.rho;
        for (auto& n : names) s << "," << r.prm.at(n);
        s << "," << r.seed;
        return s.str();
    };
    std::vector<std::string> done;
    auto write_file = [&](const std::string& suffix, auto&& body) {
        const std::string fin = out + suffix, tmp = fin + ".tmp";
        {
            std::ofstream fo(tmp);
            if (!fo) { std::cerr << "cannot write " << tmp << " (does the folder exist?)\n"; return false; }
            body(fo);
            if (!fo) { std::cerr << "write error on " << tmp << "\n"; return false; }
        }
        std::remove(fin.c_str());
        if (std::rename(tmp.c_str(), fin.c_str()) != 0) { std::cerr << "cannot rename " << tmp << "\n"; return false; }
        return true;
    };
    std::remove((out + "_runs.csv").c_str());
    if (ckpt > 0 && !write_file("_trace.csv", [&](std::ofstream& ft) {
            ft << head << ",seed,nfe,HV\n" << std::setprecision(10);
            for (auto& r : res) for (auto& [e, h] : r.trace) ft << prefix(r) << "," << e << "," << h << "\n";
        })) return 3;
    if (want_front && !write_file("_front.csv", [&](std::ofstream& ff) {
            ff << head << ",seed,cost,carbon,stopping_point,sequence\n" << std::setprecision(15);
            for (auto& r : res)
                for (auto& p : r.fa) {
                    ff << prefix(r) << "," << p.f1 << "," << p.f2 << "," << p.k << ",\"";
                    for (size_t i = 0; i < p.seq.size(); ++i) ff << (i ? " " : "") << p.seq[i];
                    ff << "\"\n";
                }
        })) return 3;
    if (!write_file("_runs.csv", [&](std::ofstream& fr) {
            fr << head << ",seed,HV,front_size,nfe,nfe_opt,sec\n" << std::setprecision(10);
            for (auto& r : res)
                fr << prefix(r) << "," << r.hv << "," << r.front << "," << r.nfe << "," << r.nfe_opt << "," << r.sec << "\n";
        })) return 3;
    double m = 0; for (auto& r : res) m += r.hv;
    std::cout << alg << " " << prob.name << " budget=" << budget << " sets=" << sets.size() << " runs=" << res.size()
              << " meanHV=" << std::setprecision(6) << m / res.size() << " -> " << out << "_runs.csv\n";
    return 0;
}
