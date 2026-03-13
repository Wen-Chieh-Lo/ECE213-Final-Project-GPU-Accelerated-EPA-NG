#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>
#include <libpll/pll.h>

#include "precision.hpp"
#include "root_likelihood.cuh"
#include "tree.hpp"
#include "tree_placement.cuh"
#include "parse_file.hpp"
#include "seq_preproc.hpp"
#include "../mlipper_util.h"

namespace {

// ----- File and path helpers -----

std::string read_file_to_string(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) throw std::runtime_error("Cannot open file: " + path);
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

std::string resolve_path(const std::filesystem::path& base, const std::string& p) {
    namespace fs = std::filesystem;
    fs::path candidate(p);
    if (candidate.empty()) return {};
    if (candidate.is_absolute()) return candidate.string();
    return (base / candidate).string();
}

// ----- Model input normalization helpers -----

void normalize_vector(std::vector<double>& vec) {
    double sum = 0.0;
    for (double v : vec) sum += v;
    if (sum <= 0.0) return;
    for (double& v : vec) v /= sum;
}

std::vector<double> ensure_normalized_pi(std::vector<double> pi, int states) {
    if ((int)pi.size() != states) pi.assign(states, 1.0 / states);
    normalize_vector(pi);
    return pi;
}

// ----- Environment helpers -----

void set_int_env_if_specified(const char* name, int value) {
    if (value < 0) return;
    setenv(name, std::to_string(value).c_str(), 1);
}

void set_double_env_if_specified(const char* name, double value) {
    if (value < 0.0) return;
    setenv(name, std::to_string(value).c_str(), 1);
}

bool env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value && value[0] && std::string(value) != "0";
}

} // namespace

int main(int argc, char** argv) {
    const auto start_gpu = std::chrono::steady_clock::now();
    cudaEvent_t start, stop;
    float gpu_ms_kernel = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    CLI::App app{"MLIPPER"};
    app.get_formatter()->column_width(40);

    // Single config object filled directly by CLI11 options.
    parse::RunConfig config;

    // ---- Input (files/tree) ----
    std::string tree_newick;
    std::string jplace_out;

    auto* opt_tree_alignment = app.add_option("--tree-alignment", config.files.tree_alignment, "Reference alignment (tree MSA)")
                                  ->group("Input")
                                  ->check(CLI::ExistingFile);
    auto* opt_query_alignment = app.add_option("--query-alignment", config.files.query_alignment,
                                               "Query alignment for placement (optional; defaults to --tree-alignment)")
                                   ->group("Input")
                                   ->check(CLI::ExistingFile);
    auto* opt_tree_file = app.add_option("--tree", config.files.tree, "Reference tree topology (Newick file)")
                              ->group("Input")
                              ->check(CLI::ExistingFile);
    auto* opt_tree_newick = app.add_option("--tree-newick", tree_newick, "Reference tree topology (Newick string)")
                                ->group("Input");
    app.add_option("--jplace-out", jplace_out, "Optional output path for a single-best-placement jplace file")
        ->group("Output");
    opt_tree_file->excludes(opt_tree_newick);
    opt_tree_newick->excludes(opt_tree_file);

    // ---- Model ----
    // Defaults match the previous `config.yaml` defaults (pre-CLI refactor).
    config.model.states = 4;
    config.model.subst_model = "GTR";
    config.model.ncat = 4;
    config.model.alpha = 0.3;
    config.model.pinv = 0.0;
    config.model.rates = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    config.model.per_rate_scaling = true;
    bool no_per_rate_scaling = false;
    int full_opt_passes = -1;
    int refine_global_passes = -1;
    int refine_extra_passes = -1;
    int refine_detect_topk = -1;
    int refine_topk = -1;
    double refine_gap_top2 = -1.0;
    double refine_gap_top5 = -1.0;
    double refine_converged_loglk_eps = -1.0;
    double refine_converged_length_eps = -1.0;

    app.add_option("--states", config.model.states, "Number of states")->group("Model");
    app.add_option("--subst-model", config.model.subst_model, "Substitution model")->group("Model");
    app.add_option("--ncat", config.model.ncat, "Number of rate categories")->group("Model");
    app.add_option("--alpha", config.model.alpha, "Gamma shape alpha")->group("Model");
    app.add_option("--pinv", config.model.pinv, "Proportion of invariant sites")->group("Model");
    app.add_option("--freqs", config.model.freqs, "Equilibrium freqs (list)")->group("Model")->delimiter(',');
    app.add_option("--rates", config.model.rates, "GTR rates rAC,rAG,rAT,rCG,rCT,rGT (list)")->group("Model")->delimiter(',');
    app.add_option("--rate-weights", config.model.rate_weights, "Rate category weights (list)")->group("Model")->delimiter(',');
    app.add_flag("--no-per-rate-scaling", no_per_rate_scaling, "Disable per-rate scaling")->group("Model");

    app.add_option("--full-opt-passes", full_opt_passes,
                   "Full optimization passes when refine is disabled")
        ->group("Placement");
    app.add_option("--refine-global-passes", refine_global_passes,
                   "Full-op passes before refine selection")
        ->group("Placement");
    app.add_option("--refine-extra-passes", refine_extra_passes,
                   "Maximum extra passes after refine top-K selection")
        ->group("Placement");
    app.add_option("--refine-detect-topk", refine_detect_topk,
                   "Top-K candidates examined when deciding whether to refine")
        ->group("Placement");
    app.add_option("--refine-topk", refine_topk,
                   "Top-K candidates retained for refine passes")
        ->group("Placement");
    app.add_option("--refine-gap-top2", refine_gap_top2,
                   "Ambiguity threshold for top1-top2 log-likelihood gap")
        ->group("Placement");
    app.add_option("--refine-gap-top5", refine_gap_top5,
                   "Ambiguity threshold for top1-top5 log-likelihood gap")
        ->group("Placement");
    app.add_option("--refine-converged-loglk-eps", refine_converged_loglk_eps,
                   "Per-op refine early-stop threshold for log-likelihood change")
        ->group("Placement");
    app.add_option("--refine-converged-length-eps", refine_converged_length_eps,
                   "Per-op refine early-stop threshold for branch-length change")
        ->group("Placement");

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }

    namespace fs = std::filesystem;

    parse::RunInputs inputs;
    fs::path config_base = fs::current_path();
    try {
        if (no_per_rate_scaling) config.model.per_rate_scaling = false;
        if (config.files.tree_alignment.empty()) throw CLI::RequiredError("--tree-alignment");
        if (tree_newick.empty() && config.files.tree.empty()) throw CLI::RequiredError("one of [--tree, --tree-newick]");

        if (config.model.states <= 0) throw CLI::ValidationError("--states", "must be positive");
        if (config.model.ncat <= 0) throw CLI::ValidationError("--ncat", "must be positive");

        if (!config.model.freqs.empty() && (int)config.model.freqs.size() != config.model.states) {
            throw CLI::ValidationError("--freqs",
                                       "must have exactly " + std::to_string(config.model.states) + " values (states)");
        }
        if (!config.model.rates.empty() && config.model.rates.size() != 6) {
            throw CLI::ValidationError("--rates", "must have exactly 6 values for 4-state GTR");
        }
        if (!config.model.rate_weights.empty() && (int)config.model.rate_weights.size() != config.model.ncat) {
            throw CLI::ValidationError("--rate-weights",
                                       "must have exactly " + std::to_string(config.model.ncat) + " values (ncat)");
        }

        parse::Alignment tree_alignment;
        try {
            tree_alignment = parse::read_alignment_file(resolve_path(config_base, config.files.tree_alignment));
        } catch (const std::exception& e) {
            throw CLI::ValidationError("--tree-alignment", e.what());
        }

        parse::Alignment query_alignment;
        try {
            query_alignment = parse::read_alignment_file(resolve_path(config_base, config.files.query_alignment));
        } catch (const std::exception& e) {
            throw CLI::ValidationError("--query-alignment", e.what());
        }

        std::string tree_text;
        if (tree_newick.empty()) {
            try {
                tree_text = read_file_to_string(resolve_path(config_base, config.files.tree));
            } catch (const std::exception& e) {
                throw CLI::ValidationError("--tree", e.what());
            }
        } else {
            tree_text = tree_newick;
        }
        tree_text = parse::normalize_newick(tree_text);

        inputs = parse::RunInputs{std::move(config), std::move(tree_alignment), std::move(tree_text)};

        if (inputs.tree_alignment.names.empty())
            throw CLI::ValidationError("--tree-alignment", "contains no sequences");
        if (inputs.tree_alignment.sites == 0)
            throw CLI::ValidationError("--tree-alignment", "contains zero sites");

        if (query_alignment.names.empty())
            throw CLI::ValidationError("--query-alignment", "contains no sequences");
        if (query_alignment.sites == 0)
            throw CLI::ValidationError("--query-alignment", "contains zero sites");
        if (query_alignment.sites != inputs.tree_alignment.sites) {
            throw CLI::ValidationError("--query-alignment",
                                       "sites mismatch with --tree-alignment (" +
                                           std::to_string(query_alignment.sites) + " vs " +
                                           std::to_string(inputs.tree_alignment.sites) + ")");
        }
    } catch (const CLI::Error& e) {
        return app.exit(e);
    }

    const auto& alignment = inputs.tree_alignment;

    const auto& msa_names = alignment.names;
    std::vector<std::string> rows = alignment.sequences;
    size_t sites = alignment.sites;
    const std::string& newick = inputs.tree;
    const auto& model = inputs.config.model;
    int states = model.states;
    int rate_cats = model.ncat;
    bool per_rate_scaling = model.per_rate_scaling;
    std::vector<unsigned> pattern_weights(sites, 1u);

    std::vector<double> pi = ensure_normalized_pi(model.freqs, states);
    std::vector<double> rate_weights = build_mixture_weights(model, rate_cats);
    std::vector<double> rate_multipliers = build_gamma_rate_categories(model.alpha, rate_cats);
    std::vector<double> Q = build_gtr_q_matrix(states, model, pi);

    const std::string& query_alignment_cfg = inputs.config.files.query_alignment;
    std::vector<NewPlacementQuery> placement_queries =
        build_placement_query(resolve_path(config_base, query_alignment_cfg));
    if (!env_flag_enabled("MLIPPER_DISABLE_REPETITIVE_COLUMNS")) {
        remove_repetitive_columns(rows, placement_queries, pattern_weights, sites);
        if (sites == 0) {
            throw std::runtime_error("All columns were removed after repetitive-column compression.");
        }
    }
    const bool disable_pattern_weights = env_flag_enabled("MLIPPER_DISABLE_PATTERN_WEIGHTS");
    const std::vector<unsigned> no_pattern_weights;
    const std::vector<unsigned>& pattern_weights_arg =
        disable_pattern_weights ? no_pattern_weights : pattern_weights;

    set_int_env_if_specified("MLIPPER_FULL_OPT_PASSES", full_opt_passes);
    set_int_env_if_specified("MLIPPER_REFINE_GLOBAL_PASSES", refine_global_passes);
    set_int_env_if_specified("MLIPPER_REFINE_EXTRA_PASSES", refine_extra_passes);
    set_int_env_if_specified("MLIPPER_REFINE_DETECT_TOPK", refine_detect_topk);
    set_int_env_if_specified("MLIPPER_REFINE_TOPK", refine_topk);
    set_double_env_if_specified("MLIPPER_REFINE_GAP_TOP2", refine_gap_top2);
    set_double_env_if_specified("MLIPPER_REFINE_GAP_TOP5", refine_gap_top5);
    set_double_env_if_specified("MLIPPER_REFINE_CONVERGED_LOGLK_EPS", refine_converged_loglk_eps);
    set_double_env_if_specified("MLIPPER_REFINE_CONVERGED_LENGTH_EPS", refine_converged_length_eps);

    printf("Precision mode: %s\n", FP_MODE_NAME);
    // const auto start_gpu = std::chrono::steady_clock::now();
    auto res = BuildAllToGPU(
        msa_names,
        rows,
        newick,
        Q,
        pi,
        rate_multipliers,
        rate_weights,
        pattern_weights_arg,
        sites,
        states,
        rate_cats,
        per_rate_scaling,
        placement_queries);
    if (res.tree.nodes.empty() || res.dev.N == 0) {
        throw std::runtime_error("BuildAllToGPU returned empty tree/device structures.");
    }
    if (res.tree.root_id < 0) {
        throw std::runtime_error("BuildAllToGPU produced tree with invalid root_id.");
    }

    std::cout << "Uploaded. N=" << res.dev.N << ", tips=" << res.dev.tips
                << ", per_node_elems=" << res.dev.per_node_elems() << "\n";

    
    
    cudaEventRecord(start);
    PlacementOpBuffer placement_ops;
    UpdateTreeClvs(
        res.dev,
        res.tree,
        res.hostPack,
        placement_ops,
        0);
    double logL = root_likelihood::compute_root_loglikelihood_total(
        res.dev,
        res.tree.root_id,
        res.dev.d_pattern_weights_u,
        nullptr,
        0.0,
        0);
    printf("Initial tree log-likelihood = %.12f\n", logL);
    std::vector<PlacementResult> placement_results;
    EvaluatePlacementQueries(
        res.dev,
        res.eig,
        rate_multipliers,
        placement_ops,
        &placement_results,
        1,
        0);
    free_placement_op_buffer(placement_ops, 0);

    if (!jplace_out.empty()) {
        if (placement_results.size() != placement_queries.size()) {
            std::cerr << "placement result count mismatch before jplace export: got "
                      << placement_results.size() << ", expected " << placement_queries.size()
                      << ". Exporting available prefix only.\n";
        }

        const JplaceTreeExport jplace_tree = build_jplace_tree_export(res.tree);
        std::vector<JplacePlacementRecord> jplace_records;
        const size_t export_count = std::min(placement_results.size(), placement_queries.size());
        jplace_records.reserve(export_count);
        for (size_t i = 0; i < export_count; ++i) {
            const PlacementResult& pres = placement_results[i];
            if (pres.target_id < 0 || pres.target_id >= (int)res.tree.nodes.size()) {
                throw std::runtime_error("Invalid target_id while exporting jplace.");
            }

            const TreeNode& target = res.tree.nodes[pres.target_id];
            if (target.parent < 0) {
                throw std::runtime_error("Best placement landed on root edge, which jplace writer does not support.");
            }

            JplacePlacementRecord rec;
            rec.query_name = placement_queries[i].msa_name.empty()
                ? ("query_" + std::to_string(i))
                : placement_queries[i].msa_name;
            rec.edge_num = jplace_tree.edge_num_by_node[pres.target_id];
            if (rec.edge_num < 0) {
                throw std::runtime_error("Could not map placement edge to jplace edge number.");
            }
            rec.likelihood = pres.loglikelihood;
            rec.like_weight_ratio = 1.0;
            rec.distal_length = pres.proximal_length;
            rec.pendant_length = pres.pendant_length;
            jplace_records.push_back(std::move(rec));
        }

        std::ostringstream invocation;
        for (int i = 0; i < argc; ++i) {
            if (i) invocation << ' ';
            invocation << argv[i];
        }
        write_jplace(jplace_out, jplace_tree.tree, jplace_records, invocation.str());
        std::cout << "Wrote jplace to " << jplace_out << "\n";
    }

    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Get elapsed time (milliseconds)
    cudaEventElapsedTime(&gpu_ms_kernel, start, stop);
    const auto end_gpu = std::chrono::steady_clock::now();
    const double gpu_ms = std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();
    printf("GPU kernel time = %.3f ms\n", gpu_ms_kernel);
    printf("GPU Wall Clock time = %.3f ms\n", gpu_ms);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free_device_tree(res.dev);

    return 0;
}
