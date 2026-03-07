#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>
#include <libpll/pll.h>

#include "tree.hpp"
#include "parse_file.hpp"
#include "seq_preproc.cpp"

static std::string read_file_to_string(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) throw std::runtime_error("Cannot open file: " + path);
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

static std::string resolve_path(const std::filesystem::path& base, const std::string& p) {
    namespace fs = std::filesystem;
    fs::path candidate(p);
    if (candidate.empty()) return {};
    if (candidate.is_absolute()) return candidate.string();
    return (base / candidate).string();
}

void print_tree_rec(const TreeBuildResult& T, int node_id, int depth)
{
    if (node_id < 0) return;
    const TreeNode &nd = T.nodes[node_id];

    for (int i = 0; i < depth; ++i) std::cout << "  ";

    std::cout << "[" << nd.id << "]";
    if (nd.is_tip) {
        std::cout << " (tip: " << nd.name << ")";
    } else {
        std::cout << " (inner)";
    }

    if (nd.parent >= 0) {
        std::cout << "  len=" << nd.branch_length_to_parent
                  << "  parent=" << nd.parent;
    } else {
        std::cout << "  <ROOT>";
    }
    std::cout << "\n";

    if (!nd.is_tip) {
        if (nd.left  >= 0) print_tree_rec(T, nd.left,  depth + 1);
        if (nd.right >= 0) print_tree_rec(T, nd.right, depth + 1);
    }
}

void print_tree_structure(const TreeBuildResult& T)
{
    std::cout << "==== Tree structure (indented) ====\n";
    if (T.root_id < 0) {
        std::cout << "No root_id set!\n";
        return;
    }
    print_tree_rec(T, T.root_id, 0);
    std::cout << "===================================\n";
}

static void print_pmat_host_node_rc0(const HostPacking& H, int node_id, int states, int rate_cats)
{
    if (node_id < 0 || states <= 0 || rate_cats <= 0) return;
    const size_t per_node = (size_t)rate_cats * (size_t)states * (size_t)states;
    const size_t base = (size_t)node_id * per_node;
    if (H.pmats.size() < base + (size_t)states * (size_t)states) return;

    std::cout << "PMAT node " << node_id << " rc 0:\n";
    const double* P = H.pmats.data() + base;
    for (int i = 0; i < states; ++i) {
        for (int j = 0; j < states; ++j) {
            std::cout << P[i * states + j];
            if (j + 1 < states) std::cout << " ";
        }
        std::cout << "\n";
    }
}

static void print_edges_by_length(const TreeBuildResult& T, double target_len, double tol = 1e-6)
{
    std::cout << "Edges with branch length ~= " << target_len << ":\n";
    for (const auto& nd : T.nodes) {
        if (nd.parent < 0) continue;
        if (std::fabs(nd.branch_length_to_parent - target_len) <= tol) {
            std::cout << "  node_id=" << nd.id
                      << " parent=" << nd.parent
                      << " len=" << nd.branch_length_to_parent << "\n";
        }
    }
}

static void normalize_vector(std::vector<double>& vec) {
    double sum = 0.0;
    for (double v : vec) sum += v;
    if (sum <= 0.0) return;
    for (double& v : vec) v /= sum;
}

static std::vector<double> ensure_normalized_pi(std::vector<double> pi, int states) {
    if ((int)pi.size() != states) pi.assign(states, 1.0 / states);
    normalize_vector(pi);
    return pi;
}

static std::vector<double> build_mixture_weights(const parse::ModelConfig& model, int rate_cats) {
    std::vector<double> weights;
    if (rate_cats <= 0) return weights;
    if ((int)model.rate_weights.size() == rate_cats) {
        weights = model.rate_weights;
    } else {
        weights.assign(rate_cats, 1.0 / rate_cats);
    }

    double sum = 0.0;
    for (double v : weights) sum += v;
    if (sum <= 0.0) {
        weights.assign(rate_cats, 1.0 / rate_cats);
        return weights;
    }
    for (double& v : weights) v /= sum;
    return weights;
}

static std::vector<double> build_gamma_rate_categories(double alpha, int rate_cats) {
    std::vector<double> rates(rate_cats, 1.0);
    if (rate_cats <= 1 || alpha <= 0.0) return rates;
    std::vector<double> gamma_tmp(rate_cats);
    int status = pll_compute_gamma_cats(static_cast<float>(alpha), rate_cats, gamma_tmp.data(), PLL_GAMMA_RATES_MEAN);
    if (status != PLL_SUCCESS) {
        throw std::runtime_error("pll_compute_gamma_cats failed.");
    }
    for (int i = 0; i < rate_cats; ++i) {
        rates[i] = static_cast<double>(gamma_tmp[i]);
    }
    return rates;
}

static std::vector<double> build_gtr_q_matrix(
    int states,
    const parse::ModelConfig& model,
    const std::vector<double>& pi)
{
    std::vector<double> Q(states * states, 0.0);
    if (states != 4 || model.rates.size() < 6 || pi.size() != 4)
        return Q;

    auto set_pair = [&](int i, int j, double rate) {
        Q[i * states + j] = rate * pi[j];
        Q[j * states + i] = rate * pi[i];
    };

    set_pair(0, 1, model.rates[0]); // A-C
    set_pair(0, 2, model.rates[1]); // A-G
    set_pair(0, 3, model.rates[2]); // A-T
    set_pair(1, 2, model.rates[3]); // C-G
    set_pair(1, 3, model.rates[4]); // C-T
    set_pair(2, 3, model.rates[5]); // G-T

    for (int i = 0; i < states; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j < states; ++j) {
            if (i != j) row_sum += Q[i * states + j];
        }
        Q[i * states + i] = -row_sum;
    }

    // 1) Compute μ = - Σ_i π_i * Q_ii
    double mu = 0.0;
    for (int i = 0; i < states; ++i) {
        mu -= pi[i] * Q[i * states + i];
    }

    // 2) Scale the entire Q by the same μ
    for (int idx = 0; idx < states * states; ++idx) {
        Q[idx] /= mu;
    }

    return Q;
}

int main(int argc, char** argv) {
    CLI::App app{"test_tree_gen"};
    app.get_formatter()->column_width(40);

    // Single config object filled directly by CLI11 options.
    parse::RunConfig config;

    // ---- Input (files/tree) ----
    std::string tree_newick;

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

    app.add_option("--states", config.model.states, "Number of states")->group("Model");
    app.add_option("--subst-model", config.model.subst_model, "Substitution model")->group("Model");
    app.add_option("--ncat", config.model.ncat, "Number of rate categories")->group("Model");
    app.add_option("--alpha", config.model.alpha, "Gamma shape alpha")->group("Model");
    app.add_option("--pinv", config.model.pinv, "Proportion of invariant sites")->group("Model");
    app.add_option("--freqs", config.model.freqs, "Equilibrium freqs (list)")->group("Model")->delimiter(',');
    app.add_option("--rates", config.model.rates, "GTR rates rAC,rAG,rAT,rCG,rCT,rGT (list)")->group("Model")->delimiter(',');
    app.add_option("--rate-weights", config.model.rate_weights, "Rate category weights (list)")->group("Model")->delimiter(',');
    app.add_flag("--no-per-rate-scaling", no_per_rate_scaling, "Disable per-rate scaling")->group("Model");

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

    auto& alignment = inputs.tree_alignment;

    auto& msa_names = alignment.names;
    auto& rows = alignment.sequences;
    size_t sites = alignment.sites;
    const std::string& newick = inputs.tree;
    const auto& model = inputs.config.model;
    int states = model.states;
    int rate_cats = model.ncat;
    bool per_rate_scaling = model.per_rate_scaling;

    std::vector<double> pi = ensure_normalized_pi(model.freqs, states);
    std::vector<double> rate_weights = build_mixture_weights(model, rate_cats);
    std::vector<double> rate_multipliers = build_gamma_rate_categories(model.alpha, rate_cats);
    std::vector<double> Q = build_gtr_q_matrix(states, model, pi);

    const std::string& query_alignment_cfg = inputs.config.files.query_alignment;
    std::vector<NewPlacementQuery> placement_queries =
        build_placement_query(resolve_path(config_base, query_alignment_cfg));
    printf("Query sequences for placement: %zu\n", placement_queries.size());
    for(auto & q : placement_queries) {
        printf("  Query '%s'\n", q.msa_name.c_str());
    }

    // sequence preprocessing
    remove_sparse_columns(rows, placement_queries, sites, 0.7);
    remove_repetitive_columns(rows, placement_queries, sites);

    // const auto start_gpu = std::chrono::steady_clock::now();
    auto res = BuildAllToGPU(
        msa_names,
        rows,
        newick,
        Q,
        pi,
        rate_multipliers,
        rate_weights,
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
    print_tree_structure(res.tree);

    
    const auto start_gpu = std::chrono::steady_clock::now();
    cudaEvent_t start, stop;
    float gpu_ms_kernel = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    double logL = EvaluateTreeLogLikelihood(
        res.dev,
        res.tree,
        res.hostPack,
        pi,
        rate_weights,
        0
    );
    printf("Initial tree log-likelihood = %.12f\n", logL);
    UpdateTreeLogLikelihood_device(
        res.dev,
        res.tree,
        res.hostPack,
        res.eig,
        res.queries,
        rate_multipliers,
        &placement_queries,
        &pi,
        &rate_weights,
        logL,
        /*debug_mid=*/true,
        1,
        0);

    
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

    print_tree_structure(res.tree);
    free_device_tree(res.dev);

    return 0;
}
