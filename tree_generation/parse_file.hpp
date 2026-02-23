#pragma once

#include <string>
#include <vector>

namespace parse {

struct Alignment {
    std::vector<std::string> names;
    std::vector<std::string> sequences;
    size_t sites = 0;
};

struct ModelConfig {
    int states = 4;
    std::string subst_model;
    int ncat = 1;
    double alpha = 1.0;
    double pinv = 0.0;
    std::vector<double> freqs;
    std::vector<double> rates;
    std::vector<double> rate_weights;
    bool per_rate_scaling = false;
};

struct FilesConfig {
    std::string tree_alignment;  // tree MSA
    std::string query_alignment; // placement queries MSA
    std::string tree;
};

struct RunConfig {
    ModelConfig model;
    FilesConfig files;
};

Alignment read_alignment_file(const std::string& path);
struct RunInputs {
    RunConfig config;
    Alignment tree_alignment;
    std::string tree;
};

// Normalize Newick text by resolving polytomies into a binary tree.
// If parsing fails, returns the input unchanged.
std::string normalize_newick(const std::string& raw);

} // namespace parse
