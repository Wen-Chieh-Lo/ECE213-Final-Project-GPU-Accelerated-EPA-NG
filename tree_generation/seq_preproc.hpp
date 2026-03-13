#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "tree.hpp"

void remove_sparse_columns(
    std::vector<std::string>& rows,
    std::vector<NewPlacementQuery>& queries,
    std::vector<unsigned>& pattern_weights,
    size_t& sites,
    double gap_threshold);

void remove_repetitive_columns(
    std::vector<std::string>& rows,
    std::vector<NewPlacementQuery>& queries,
    std::vector<unsigned>& pattern_weights,
    size_t& sites);
