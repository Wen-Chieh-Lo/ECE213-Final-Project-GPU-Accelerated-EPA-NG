#pragma once
#include <vector>
#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include <algorithm>
#include <cassert>

struct EigResult {
    std::vector<double> lambdas;
    std::vector<double> V;
    std::vector<double> Vinv;
    std::vector<double> U;   // Store U for the direct formula path
};

EigResult gtr_eigendecomp_cpu(
    const double* Q_rowmajor,   // Q (row-major, size n*n)
    const double* pi,           // Stationary frequencies (length n, sum=1)
    int n);

void pmatrix_from_triple(const double* Vinv, const double* V,
                                const double* lamb, double r, double t, double p,
                                double* P, int n);

void pmatrix_direct(const double* U, const double* pi,
                           const double* lamb, double r, double t, double p,
                           double* P, int n);
