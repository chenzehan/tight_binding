#pragma once

#include <variant>

#include "lattice.h"

struct GreenFunctionVariants
{};

struct OperatorPair {
	Eigen::SparseMatrix<double> Op;
	GreenFunctionVariants G;
};