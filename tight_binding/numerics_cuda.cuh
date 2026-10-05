#ifndef NUMERICS_CUDA_CUH
#define NUMERICS_CUDA_CUH

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>

namespace cuda {
	using Complex = std::complex<double>;
	std::vector<double> Cheb_first_kind_seq_braket(const Eigen::SparseMatrix<double>& M, int n,
		const Eigen::MatrixXcd& bras_T, const Eigen::MatrixXcd& kets);
	std::vector<double> Cheb_first_kind_seq_braket(const Eigen::SparseMatrix<Complex>& M, int n,
		const Eigen::MatrixXcd& bras_T, const Eigen::MatrixXcd& kets);
}

#endif