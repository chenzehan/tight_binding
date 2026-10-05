#pragma once

#define _SILENCE_CXX23_DENORM_DEPRECATION_WARNING
#include <Eigen/Sparse>
#include <vector>
#include <complex>

namespace numerical {
	using Complex = std::complex<double>;
	using namespace std::literals::complex_literals;

	template <typename Scalar>
	concept ScalarType = requires
	{ std::is_same_v<Scalar, double> || std::is_same_v<Scalar, Complex>; };

	namespace linalg {
		#undef min
		#undef max
		#include <Spectra/SymEigsSolver.h>
		#include <Spectra/MatOp/SparseSymMatProd.h>
		#include <Spectra/MatOp/SparseSymShiftSolve.h>
		#include <Spectra/SymEigsShiftSolver.h>
		// return eigen values of min, max
		auto matrix_spectral_range_sparse(const Eigen::SparseMatrix<double>& A) -> std::pair<double, double> {
			if (A.rows() > 5000) {
				std::cerr << "matrix_spectral_range_sparse: matrix too large > 5000\n";
				return { 0.0, 0.0 };
			}
			int ncv = std::max(static_cast<int>(A.rows() / 10), 4);
			Spectra::SparseSymMatProd<double> op(A);
			Spectra::SymEigsSolver<Spectra::SparseSymMatProd<double>> eigs(op, 2, ncv);
			eigs.init();
			int nconv = eigs.compute(Spectra::SortRule::BothEnds);
			Eigen::VectorXd evalues;
			if (eigs.info() == Spectra::CompInfo::Successful)
				evalues = eigs.eigenvalues().real();
			else std::cerr << "spectral computation failed\n";
			double E_max = evalues.maxCoeff();
			double E_min = evalues.minCoeff();
			return { E_min, E_max };
		}
		auto complex_matrix_convert_to_real_sparse(const Eigen::SparseMatrix<Complex>& A) -> Eigen::SparseMatrix<double>;
		// return eigen values of min, max, conversion to real
		auto matrix_spectral_range_sparse(const Eigen::SparseMatrix<Complex>& A) -> std::pair<double, double> {
			return matrix_spectral_range_sparse(complex_matrix_convert_to_real_sparse(A));
		}
		auto matrix_spectral_range_dense(const Eigen::MatrixXcd& A) -> std::pair<double, double> {
			Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(A);
			Eigen::VectorXd e = es.eigenvalues();
			return { e.minCoeff(), e.maxCoeff() };
		}
		// using 2 x 2 rotation matrix to reserve complex structure
		auto complex_matrix_convert_to_real_sparse(const Eigen::SparseMatrix<Complex>& A) -> Eigen::SparseMatrix<double> {
			Eigen::SparseMatrix<double> A_real(2 * A.rows(), 2 * A.cols());
			for (int k = 0; k < A.outerSize(); ++k) {
				for (Eigen::SparseMatrix<std::complex<double>>::InnerIterator it(A, k); it; ++it) {
					int row = it.row();
					int col = it.col();
					std::complex<double> value = it.value();

					// convert complex to 2 x 2 real matrix (90 degree rotation)
					Eigen::Matrix2d realMatrix;
					realMatrix << value.real(), -value.imag(),
								  value.imag(),  value.real();

					// into the real matrix
					A_real.insert(row * 2, col * 2) = realMatrix(0, 0);
					A_real.insert(row * 2, col * 2 + 1) = realMatrix(0, 1);
					A_real.insert(row * 2 + 1, col * 2) = realMatrix(1, 0);
					A_real.insert(row * 2 + 1, col * 2 + 1) = realMatrix(1, 1);
				}
			}
			A_real.makeCompressed();
			return A_real;
		}
		auto sparse_matrix_n_eigen_closest_to(const Eigen::SparseMatrix<double>& A, double sigma, int n) -> Eigen::VectorXd {
			// int ncv = std::max(static_cast<int>(A.rows() / 10), 4);
			int ncv = 2 * n + 1;
			Spectra::SparseSymShiftSolve<double> op(A);
			Spectra::SymEigsShiftSolver<decltype(op)> eigs(op, n, ncv, sigma);

			eigs.init();
			// std::cerr << "sparse_matrix_n_eigen_closest_to: start\n";
			eigs.compute(Spectra::SortRule::LargestMagn);
			if (eigs.info() == Spectra::CompInfo::Successful)
			{
				return eigs.eigenvalues();
			}
			std::cerr << "sparse_matrix_n_eigen_closest_to: computation failed\n";
			return {};
		}
		auto sparse_matrix_n_eigen_closest_to(const Eigen::MatrixXd& A, double sigma, int n) -> Eigen::VectorXd {
			// int ncv = std::max(static_cast<int>(A.rows() / 10), 4);
			int ncv = std::max(2 * n + 1, (int)A.rows());
			Spectra::DenseSymShiftSolve<double> op(A);

			// Construct eigen solver object with shift 0
			// This will find eigenvalues that are closest to 0
			Spectra::SymEigsShiftSolver<decltype(op)> eigs(op, n, ncv, sigma);

			eigs.init();
			eigs.compute(Spectra::SortRule::LargestMagn);
			if (eigs.info() == Spectra::CompInfo::Successful)
			{
				return eigs.eigenvalues();
			}
			std::cerr << "sparse_matrix_n_eigen_closest_to: computation failed\n";
			return {};
		}
		template <typename Scalar>
		auto trace_sparse(const Eigen::SparseMatrix<Scalar>& A) -> Scalar {
			Scalar trace = 0.0;
			for (int i = 0; i < A.rows(); ++i) {
				trace += A.coeff(i, i);
			}
			return trace;
		}

		// n: dimension, m: number of vectors
		auto bases_of_uniform_ampl_with_random_phases(int n, int m) -> Eigen::MatrixXcd {
			return (1i * pi * Eigen::ArrayXXd::Random(n, m)).exp().matrix() / sqrt(n);
		}
		auto add_random_phases_to(const Eigen::MatrixXcd& kets) -> Eigen::MatrixXcd {
			return ((1i * pi * Eigen::ArrayXXd::Random(kets.rows(), kets.cols())).exp() * kets.array()).matrix();
		}
		// return Re tr{ A^\dagger * B }
		auto trace_braket(const Eigen::MatrixXcd& bras_T, const Eigen::MatrixXcd& kets) -> double {
			double trace = 0.0;
			for (int i = 0; i < bras_T.rows(); ++i) {
				for (int j = 0; j < bras_T.cols(); ++j) {
					trace += bras_T(i, j).real() * kets(i, j).real() + bras_T(i, j).imag() * kets(i, j).imag();
				}
			}
			return trace;
		}
		auto trace_braket_complex(const Eigen::MatrixXcd& bras_T, const Eigen::MatrixXcd& kets) -> Complex {
			Complex trace = 0.0;
			for (int i = 0; i < bras_T.rows(); ++i) {
				for (int j = 0; j < bras_T.cols(); ++j) {
					trace += std::conj(bras_T(i, j)) * kets(i, j);
				}
			}
			return trace;
		}
	}

	namespace fft {
		bool is_power_of_two(int n) {
			return n > 0 && (n & (n - 1)) == 0;
		}
		struct slice {
			int start, step, n, N;
			int operator()(int i) const {
				int j = start + i * step;
				return j < 0 ? j + N : j;
			}
			template <typename Type>
			auto add_sub(const std::vector<Type>& x, int i, int j) const -> std::pair<Type, Type> {
				int xN = x.size();
				if (i < xN && j < xN) return {x[i] + x[j], x[i] - x[j]};
				if (j >= xN) return { x[i], x[i] };
				return { x[j], -x[j] };
			}
		};
		template <typename Type>
		auto radix_fft(const std::vector<Type>& x, slice s) -> std::vector<Type> {
			int N = s.n;
			if (N <= 1) return { x[s(0)] };
			if (N == 2) {
				int start = s(0), end = s(1);
				return { x[start] + x[end], x[start] - x[end] };
			}
			if (N == 4) {
				int i0 = s(0), i1 = s(1), i2 = s(2), i3 = s(3);
				Type a = x[i0] + x[i2];
				Type b = x[i0] - x[i2];
				Type c = x[i1] + x[i3];
				Type d = x[i1] - x[i3];
				return { a + c, b - 1i * d, a - c, b + 1i * d };
			}
			auto U = radix_fft(x, { s.start, 2 * s.step, N / 2, s.N });
			auto Z = radix_fft(x, { s.start + s.step, 4 * s.step, N / 4, s.N });
			auto Z_ = radix_fft(x, { s.start - s.step, 4 * s.step, N / 4, s.N });
			std::vector<Type> X(N);
			for (int i = 0; i < N / 4; i++) {
				Complex omega = std::exp(-2.0i * pi * static_cast<double>(i) / static_cast<double>(N));
				auto wZ = Z[i] * omega;
				auto wZ_ = Z_[i] * std::conj(omega);
				auto wZc = wZ + wZ_;
				auto wZc_ = 1.0i * (wZ - wZ_);
				X[i] = U[i] + wZc;
				X[i + N / 4] = U[i + N / 4] - wZc_;
				X[i + N / 2] = U[i] - wZc;
				X[i + 3 * N / 4] = U[i + N / 4] + wZc_;
			}
			return X;
		}
		template <typename Type>
		auto radix_fft_conj(const std::vector<Type>& x, slice s, int select = 0) -> std::vector<Type> {
			int N = s.n;
			if (N <= 1) return { x[s(0)] };
			if (N == 2) {
				auto [s0, s1] = s.add_sub(x, s(0), s(1));
				return { s0, s1 };
			}
			if (N == 4) {
				int i0 = s(0), i1 = s(1), i2 = s(2), i3 = s(3);
				auto [a, b] = s.add_sub(x, i0, i2);
				auto [c, d] = s.add_sub(x, i1, i3);
				if (select > 0) return { a + c, b + 1i * d };
				return { a + c, b + 1i * d, a - c, b - 1i * d };
			}
			auto U = radix_fft_conj(x, { s.start, 2 * s.step, N / 2, s.N });
			auto Z = radix_fft_conj(x, { s.start + s.step, 4 * s.step, N / 4, s.N });
			auto Z_ = radix_fft_conj(x, { s.start - s.step, 4 * s.step, N / 4, s.N });
			std::vector<Type> X(N);
			if (select > 0) X.resize(N / 2);
			for (int i = 0; i < N / 4; i++) {
				Complex omega = std::exp(-2.0i * pi * static_cast<double>(i) / static_cast<double>(N));
				auto wZ = Z[i] * std::conj(omega);
				auto wZ_ = Z_[i] * omega;
				auto wZc = wZ + wZ_;
				auto wZc_ = -1i * (wZ - wZ_);
				X[i] = U[i] + wZc;
				X[i + N / 4] = U[i + N / 4] - wZc_;
				if (select > 0) continue;
				X[i + N / 2] = U[i] - wZc;
				X[i + 3 * N / 4] = U[i + N / 4] + wZc_;
			}
			return X;
		}
		// FFT: DCT-III
		template <typename Type>
		auto dct_iii_complex(const std::vector<Type>& C) -> std::vector<Type> {
			int N = C.size();
			if (!is_power_of_two(N)) {
				std::cerr << "dct_iii_complex: N must be power of 2\n";
				return {};
			}
			std::vector<Type> Z(N);
			Z[0] = C[0];
			for (int i = 1; i < N / 2; i++) {
				Complex omega = std::exp(-2.0i * pi * static_cast<double>(i) / static_cast<double>(4 * N));
				Z[i] = std::conj(omega) * (C[i] - 1.0i * C[N - i]);
				Z[N - i] = omega * (C[i] + 1.0i * C[N - i]);
			}
			Z[N / 2] = sqrt(2) * C[N / 2];
			auto y = radix_fft_conj(Z, { 0, 1, N, N });
			std::vector<Type> x(N);
			for (int i = 0; i < N / 2; i++) {
				x[2 * i] = std::move(y[i]);
				x[2 * i + 1] = std::move(y[N - i - 1]);
			}
			return x;
		}

		template <typename Type>
		auto fft(const std::vector<Type>& C) -> std::vector<Type> {
			int N = C.size();
			if (!is_power_of_two(N)) {
				std::cerr << "fft: N must be power of 2\n";
				return {};
			}
			return radix_fft(C, { 0, 1, N, N });
		}
		template <typename Type>
		auto fft_conj(const std::vector<Type>& C) -> std::vector<Type> {
			int N = C.size();
			if (!is_power_of_two(N)) {
				std::cerr << "fft_conj: N must be power of 2\n";
				return {};
			}
			return radix_fft_conj(C, { 0, 1, N, N });
		}
		template <typename Type>
		auto fft_half_conj(const std::vector<Type>& C) -> std::vector<Type> {
			int N = C.size();
			if (!is_power_of_two(N)) {
				std::cerr << "fft_half_conj: N must be power of 2\n";
				return {};
			}
			return radix_fft_conj(C, { 0, 1, 2 * N, 2 * N }, 1);
		}
	}

	namespace Chebyshev {
		// Chebyshev coefficients: 1 / (±z - x) = sum_n alpha_n±(x) * T_n(z)
		//auto expn_coef_alpha_seq(Complex z, int n, int sgn = 1) -> std::vector<Complex> {
		//	std::vector<Complex> alpha(n);
		//	Complex zs = sgn > 0 ? z : -z;
		//	// 注意数值稳定性，确保浮点迭代不发散
		//	Complex s1 = 1.0 / (sqrt(1.0 - 1.0 / (z * z)) * zs);
		//	Complex s2 = 1.0 / ((1.0 + sqrt(z * z) * sqrt(z * z - 1.0) / (z * z)) * zs);
		//	Complex s21n = 2.0 * s1;
		//	Complex s2n = Complex(1.0, 0.0);
		//	alpha[0] = s1;
		//	for (int i = 1; i < n; i++) {
		//		s21n *= s2;
		//		alpha[i] = s21n;
		//	}
		//	return alpha;
		//}
		// Chebyshev coefficients: 1 / (z - x) = sum_n alpha_n(x) * T_n(z)
		auto expn_coef_alpha_seq(Complex z, int n) -> std::vector<Complex> {
			std::vector<Complex> g(n);
			// 注意数值稳定性，确保浮点迭代不发散
			Complex s1 = sqrt(1.0 - z * z);
			Complex s2 = z - 1.0i * s1;
			Complex sn = -2.0i / s1;
			g[0] = -1.0i / s1;
			for (int i = 1; i < n; i++) {
				sn *= s2;
				g[i] = sn;
			}
			return g;
		}
		template <typename T>
		auto Cheb_first_kind_seq(const T& x, int n, T&& identity) -> std::vector<T> {
			std::vector<T> Tn(n);
			Tn[0] = identity;
			if (n == 0) return Tn;
			Tn[1] = x;
			for (int i = 2; i < n; i++) {
				Tn[i] = 2.0 * x * Tn[i - 1] - Tn[i - 2];
			}
			return Tn;
		}

		template <typename Mat, typename Vec>
		concept MatrixVector = requires(Mat m, Vec v) {
			{ m * v } -> std::convertible_to<Vec>;
		};
		// will be memory consuming for very large matrix
		template <typename Mat, typename Vec> requires MatrixVector<Mat, Vec>
		auto Cheb_first_kind_seq_on_bases(const Mat& x, int n, const Vec& v) -> std::vector<Vec> {
			std::vector<Vec> Tn(n);
			Tn[0] = v;
			if (n == 0) return Tn;
			Tn[1] = x * Tn[0];
			for (int i = 2; i < n; i++) {
				Tn[i] = 2.0 * x * Tn[i - 1] - Tn[i - 2];
			}
			return Tn;
		}

		template <typename Mat, typename Vec> requires MatrixVector<Mat, Vec>
		struct Cheb_first_kind_seq_view
		{
			int i;
			const Mat& x;
			Vec v_1, v_0;
			Cheb_first_kind_seq_view(const Mat& x, const Vec& v) : i(0), x(x), v_1(x * v), v_0(v) {}
			auto next() -> std::pair<int, Vec> {
				int i_c = i;
				Vec v = 2.0 * x * v_1 - v_0;
				std::swap(v_1, v);
				std::swap(v_0, v); // retire v_0
				i++;
				return { i_c, v };
			}
		};

		template <ScalarType Scalar>
		auto Cheb_first_kind_seq_braket(const Eigen::SparseMatrix<Scalar>& M, int n,
			const Eigen::MatrixXcd& bras_T, const Eigen::MatrixXcd& kets) -> std::vector<double> {
			std::vector<double> mu_n(n);
			Eigen::MatrixXcd Tn_2 = kets;
			Eigen::MatrixXcd Tn_1 = M * kets;
			mu_n[0] = linalg::trace_braket(bras_T, kets);
			if (n == 0) return mu_n;
			mu_n[1] = linalg::trace_braket(bras_T, Tn_1);
			for (int i = 2; i < n; i++) {
				std::cout << i << std::endl;
				Eigen::MatrixXcd Tn = 2.0 * M * Tn_1 - Tn_2;
				mu_n[i] = linalg::trace_braket(bras_T, Tn);
				Tn_2 = std::move(Tn_1);
				Tn_1 = std::move(Tn);
				std::cout << mu_n[i] << std::endl;
			}
			return mu_n;
		}

		template <ScalarType Scalar>
		Scalar weight(Scalar z) {
			return 1.0 / sqrt(1.0 - z * z);
		}

		int ortho_const(int n) {
			return n == 0 ? 1 : 2;
		}
	}

	namespace kernels {
		struct Dirichlet {
			auto operator()(int n) -> double const {
				return 1.0;
			}
		};

		struct Fejer {
			int N;
			Fejer(int N) : N(N) {}
			auto operator()(int n) -> double const {
				return 1.0 - static_cast<double>(n) / N;
			}
		};

		struct Lorentz {
			int N;
			double lambda;
			// best for Green function, lambda usually 3 ~ 5
			Lorentz(int N, double lambda) : N(N), lambda(lambda) {}
			auto operator()(int n) -> double const {
				return std::sinh(lambda * (1.0 - static_cast<double>(n) / N)) / std::sinh(lambda);
			}
		};

		struct Jackson {
			int N1;
			Jackson(int N) : N1(N + 1) {}
			auto operator()(int n) -> double const {
				double s1 = pi / N1;
				double s2 = n * s1;
				return ((N1 - n) * std::cos(s2) + std::sin(s2) / std::tan(s1)) / N1;
			}
		};

		struct Lanczos {
			double pi_N;
			int M;
			Lanczos(int N, int M) : pi_N(pi / N), M(M) {}
			auto operator()(int n) -> double const {
				return std::pow(std::sin(n * pi_N) / pi_N, M);
			}
		};

		struct Wang_and_Zunger {
			double alpha_N, beta;
			Wang_and_Zunger(int N, double alpha, double beta) : alpha_N(alpha / N), beta(beta) {}
			auto operator()(int n) -> double const {
				return std::exp(-std::pow(n * alpha_N, beta));
			}
		};

		auto kernel_seq(int N, const auto& kernel) -> std::vector<double> {
			return std::views::iota(0, N) | std::views::transform(kernel) | std::ranges::to<std::vector>();
		}
	}
}
