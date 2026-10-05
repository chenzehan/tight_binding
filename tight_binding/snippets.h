#pragma once

#define _SILENCE_CXX23_DENORM_DEPRECATION_WARNING
#include <iostream>

#include "unit_cell.h"
#include "lattice.h"
#include "models.h"
#include "functions.h"

#include <iostream>
#include <cmath>
#include <numbers>
#include <print>
using namespace std;
using numbers::pi;
using numerical::Complex;

void s_ip_model()
{
    auto p = models::TI_sp;
   
    auto X = PartialLattice{ p, {1, 40} };
    auto Kx = utils::linspace(-1.2*pi, 1.2*pi, 100) | ranges::to<vector>();
    auto E = Kx | std::views::transform([&X](auto kx) {
   	//Eigen::MatrixXcd H = X.get_Hamiltonian();
        Eigen::MatrixXcd H = X.get_Hamiltonian_periodic_extend({ kx, 0, 0 }, { 1, 0 });
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(H);
        return es.eigenvalues();
    });
   
    std::vector<std::vector<double>> band_dispersion(X.get_total_number());
    for (auto&& e : E) {
        for (int i = 0; i < e.size(); i++) {
            band_dispersion[i].push_back(e(i));
        }
    }
    plt::figure(true);
    plt::hold(true);
    for (auto&& e : band_dispersion) {
        plt::plot(Kx, e)->color("blue");
    }
    plt::show();
}

void DoS_of_Chain()
{
    auto d = models::Chain;
    d.add_hopping("A", 1_NN, 1.);
    // d.draw_band_dispersion();
    auto X = PartialLattice{ d, LatticeVector<1>{100} };
    auto G = PartialLattice<1>::GreenFunction<double>(100, X.get_Hamiltonian_sparse<double>());
    auto [a, b] = G.spectral_range();
    std::print("{} {}\n", a, b);
    auto DoS = [&G](double E) {
        return -1.0 / pi * numerical::linalg::trace_sparse(G.direct_eval(E));
        };
    auto E = utils::linspace(a, b, 200) | std::ranges::to<std::vector>();
    auto DOS = E | std::views::transform(DoS) | std::ranges::to<std::vector>();
    plt::figure(true);
    plt::plot(E, DOS);
    plt::show();
}

void DoS_of_Square()
{
	auto d = models::Square;
	double t = 1.;
	d.add_hopping("A", 1_NN, t);
	// d.draw_band_dispersion();
	int N = 100;
	int ChebN = 1000;
	auto X = PartialLattice{ d, LatticeVector<2>{N, N} };
	auto G = PartialLattice<1>::GreenFunction<double>(ChebN, X.get_Hamiltonian_sparse<double>(), -4.001 * t, 4.001 * t);
	Eigen::MatrixXcd kets = numerical::linalg::bases_of_uniform_ampl_with_random_phases(N * N, 4);
	auto f = G.imag(kets, kets);
	auto DoS = [&](double E) -> double {
		Complex omega_t = G.scale(E + 0e-8i, 1.0i); // This imaginary part controls the spread
		double d = -1.0 / 4 / pi * f(omega_t);
		return d;
		};
	auto E = utils::linspace(-4 * t, +4 * t, 2000) | std::ranges::to<std::vector>();
	auto DOS = E | std::views::transform(DoS) | std::ranges::to<std::vector>();
	for (int i = 0; i < 2000; i++) {
		std::print("E = {}, DoS = {}\n", E[i], DOS[i]);
	}
	plt::figure(true);
	plt::plot(E, DOS);
	//plt::ylim({ -1.05, 3.35 });
	plt::show();
}

void DoS_of_Graphene()
{
	auto d = models::graphene;
	int N = 100;
	int ChebN = 4000;
	auto X = PartialLattice{ d, LatticeVector<2>{N, N} };
	// auto G = PartialLattice<1>::GreenFunction<double>(ChebN, X.get_Hamiltonian_periodic_extend_bare_sparse<double>({1, 1}), -3.001, 3.001);
	auto G = PartialLattice<1>::GreenFunction<Complex>(ChebN, X.get_Hamiltonian_bare_sparse<Complex>(), -6.001, 6.001);
	//Eigen::MatrixXcd kets = numerical::linalg::bases_of_uniform_ampl_with_random_phases(X.get_total_number(), 4);
	//auto f = G.imag(kets, kets);
	//auto DoS = [&](double E) -> double {
	//	Complex omega_t = G.scale(E + 0e-8i, 1.0i); // This imaginary part controls the spread
	//	double d = -1.0 / 4 / pi * f(omega_t);
	//	return d;
	//	};
	auto DoS = density_of_states_function(G);
	auto E = utils::linspace(-5, +5, 2000) | std::ranges::to<std::vector>();
	auto DOS = E | std::views::transform(DoS) | std::ranges::to<std::vector>();
	/*for (int i = 0; i < 2000; i++) {
		std::print("E = {}, DoS = {}\n", E[i], DOS[i]);
	}*/
	plt::figure(true);
	plt::plot(E, DOS);
	//plt::ylim({ -1.05, 3.35 });
	plt::show();
}

void DoS_of_stub()
{
	auto d = models::stub_1d_p(1.0, 0.5);
	auto H_ref = PartialLattice{ d, LatticeVector<1>{100} }.get_Hamiltonian_sparse<double>();
	const int LatX = 100000, ChebN = 5000; // larger ChebN gives better accuracy
	auto X = PartialLattice{ d, LatticeVector<1>{LatX} };
	// X.remove_atoms_random("B", 90000);
	/*Eigen::MatrixXcd H_ = X.get_Hamiltonian_periodic_extend({ 0, 0, 0 }, LatticeVector<1>{1});
	Eigen::MatrixXcd Id(H_.rows(), H_.cols());
	Id.setIdentity();*/
	auto G = PartialLattice<1>::GreenFunction<double>(ChebN, X.get_Hamiltonian_sparse<double>(), H_ref, 0.01);
	// H_ = G.scale(H_, Eigen::MatrixXcd(Id));
	Eigen::MatrixXcd kets = numerical::linalg::bases_of_uniform_ampl_with_random_phases(X.get_total_number(), 4);

	// timer
	auto start = std::chrono::high_resolution_clock::now();

	auto f = G.imag(kets, kets);

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
#ifdef USE_CUDA
	std::cout << "Using CUDA, elapsed time: " << elapsed.count() << " s\n";
#else
	std::cout << "Using MKL, elapsed time: " << elapsed.count() << " s\n";
#endif

	// utils::output_to_file("D:/documents/OneDrive - HKUST Connect/Research/Projects/Tight Binding/test data/moments.txt", f.moments);

	auto DoS = [&](double E) -> double {
		Complex omega_t = E + 0e-8i; // This imaginary part controls the spread
		double d = -1.0 / 4 / pi * f(omega_t);
		// std::print("doing E = {}, DoS = {}\n", E, d);
		return d;
		// return -1.0 / pi * numerical::linalg::trace_sparse(G.direct_eval(E));
		};
	auto E = utils::linspace(-0.3, +0.3, 1000) | std::ranges::to<std::vector>();
	auto DOS = E | std::views::transform(DoS) | std::ranges::to<std::vector>();
	/*auto [E_f, DOS_f] = f.fft(1);
	auto [sl, sf] = f.slice_range(-0.3, 0.3);
	auto E = E_f | std::views::drop(sl) | std::views::take(sf - sl) | std::ranges::to<std::vector>();
	auto DOS = DOS_f | std::views::drop(sl) | std::views::take(sf - sl) | std::ranges::to<std::vector>();*/

	const string folder = "D:/documents/OneDrive - HKUST Connect/Research/Projects/Tight Binding/test data/";
	////const string folder = "C:/Users/zchende/OneDrive - HKUST Connect/Research/Projects/Tight Binding/test data/";
	utils::output_to_file(folder + 
		std::format("DoS alpha={:.1f} sites={} ChebN={} imp={:.2f} no={}.txt", 0.5, LatX, ChebN, 0.0, 0), E, DOS);
	plt::figure(true);
	plt::plot(E, DOS);
	// plt::ylim({ -0.05, 0.35 });
	plt::show();
}

double conductivity_test_sawtooth(PartialLattice<1>& X, int N, double E)
{
	auto a_d = [](int i) { return i == 0 ? 1 : 2; };
	auto H_og = X.get_Hamiltonian_sparse<double>();
	// auto H_b = H.toDense();
	// auto G = PartialLattice<1>::GreenFunction<double>(N, H_og, -4./3 - 1e-6, 2./3 + 1e-6);
	// auto G = PartialLattice<1>::GreenFunction<double>(N, H_og, -1.000001, 1.000001);
	double t = -E / 2.;
	auto G = PartialLattice<1>::GreenFunction<double>(N, H_og, 4 * t - 2. - 1e-6, 4 * t + 1e-6);
	double e = G.scale(E, 1.0);
	// Eigen::MatrixXcd G_b = ((e + 1e-2i) * Eigen::MatrixXcd::Identity(H_b.rows(), H_b.cols()) - H_b).inverse();
	// Eigen::MatrixXd ImG_b = G_b.imag();
	auto& H = G.scaled_HE;
	auto op_x = X.op_position(0);
	auto v_x = op_x * H - H * op_x;
	int R = 4;
	Eigen::MatrixXcd kets = numerical::linalg::bases_of_uniform_ampl_with_random_phases(X.get_total_number(), R);
	auto g = numerical::kernels::Lorentz(N, 6.);
	// right
	Eigen::MatrixXcd kets_R(kets.rows(), kets.cols()), aR_m(kets.rows(), kets.cols());
	kets_R.setZero();
	numerical::Chebyshev::Cheb_first_kind_seq_view TnR_v(e, 1.);
	numerical::Chebyshev::Cheb_first_kind_seq_view aR_v(H, kets);
	auto alpha = numerical::Chebyshev::expn_coef_alpha_seq(e, N + 1);
	int i = 0;
	while (i < N) {
		std::tie(i, aR_m) = aR_v.next();
		auto [_, Tn_e] = TnR_v.next();
		kets_R += Tn_e * a_d(i) * aR_m;
		// kets_R += alpha[i].imag() * g(i) * aR_m;
	}
	// kets_R = ImG_b * kets;
	// std::cout << kets_R << std::endl;
	// left
	Eigen::MatrixXcd kets_L(kets.rows(), kets.cols()), aL_m(kets.rows(), kets.cols()), v_kets = v_x * kets;
	kets_L.setZero();
	numerical::Chebyshev::Cheb_first_kind_seq_view TnL_v(e, 1.);
	numerical::Chebyshev::Cheb_first_kind_seq_view aL_v(H, v_kets);
	i = 0;
	while (i < N) {
		std::tie(i, aL_m) = aL_v.next();
		auto [_, Tn_e] = TnL_v.next();
		kets_L += Tn_e * a_d(i) * aL_m;
		// kets_L += alpha[i].imag() * g(i) * aL_m;
	}
	// kets_L = ImG_b * v_kets;
	kets_L = v_x * kets_L;
	// std::cout << kets_L << std::endl;
	auto sigma_t = numerical::linalg::trace_braket(kets_L, kets_R);
	return sigma_t / R;
}

void conductivity_spectrum_test(const Eigen::SparseMatrix<Complex>& H_og, const Eigen::SparseMatrix<double>& op_x, const Eigen::SparseMatrix<double>& op_y,
	int N)
{
	auto a_d = [](int i) { return i == 0 ? 1 : 2; };
	// auto H_og = X.get_Hamiltonian_sparse<double>();
	auto G = PartialLattice<2>::GreenFunction<Complex>(N, H_og, -4.0001, 4.0001);
	// Eigen::MatrixXcd G_b = ((e + 1e-2i) * Eigen::MatrixXcd::Identity(H_b.rows(), H_b.cols()) - H_b).inverse();
	// Eigen::MatrixXd ImG_b = G_b.imag();
	auto& H = G.scaled_HE;
	// auto op_x = X.op_position(0);
	auto v_x = op_x * H - H * op_x;
	// auto v_y = op_y * H - H * op_y;
	int R = 128;
	Eigen::MatrixXcd kets = numerical::linalg::bases_of_uniform_ampl_with_random_phases(H_og.rows(), R);
	auto g = numerical::kernels::Lorentz(N, 6.);
	// right
	auto aR_vec = numerical::Chebyshev::Cheb_first_kind_seq_on_bases(H, N, kets);
	for (int i = 0; i < N; i++) {
		aR_vec[i] *= g(i);
	}
	auto phi_R = numerical::fft::dct_iii_complex(aR_vec);
	// left
	Eigen::MatrixXcd v_kets = v_x * kets;
	auto aL_vec = numerical::Chebyshev::Cheb_first_kind_seq_on_bases(H, N, v_kets);
	for (int i = 0; i < N; i++) {
		aL_vec[i] = g(i) * (v_x * aL_vec[i]);
	}
	auto phi_L = numerical::fft::dct_iii_complex(aL_vec);
	std::vector<double> epsilon(N), sigma(N);
	for (int k = 0; k < N; k++) {
		double epsilon_k = cos((k + 0.5) / N * pi);
		auto sigma_k = numerical::linalg::trace_braket(phi_L[k], phi_R[k]) * pow(numerical::Chebyshev::weight(epsilon_k), 2);
		// std::cout << k << " " << epsilon_k << " " << sigma_k / R << std::endl;
		epsilon[k] = epsilon_k;
		sigma[k] = -sigma_k / R;
	}
	plt::figure(true);
	plt::plot(epsilon, sigma);
	plt::show();

	/*Eigen::MatrixXcd kets_R(kets.rows(), kets.cols()), kets_L(kets.rows(), kets.cols());
	kets_R.setZero();
	kets_L.setZero();
	double e = G.scale(E, 1.0);
	auto alpha = numerical::Chebyshev::Cheb_first_kind_seq(e, N, 1.);
	for (int i = 0; i < N; i++) {
		kets_R += cos(i * (31.5) / N * pi) * a_d(i) * aR_vec[i];
		kets_L += cos(i * (31.5) / N * pi) * a_d(i) * aL_vec[i];
	}
	Eigen::MatrixXcd kets_Lt(kets.rows(), kets.cols()), aL_m(kets.rows(), kets.cols());
	kets_Lt.setZero();
	numerical::Chebyshev::Cheb_first_kind_seq_view TnL_v(e, 1.);
	numerical::Chebyshev::Cheb_first_kind_seq_view aL_v(H, Eigen::MatrixXcd(v_x * kets));
	Eigen::MatrixXcd kets_Rt(kets.rows(), kets.cols()), aR_m(kets.rows(), kets.cols());
	kets_Rt.setZero();
	numerical::Chebyshev::Cheb_first_kind_seq_view TnR_v(e, 1.);
	numerical::Chebyshev::Cheb_first_kind_seq_view aR_v(H, kets);*/
	/*int i = 0;
	while (i < N - 1) {
		std::tie(i, aR_m) = aR_v.next();
		auto [_, Tn_e] = TnR_v.next();
		kets_Rt += Tn_e * a_d(i) * g(i) * aR_m;
	}
	i = 0;
	while (i < N - 1) {
		std::tie(i, aL_m) = aL_v.next();
		auto [_, Tn_e] = TnL_v.next();
		kets_Lt += Tn_e * g(i) * a_d(i) * aL_m;
	}
	kets_Lt = v_x * kets_Lt;
	double diff = (kets_Lt - kets_L).norm();
	std::cout << " diff " << diff << std::endl;
	diff = (kets_Rt - kets_R).norm();
	std::cout << " diff " << diff << std::endl;
	auto sigma_t = numerical::linalg::trace_braket(kets_L, kets_R) * pow(numerical::Chebyshev::weight(e), 2);
	auto sigma_u = numerical::linalg::trace_braket(kets_Lt, kets_Rt) * pow(numerical::Chebyshev::weight(e), 2);
	std::cout << "spectrum " << e << " " << sigma_t / R << std::endl;
	std::cout << "spectrum " << e << " " << sigma_u / R << std::endl;*/
	// return sigma_t / R * 3 * sqrt(3);
}

void conductivity_Kubo_Bastin_test(const Eigen::SparseMatrix<Complex>& H_og, const Eigen::SparseMatrix<double>& op_x, const Eigen::SparseMatrix<double>& op_y,
	int N, double E_min, double E_max, double Volume = 1.0)
{
	auto a_d = [](int i) { return i == 0 ? 1 : 2; };
	// auto H_og = X.get_Hamiltonian_sparse<double>();
	auto G = PartialLattice<2>::GreenFunction<Complex>(N, H_og, E_min - 1e-8, E_max + 1e-8);
	// Eigen::MatrixXcd G_b = ((e + 1e-2i) * Eigen::MatrixXcd::Identity(H_b.rows(), H_b.cols()) - H_b).inverse();
	// Eigen::MatrixXd ImG_b = G_b.imag();
	auto& H = G.scaled_HE;
	// auto op_x = X.op_position(0);
	auto v_x = op_x * H - H * op_x;
	auto v_y = op_y * H - H * op_y;
	int R = 4;
	auto g = numerical::kernels::Lorentz(N, 6.);
	Eigen::MatrixXcd kets = numerical::linalg::bases_of_uniform_ampl_with_random_phases(H_og.rows(), R);
	// Eigen::MatrixXcd kets(H_og.rows(), H_og.rows());
	// kets.setIdentity();
	// a_mr L/R
	// right
	std::cout << "generating a_mr L/R" << std::endl;
	auto aL_vec = numerical::Chebyshev::Cheb_first_kind_seq_on_bases(H, N, kets);
	for (int i = 0; i < N; i++) {
		aL_vec[i] *= g(i);
	}
	// left
	Eigen::MatrixXcd v_kets = v_y * kets;
	auto aR_vec = numerical::Chebyshev::Cheb_first_kind_seq_on_bases(H, N, v_kets);
	for (int i = 0; i < N; i++) {
		aR_vec[i] = g(i) * (v_x * aR_vec[i]);
	}
	// test
	//Eigen::ArrayX2cd mu(N, N);
	//for (int i = 0; i < N; i++) {
	//	for (int j = 0; j < N; j++) {
	//		mu(i, j) = numerical::linalg::trace_braket_complex(aL_vec[i], aR_vec[j]);
	//	}
	//}
	//auto g_mn = [a_d](double e, int m, int n) -> Complex {
	//	double pre = a_d(m) * a_d(n);
	//	double theta = acos(e), e_s = sqrt(1 - e * e), nd = static_cast<double>(n), md = static_cast<double>(m);
	//	double Tm_e = cos(md * theta), Tn_e = cos(nd * theta);
	//	// return pre * ((e ) * exp(1i * nd * theta) * Tm_e + (e ) * exp(-1i * md * theta) * Tn_e);// / pow(e_s, 4);
	//	return pre * ((e - 1i * nd * e_s) * exp(1i * nd * theta) * Tm_e + (e + 1i * md * e_s) * exp(-1i * md * theta) * Tn_e);// / pow(e_s, 4);
	//	};

	// phi_3 L/R
	std::cout << "generating phi_3 R/L" << std::endl;
	auto phi_3L = numerical::fft::dct_iii_complex(aL_vec);
	auto phi_3R = numerical::fft::dct_iii_complex(aR_vec);
	// phi_1 L/R
	for (int i = 0; i < N; i++) {
		Complex c_i = exp(1i * pi * static_cast<double>(i) / static_cast<double>(2 * N)) * static_cast<double>(a_d(i));
		aL_vec[i] = c_i * aL_vec[i];
		aR_vec[i] = c_i * aR_vec[i];
	}
	std::cout << "generating phi_1 R/L" << std::endl;
	auto phi_1L = numerical::fft::fft_half_conj(aL_vec);
	auto phi_1R = numerical::fft::fft_half_conj(aR_vec);
	// i * phi_2 L/R
	/*for (int i = 0; i < N; i++) {
		aL_vec[i] = 1i * static_cast<double>(i) * aL_vec[i];
		aR_vec[i] = 1i * static_cast<double>(i) * aR_vec[i];
	}
	auto iphi_2L = numerical::fft::fft_half_conj(aL_vec);
	auto iphi_2R = numerical::fft::fft_half_conj(aR_vec);*/
	// phi_2 L/R
	for (int i = 0; i < N; i++) {
		aL_vec[i] = static_cast<double>(i) * aL_vec[i];
		aR_vec[i] = static_cast<double>(i) * aR_vec[i];
	}
	std::cout << "generating phi_2 R/L" << std::endl;
	auto phi_2L = numerical::fft::fft_half_conj(aL_vec);
	auto phi_2R = numerical::fft::fft_half_conj(aR_vec);

	// sum of phi
	// std::vector<double> phi_sum_31(N), phi_sum_13(N), phi_sum_32(N), phi_sum_23(N);
	std::vector<Complex> phi_sum_31(N), phi_sum_13(N), phi_sum_32(N), phi_sum_23(N);
	for (int k = 0; k < N; k++) {
		phi_sum_31[k] = numerical::linalg::trace_braket_complex(phi_3L[k], phi_1R[k]);
		phi_sum_13[k] = numerical::linalg::trace_braket_complex(phi_1L[k], phi_3R[k]);
		phi_sum_32[k] = numerical::linalg::trace_braket_complex(phi_3L[k], phi_2R[k]);
		phi_sum_23[k] = numerical::linalg::trace_braket_complex(phi_2L[k], phi_3R[k]);
	}
	// std::vector<double> p(N), w(N);
	std::vector<Complex> p(N), w(N);
	for (int k = 0; k < N; k++) {
		/*p[k] = phi_sum_31[k] + phi_sum_13[k] + (k > 0 ? p[k - 1] : 0);
		w[k] = phi_sum_23[k] - phi_sum_32[k] + (k > 0 ? w[k - 1] : 0);*/
		p[k] = phi_sum_31[k] + phi_sum_13[k];
		w[k] = phi_sum_23[k] - phi_sum_32[k];
	}
	std::vector<double> epsilon(N);
	// std::vector<double> zeta(N);
	std::vector<Complex> zeta(N);
	for (int k = 0; k < N; k++) {
		double epsilon_k = cos((k + 0.5) / N * pi);
		epsilon[k] = epsilon_k;
		// zeta[k] = epsilon_k * p[k] + 1i * sqrt(1 - epsilon_k * epsilon_k) * w[k];
		zeta[k] = (epsilon_k * p[k] + 1i * sqrt(1 - epsilon_k * epsilon_k) * w[k]) / pow(1 - epsilon_k * epsilon_k, 2) / Volume;
	}
	// test
	/*std::vector<Complex> zeta_d(N);
	for (int k = 0; k < N; k++) {
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < i; j++) {
				zeta_d[k] += (mu(i, j) * g_mn(epsilon[k], i, j) + mu(j, i) * g_mn(epsilon[k], j, i));
			}
			zeta_d[k] += mu(i, i) * g_mn(epsilon[k], i, i);
		}
	}*/

	/*for (auto&& [e, z1, z2] : std::views::zip(epsilon, zeta, zeta_d)) {
		std::cout << e << " " << z1 << " " << z2 << std::endl;
	}*/

	int cut = N / 24;
	auto E = epsilon | std::views::drop(cut) | std::views::take(N - cut * 2) | std::views::reverse | std::ranges::to<std::vector>();
	auto zeta_real = zeta | std::views::transform([](Complex z) { return z.real(); })
		| std::views::drop(cut) | std::views::take(N - cut * 2) | std::views::reverse | std::ranges::to<std::vector>();
	auto zeta_imag = zeta | std::views::transform([](Complex z) { return z.imag(); })
		| std::views::drop(cut) | std::views::take(N - cut * 2) | std::views::reverse | std::ranges::to<std::vector>();
	// const string folder = "C:/Users/zchende/OneDrive - HKUST Connect/Research/Projects/Tight Binding/test data/";
	// utils::output_to_file(folder + "graphene_LL_zeta003.txt", E, zeta_real, zeta_imag);
	//plt::figure(true);
	//plt::plot(E, zeta_real)->color("blue");
	//plt::hold(true);
	//plt::plot(E, zeta_imag)->color("red");
	////plt::ylim({ -30, 30 });
	//plt::show();

	int E0_i = std::upper_bound(E.begin(), E.end(), 0.0) - E.begin();
	std::cout << E0_i << " " << E.size() << std::endl;
	std::vector<double> zeta_re_int(E.size());
	std::vector<double> zeta_im_int(E.size());
	zeta_re_int[E0_i] = (zeta_real[E0_i] * 3 / 4 + zeta_real[E0_i - 1] / 4) * (E[E0_i] - E[E0_i - 1]) / 2;
	zeta_re_int[E0_i - 1] = -(zeta_real[E0_i] / 4 + zeta_real[E0_i - 1] * 3 / 4) * (E[E0_i] - E[E0_i - 1]) / 2;
	for (int i = E0_i + 1; i < E.size(); i++) {
		zeta_re_int[i] = zeta_re_int[i - 1] + (zeta_real[i] + zeta_real[i - 1]) / 2 * (E[i] - E[i - 1]);
		zeta_im_int[i] = zeta_im_int[i - 1] + (zeta_imag[i] + zeta_imag[i - 1]) / 2 * (E[i] - E[i - 1]);
	}
	for (int i = E0_i - 2; i >= 0; i--) {
		zeta_re_int[i] = zeta_re_int[i + 1] - (zeta_real[i] + zeta_real[i + 1]) / 2 * (E[i + 1] - E[i]);
		zeta_im_int[i] = zeta_im_int[i + 1] - (zeta_imag[i] + zeta_imag[i + 1]) / 2 * (E[i + 1] - E[i]);
	}
	// utils::output_to_file(folder + "graphene_LL_Hxy003.txt", E, zeta_re_int, zeta_im_int);
	plt::figure(true);
	plt::plot(E, zeta_im_int)->color("blue");
	plt::hold(true);
	plt::plot(E, zeta_re_int)->color("red");
	plt::show();
}

double DoS_test(PartialLattice<1>& X, int N, double E)
{
	auto H = X.get_Hamiltonian_bare_sparse<double>();
	auto G = PartialLattice<1>::GreenFunction<double>(N, H, -1.000001, 1.000001);
	double e = G.scale(E, 1.0);
	auto v_x = X.op_velocity(0);
	int R = 4;
	Eigen::MatrixXcd kets = numerical::linalg::bases_of_uniform_ampl_with_random_phases(X.get_total_number(), R);
	auto f = G.imag(kets, kets);
	double DOS = -1.0 / (R) / pi * f(e);
	return DOS;
}

template <int Dim>
auto DoS_test_seq(PartialLattice<Dim>& X, int N, const std::vector<double>& E_list) -> std::vector<double>
{
	// auto H = X.get_Hamiltonian_bare_sparse<double>();
	auto H = X.get_Hamiltonian_periodic_extend_bare_sparse<double>({ 1, 1 });
	auto G = PartialLattice<1>::GreenFunction<double>(N, H, -1.000001, 1.000001);
	// auto v_x = X.op_velocity(0);
	int R = 4;
	Eigen::MatrixXcd kets = numerical::linalg::bases_of_uniform_ampl_with_random_phases(X.get_total_number(), R);
	auto f = G.imag(kets, kets);
	std::vector<double> e_list, DOS_list;
	e_list = E_list;
	// e_list = E_list | std::views::transform([&G](double E) { return G.scale(E, 1.0); }) | std::ranges::to<std::vector>();
	DOS_list = e_list | std::views::transform([&f, &X, R](double e) {
		double DOS = -1.0 / (R) / pi * f(e);
		return DOS;
		}) | std::ranges::to<std::vector>();
	return DOS_list;
}

void generate_V_moments(PartialLattice<1>& X, int N)
{
	auto H = X.get_Hamiltonian_bare_sparse<double>();
	auto G = PartialLattice<1>::GreenFunction<double>(N, H, -1.000001, 1.000001);
	auto v_x = X.op_velocity(0);
	int R = 100;
	Eigen::MatrixXcd kets = numerical::linalg::bases_of_uniform_ampl_with_random_phases(X.get_total_number(), R);
	Eigen::MatrixXcd aR_m(kets.rows(), kets.cols());
	numerical::Chebyshev::Cheb_first_kind_seq_view aR_v(G.scaled_HE, kets);
	std::vector<Eigen::MatrixXcd> kets_R = numerical::Chebyshev::Cheb_first_kind_seq_on_bases(G.scaled_HE, N, kets);
	// left
	Eigen::MatrixXcd v_kets = v_x * kets;
	std::vector<Eigen::MatrixXcd> kets_L = numerical::Chebyshev::Cheb_first_kind_seq_on_bases(G.scaled_HE, N, v_kets);
	const string folder = "D:/documents/OneDrive - HKUST Connect/Research/Projects/Tight Binding/tight_binding_pylib/V_moments.txt";
	ofstream fout(folder);
	for (int i : std::views::iota(0, N)) {
		for (int j : std::views::iota(0, i)) {
			fout << 0. << " ";
		}
		for (int j : std::views::iota(i, N)) {
			double V_nm = numerical::linalg::trace_braket(v_x * kets_L[i], kets_R[j]) / R;
			fout << V_nm << " ";
		}
		cout << i << endl;
		fout << "\n";
	}
	fout.close();
}

void DoS_of_sc()
{
	double t = 1./4;
	double t_ = sqrt(2) * t;
	auto d = models::sawtooth_chain_p(t, t_);
	// auto H_ref = PartialLattice{ d, LatticeVector<1>{100} }.get_Hamiltonian_sparse<double>();
	const int LatX = 100000, ChebN = 10000; // larger ChebN gives better accuracy
	auto X = PartialLattice{ d, LatticeVector<1>{LatX} };
	X.remove_atoms_random("B", 5000);
	/*Eigen::MatrixXcd H_ = X.get_Hamiltonian_periodic_extend({ 0, 0, 0 }, LatticeVector<1>{1});
	Eigen::MatrixXcd Id(H_.rows(), H_.cols());
	Id.setIdentity();*/
	auto G = PartialLattice<1>::GreenFunction<double>(ChebN, X.get_Hamiltonian_sparse<double>(), -1.000001, 1.000001);
	// H_ = G.scale(H_, Eigen::MatrixXcd(Id));
	Eigen::MatrixXcd kets = numerical::linalg::bases_of_uniform_ampl_with_random_phases(X.get_total_number(), 4);

	// timer
	auto start = std::chrono::high_resolution_clock::now();

	auto f = G.imag(kets, kets);

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
#ifdef USE_CUDA
	std::cout << "Using CUDA, elapsed time: " << elapsed.count() << " s\n";
#else
	std::cout << "Using MKL, elapsed time: " << elapsed.count() << " s\n";
#endif

	// utils::output_to_file("D:/documents/OneDrive - HKUST Connect/Research/Projects/Tight Binding/test data/moments.txt", f.moments);

	auto DoS = [&](double E) -> double {
		Complex omega_t = E + 0e-8i; // This imaginary part controls the spread
		double d = -1.0 / 4 / pi * f(omega_t);
		// std::print("doing E = {}, DoS = {}\n", E, d);
		return d;
		// return -1.0 / pi * numerical::linalg::trace_sparse(G.direct_eval(E));
		};
	auto E = utils::linspace(-0.8, +1.2, 2000) | std::ranges::to<std::vector>();
	auto DOS = E | std::views::transform(DoS) | std::ranges::to<std::vector>();
	/*auto [E_f, DOS_f] = f.fft(1);
	auto [sl, sf] = f.slice_range(-0.3, 0.3);
	auto E = E_f | std::views::drop(sl) | std::views::take(sf - sl) | std::ranges::to<std::vector>();
	auto DOS = DOS_f | std::views::drop(sl) | std::views::take(sf - sl) | std::ranges::to<std::vector>();*/

	const string folder = "D:/documents/OneDrive - HKUST Connect/Research/Projects/Tight Binding/test data/";
	////const string folder = "C:/Users/zchende/OneDrive - HKUST Connect/Research/Projects/Tight Binding/test data/";
	utils::output_to_file(folder +
		std::format("SC DoS t={:.2f} sites={} ChebN={} imp={:.2f} no={}.txt", t, LatX, ChebN, 0.05, 0), E, DOS);
	plt::figure(true);
	plt::plot(E, DOS);
	// plt::ylim({ -0.05, 0.35 });
	plt::show();
}

void test_Hofstadter_DoS()
{
	int N = 100;
	double t = 1;
	double Bz = 0.05;
	double lat_a = 1, lat_b = 1;
	// symmetric gauge
	auto A_mag_x = [Bz, lat_b](int i, int j) -> double {
		return -Bz * j * lat_b / 2;
		};
	auto A_mag_y = [Bz, lat_a](int i, int j) -> double {
		return Bz * i * lat_a / 2;
		};
	auto ind = [N](int i, int j) -> int {
		return i * N + j;
		};
	std::vector<Eigen::Triplet<Complex>> triplet_list;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			int ind_ij = ind(i, j);
			if (i < N - 1) {
				triplet_list.push_back({ ind_ij, ind(i + 1, j), -t * exp(1i * A_mag_x(i, j))});
				triplet_list.push_back({ ind(i + 1, j), ind_ij, -t * exp(-1i * A_mag_x(i, j)) });
			}
			if (j < N - 1) {
				triplet_list.push_back({ ind_ij, ind(i, j + 1), -t * exp(1i * A_mag_y(i, j)) });
				triplet_list.push_back({ ind(i, j + 1), ind_ij, -t * exp(-1i * A_mag_y(i, j)) });
			}
		}
	}
	Eigen::SparseMatrix<Complex> H(N * N, N * N);
	H.setFromTriplets(triplet_list.begin(), triplet_list.end());

	int ChebN = 2000;
	auto G = PartialLattice<2>::GreenFunction<Complex>(ChebN, H, -4.0001 * t, 4.0001 * t);
	Eigen::MatrixXcd kets = numerical::linalg::bases_of_uniform_ampl_with_random_phases(N * N, 4);
	auto f = G.imag(kets, kets);
	auto DoS = [&](double E) -> double {
		Complex omega_t = G.scale(E + 0e-8i, 1.0i); // This imaginary part controls the spread
		double d = -1.0 / 4 / pi * f(omega_t);
		return d;
		};
	auto E = utils::linspace(-4 * t, +4 * t, 2000) | std::ranges::to<std::vector>();
	auto DOS = E | std::views::transform(DoS) | std::ranges::to<std::vector>();
	////const string folder = "D:/documents/OneDrive - HKUST Connect/Research/Projects/Tight Binding/test data/";
	const string folder = "C:/Users/zchende/OneDrive - HKUST Connect/Research/Projects/Tight Binding/test data/";
	utils::output_to_file(folder +
		std::format("Hofs DoS t={:.2f} sites={} ChebN={} no={}.txt", t, N * N, ChebN, 0), E, DOS);
	plt::figure(true);
	plt::plot(E, DOS);
	// plt::ylim({ -0.05, 0.35 });
	plt::show();
}

void Graphene_Hofsdater_DoS()
{
	int N = 64;
	int MatN = 2 * N * N;
	double Bz = 0. / 205102. * 2 * pi;
	double t = 1.0;
	double V_rand = 0.0;
	double mu_on_B = 0.0;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<double> dis(0., V_rand);

	auto A_mag = [Bz](Eigen::Vector2d r) -> Eigen::Vector2d {
		return Bz * Eigen::Vector2d{ -r(1), r(0) } / 2;
		};
	//int Moire_N = (int)round(7.8 / (0.142 * sqrt(3)));
	//// Moire_N = 4;
	//double Moire_l = Moire_N * sqrt(3);
	//std::cout << "Moire_l = " << Moire_l << std::endl;
	/*auto A_mag = [Bz, Moire_l](Eigen::Vector2d r) -> Eigen::Vector2d {
		return Bz * Eigen::Vector2d{ 0, Moire_l / (2 * pi) * sin(2 * pi * r(0) / Moire_l)};
		};*/
	/*auto A_mag = [Bz, Moire_l](Eigen::Vector2d r) -> Eigen::Vector2d {
		return Bz * Eigen::Vector2d{ 0, Moire_l / (2 * pi) * sin(2 * pi * r(0) / Moire_l) }
		+ 0.0005 * Eigen::Vector2d{ -r(1), r(0) } / 2;
		};*/
	auto d = UnitCell<2>{
		{
			{  sqrt(3) / 2, 3. / 2, 0 },
			{ -sqrt(3) / 2, 3. / 2, 0 }
		},
		{
			{"A", {0, 0, 0}},
			{"B", {0, 1, 0}}
		}
	};
	auto outer_ind = [N](int i, int j) -> int {
		return i * N + j;
		};
	Eigen::Vector2d delta_1{ 0, -1 };
	Eigen::Vector2d delta_2{  sqrt(3) / 2, 1. / 2 };
	Eigen::Vector2d delta_3{ -sqrt(3) / 2, 1. / 2 };

	std::vector<Eigen::Triplet<Complex>> triplet_list;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			int outer_ind_ij = outer_ind(i, j);
			Eigen::Vector2d r_B = d.atom_position("B", {i, j}).head<2>();
			Eigen::Vector2d A_mag_r_B = A_mag(r_B);
			double phi_1 = A_mag_r_B.dot(delta_1);
			triplet_list.push_back({ outer_ind_ij * 2 + 1, outer_ind_ij * 2, -t * exp(1i * phi_1)});
			triplet_list.push_back({ outer_ind_ij * 2, outer_ind_ij * 2 + 1, -t * exp(-1i * phi_1)});
			/*triplet_list.push_back({ outer_ind_ij * 2 + 1, outer_ind_ij * 2, -t });
			triplet_list.push_back({ outer_ind_ij * 2, outer_ind_ij * 2 + 1, -t });*/

			/*double phi_2 = A_mag_r_B.dot(delta_2);
			triplet_list.push_back({ outer_ind_ij * 2 + 1, outer_ind((i + 1) % N, j) * 2, -t * exp(1i * phi_2) });
			triplet_list.push_back({ outer_ind((i + 1) % N, j) * 2, outer_ind_ij * 2 + 1, -t * exp(-1i * phi_2) });
			double phi_3 = A_mag_r_B.dot(delta_3);
			triplet_list.push_back({ outer_ind_ij * 2 + 1, outer_ind(i, (j + 1) % N) * 2, -t * exp(1i * phi_3) });
			triplet_list.push_back({ outer_ind(i, (j + 1) % N) * 2, outer_ind_ij * 2 + 1, -t * exp(-1i * phi_3) });*/
			if (i < N - 1) {
				double phi_2 = A_mag_r_B.dot(delta_2);
				triplet_list.push_back({ outer_ind_ij * 2 + 1, outer_ind(i + 1, j) * 2, -t * exp(1i * phi_2) });
				triplet_list.push_back({ outer_ind(i + 1, j) * 2, outer_ind_ij * 2 + 1, -t * exp(-1i * phi_2) });
			}
			if (j < N - 1) {
				double phi_3 = A_mag_r_B.dot(delta_3);
				triplet_list.push_back({ outer_ind_ij * 2 + 1, outer_ind(i, j + 1) * 2, -t * exp(1i * phi_3) });
				triplet_list.push_back({ outer_ind(i, j + 1) * 2, outer_ind_ij * 2 + 1, -t * exp(-1i * phi_3) });
			}
			if (V_rand > 0) {
				triplet_list.push_back({ outer_ind_ij * 2, outer_ind_ij * 2, dis(gen) });
				triplet_list.push_back({ outer_ind_ij * 2 + 1, outer_ind_ij * 2 + 1, dis(gen) });
			}
			if (mu_on_B != 0) {
				triplet_list.push_back({ outer_ind_ij * 2 + 1, outer_ind_ij * 2 + 1, mu_on_B });
			}
		}
	}
	Eigen::SparseMatrix<Complex> H(MatN, MatN);
	H.setFromTriplets(triplet_list.begin(), triplet_list.end());

	int ChebN = 2048;
	// auto G = PartialLattice<2>::GreenFunction<Complex>(ChebN, H, -3.0001 * t, 3.0001 * t);
	auto G = PartialLattice<2>::GreenFunction<Complex>(ChebN, H, -4 * t, 4 * t);
	Eigen::MatrixXcd kets = numerical::linalg::bases_of_uniform_ampl_with_random_phases(MatN, 4);
	auto f = G.imag(kets, kets);
	auto DoS = [&](double E) -> double {
		Complex omega_t = G.scale(E + 0e-8i, 1.0i); // This imaginary part controls the spread
		double d = -1.0 / 4 / pi * f(omega_t);
		return d;
		};
	auto E = utils::linspace(-1 * t, +1 * t, ChebN) | std::ranges::to<std::vector>();
	auto DOS = E | std::views::transform(DoS) | std::ranges::to<std::vector>();
	////const string folder = "D:/documents/OneDrive - HKUST Connect/Research/Projects/Tight Binding/test data/";
	const string folder = "C:/Users/zchende/OneDrive - HKUST Connect/Research/Projects/Tight Binding/test data/";
	utils::output_to_file(folder +
		std::format("graphene LL DoS t={:.2f} sites={} ChebN={} no={}.txt", t, MatN, ChebN, 0), E, DOS);
	plt::figure(true);
	plt::plot(E, DOS);
	// plt::ylim({ -0.05, 1.35 });
	plt::show();

	/* int R = 4;
	 Eigen::MatrixXcd kets = numerical::linalg::bases_of_uniform_ampl_with_random_phases(MatN, R);*/

	//PartialLattice<2> X{ d, LatticeVector<2>{N, N} };

	//int ChebNc = 4096;
	////// conductivity_spectrum_test(H, X.op_position(0), X.op_position(0), ChebNc);
	//conductivity_Kubo_Bastin_test(H, X.op_position(0), X.op_position(1), ChebNc, 3 * sqrt(3) / 2);
}

void test_conductivity_Kubo_xy()
{
	auto d = models::graphene;
	double t = 0.0;
	d.add_hopping("A", "A", { 0, 1 }, 1.0i * t);
	d.add_hopping("A", "A", { 1, -1 }, 1.0i * t);
	d.add_hopping("A", "A", { -1, 0 }, 1.0i * t);
	d.add_hopping("B", "B", { 0, 1 }, -1.0i * t);
	d.add_hopping("B", "B", { 1, -1 }, -1.0i * t);
	d.add_hopping("B", "B", { -1, 0 }, -1.0i * t);
	// auto d = models::Square;
	// d.add_hopping("A", 1_NN, 1.0);
	double V_rand = 0.1;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<double> dis(0., V_rand);
	int N = 128;
	PartialLattice<2> X{ d, LatticeVector<2>{ N, N } };
	double Bz = 300. / 205102. * 2 * pi;
	auto A_mag = [Bz](Eigen::Vector3d r) -> Eigen::Vector3d {
		return Bz * Eigen::Vector3d{ -r(1), r(0), 0 } / 2;
		};
	auto V_r = [&dis, &gen](Eigen::Vector3d r) -> double {
		return dis(gen);
		};
	X.gauge_field = A_mag;
	// X.set_scalar_field(V_r);
	auto E = utils::linspace(-1, +1, 2000) | std::ranges::to<std::vector>();
	auto DoS = density_of_states_seq<Complex>(X, 1024, E, -6, 6);
	plt::figure(true);
	plt::plot(E, DoS);
	plt::show();
	// auto H = X.get_Hamiltonian_periodic_extend_bare_sparse<Complex>({ 1, 1 });
	auto H = X.get_Hamiltonian_sparse<Complex>();
	//conductivity_Kubo_Bastin_test(H, X.op_position(0), X.op_position(1), 1024, -6, 6, 3 * sqrt(3) / 2);
}

void Bilayer_Graphene_PMF_DoS()
{
	int N = 1536;
	double Bz0 = 40.;
	double Bz = Bz0 / 205102.0 * 2 * pi; // unit 205102T / 2pi, real B / 2pi = Phi_0 / a_0^2, a_0 is the unit of real space defined in unit cell
	double t = 1.0;
	double gamma_1 = 1.0;
	double V_rand = 0.0;
	double mu_on_B = 0.0;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<double> dis(0., V_rand);

	auto A_mag_uniform = [B_ext = Bz](Eigen::Vector3d r) -> Eigen::Vector3d {
		return B_ext * Eigen::Vector3d{ -r(1), r(0), 0. } / 2;
		};
	int Moire_Nx = (int)round(5 / (0.142 * sqrt(3)));
	double Moire_lx = Moire_Nx * sqrt(3);
	double Moire_lx_ = 0.75 * Moire_lx;
	int Moire_Ny = (int)round(5 / (0.142 * sqrt(3)));
	double Moire_ly = Moire_Ny * sqrt(3);
	/*auto A_mag = [Bz, Moire_l](Eigen::Vector3d r) -> Eigen::Vector3d {
		return Bz * Eigen::Vector3d{ 0., Moire_l / (2 * pi) * sin(2 * pi * r(0) / Moire_l), 0.};
		};*/
	auto Period_f_1 = [](double M_l, double M_l_, double x) -> double {
		x = fmod(x, M_l);
		if (x < 0) x += M_l;
		double f = 0.;
		if (x < M_l_)
			f = -4. / (M_l_ * M_l_) * (pow(x, 3) / 3. - 0.5 * M_l_ * pow(x, 2));
		else {
			double dl = M_l - M_l_;
			x -= M_l_;
			double a1 = M_l_ / dl;
			f = 4 * a1 / (dl * dl) * (pow(x, 3) / 3. - 0.5 * dl * pow(x, 2))
				+ 2. / 3. * a1 * M_l_;
		}
		return f;
	};
	auto Period_f_sin = [](double M_l, double x) {
		return M_l / (2 * pi) * sin(2 * pi * x / M_l);
		};
	/*auto A_mag_TBG = [Bz, Moire_lx](Eigen::Vector3d r) -> Eigen::Vector3d {
		Complex A{ 0., 0. };
		double g = 4. * pi / sqrt(3) / Moire_lx;
		for (int i = 0; i < 6; i++) {
			Eigen::Vector3d G_i{ g * cos(2 * pi * i / 6), g * sin(2 * pi * i / 6), 0. };
			A += exp(1i * (G_i.dot(r) + pi * i / 3.));
		}
		return Bz * Eigen::Vector3d{ A.real(), A.imag(), 0. };
	};*/
	auto A_mag_GBP = [=](Eigen::Vector3d r) -> Eigen::Vector3d {
		double Ay = Bz * Period_f_1(Moire_lx, Moire_lx_, r(1));
		double Ax = -Bz * Period_f_sin(Moire_ly, r(0));
		return Eigen::Vector3d{ Ax, Ay, 0. };
	};
	auto A_mag = [=](Eigen::Vector3d r) -> Eigen::Vector3d {
		return A_mag_GBP(r) + A_mag_uniform(r);
	};
	auto A_mag_TBG = [
		a1 = Moire_lx * Vector3d{ sqrt(3) / 2, 1. / 2, 0. },
		a2 = Moire_lx * Vector3d{ 0, 1, 0. },
		a3 = Moire_lx * Vector3d{ sqrt(3) / 2, -1. / 2, 0. },
		G1 = (4 * pi / sqrt(3) / Moire_lx) * Vector3d{ 1, 0, 1 },
		G3 = (4 * pi / sqrt(3) / Moire_lx) * Vector3d{ cos(2 * pi / 3), sin(2 * pi / 3), 0},
		G5 = (4 * pi / sqrt(3) / Moire_lx) * Vector3d{ cos(4 * pi / 3), sin(4 * pi / 3), 0 },
		A0 = sqrt(3) * Bz / 4 / pi
	](Eigen::Vector3d r) -> Eigen::Vector3d {
		return A0 * (sin(G1.dot(r)) * a2 - sin(G3.dot(r)) * a1 - sin(G5.dot(r)) * a3);
	};
	auto V_TBG = [
		G1 = (4 * pi / sqrt(3) / Moire_lx) * Vector3d{ 1, 0, 0 },
		G3 = (4 * pi / sqrt(3) / Moire_lx) * Vector3d{ cos(2 * pi / 3), sin(2 * pi / 3), 0 },
		G5 = (4 * pi / sqrt(3) / Moire_lx) * Vector3d{ cos(4 * pi / 3), sin(4 * pi / 3), 0 },
		V0 = 0.1
	](Eigen::Vector3d r) -> double {
		return 2 * V0 * (cos(G1.dot(r)) + cos(G3.dot(r)) + cos(G5.dot(r)));
	};
	auto d = UnitCell<2>{
		{
			{ sqrt(3), 0., 0. },
			{ sqrt(3) / 2, 3. / 2, 0. }
		},
		{
			{"A", {0, 0, 0}},
			{"B", {0, 1, 0}}
		}
	}.add_hopping("A", 1_NN, -t);
	/*auto d = UnitCell<2>{
		{
			{  sqrt(3) / 2, 3. / 2, 0 },
			{ -sqrt(3) / 2, 3. / 2, 0 },
		},
		{
			{"A1", {0, 0, 0}},
			{"B1", {0, 1, 0}},
			{"A2", {0, 0, 3}},
			{"B2", {0, -1, 3}}
		}
	};
	d.add_hopping("A1", 1_NN, -t);
	d.add_hopping("A2", 1_NN, -t);
	d.add_hopping("A1", "A2", { 0, 0 }, -gamma_1);*/

	/*auto K = utils::linspace(0, 4, 200) | std::ranges::to<std::vector>();
	auto K_line = K | std::views::transform([](double k) -> Eigen::Vector3d {
		return { k, 0, 0 };
		}) | std::ranges::to<std::vector>();
	auto E_k = d.get_band_dispersion_as_vecs(K_line);
	plt::figure(true);
	for (const auto& e : E_k) {
		plt::plot(K, e);
		plt::hold(true);
	}
	plt::show();*/
	
	auto X = PartialLattice{ d, { Moire_Nx * 20, Moire_Ny * 20 } };
	// auto X = PartialLattice{ d, { 1024, 1024 } };
	// X.gauge_field = A_mag_TBG;
	X.set_scalar_field(V_TBG);
	int ChebNc = 1024;
	auto G = PartialLattice<2>::GreenFunction<Complex>(ChebNc, X.get_Hamiltonian_periodic_extend_bare_sparse<Complex>({ 1, 1 }), -4.0001 * t, 4.0001 * t);
	// auto G = PartialLattice<2>::GreenFunction<Complex>(ChebNc, X.get_Hamiltonian_sparse<Complex>(), -4.0001 * t, 4.0001 * t);
	 /*auto G = PartialLattice<2>::GreenFunction<double>(ChebNc, 
		X.get_Hamiltonian_periodic_extend_bare_sparse({1, 1}), -4.0001 * t, 4.0001 * t);*/
	Eigen::MatrixXcd kets = numerical::linalg::bases_of_uniform_ampl_with_random_phases(G.scaled_HE.rows(), 4);
	auto f = G.imag(kets, kets);
	auto DoS = [&](double E) -> double {
		Complex omega_t = G.scale(E + 0e-8i, 1.0i); // This imaginary part controls the spread
		double d = -1.0 / 4 / pi * f(omega_t);
		return d;
		};
	auto E = utils::linspace(-0.7, +0.7, 8192) | std::ranges::to<std::vector>();
	auto DOS = E | std::views::transform(DoS) | std::ranges::to<std::vector>();
	// const string folder = "D:/documents/OneDrive - HKUST Connect/Research/Projects/Tight Binding/test data/";
	const string folder = "C:/Users/zchende/OneDrive - HKUST Connect/Research/Projects/Tight Binding/test data/";
	/*utils::output_to_file(folder +
		std::format("G-BP with ext B=0 DoS B={} Moire_N={} ChebN={} no={}.txt", Bz0, Moire_N, ChebNc, 0), E, DOS);*/
	utils::output_to_file(folder +
		std::format("Graphene LL xy DoS B={} ChebN={} no={}.txt", Bz0, ChebNc, 0), E, DOS);
	plt::figure(true);
	plt::plot(E, DOS);
	// plt::ylim({ -0.05, 1.35 });
	plt::show();
}

void test_graphene_PBC()
{
	double t = 1.0;
	double gamma_1 = 0.1;
	auto d = UnitCell<2>{
		{
			{ sqrt(3), 0., 0. },
			{ sqrt(3) / 2, 3. / 2, 0. }
		},
		{
			{"A", {0, 0, 0}},
			{"B", {0, 1, 0}}
		}
	}.add_hopping("A", 1_NN, -t);
	/*auto d = UnitCell<2>{
		{
			{ sqrt(3), 0., 0. },
			{ sqrt(3) / 2, 3. / 2, 0. }
		},
		{
			{"A1", {0, 0, 0}},
			{"B1", {0, 1, 0}},
			{"A2", {0, 0, 3}},
			{"B2", {0, -1, 3}}
		}
	}.add_hopping("A1", 1_NN, -t)
	 .add_hopping("A2", 1_NN, -t)
	 .add_hopping("A1", "A2", { 0, 0 }, -gamma_1);*/
	double K_point = 4 * pi / (3 * sqrt(3));

	// d.draw_k_space();
	// d.draw_band_dispersion();

	int Moire_Nx = (int)round(3. / (0.142 * sqrt(3)));
	int Moire_Ny = (int)round(3. / (0.142 * sqrt(3)));
	Moire_Ny = 8;
	Moire_Nx = 8;
	double Moire_lx = Moire_Nx * sqrt(3);
	double Moire_lx_ = Moire_lx * 0.75;
	double Moire_ly = Moire_Ny * sqrt(3);
	double Moire_ly_ = Moire_ly * 0.75;
	double Bz = 40. / 205102.0 * 2 * pi;
	/*auto A_mag = [Bz, Moire_l](Eigen::Vector3d r) -> Eigen::Vector3d {
		return Bz * Eigen::Vector3d{ 0., Moire_l / (2 * pi) * sin(2 * pi * r(0) / Moire_l), 0.};
		};*/
	auto Period_f_1 = [](double M_l, double M_l_, double x) -> double {
		x = fmod(x, M_l);
		if (x < 0) x += M_l;
		double f = 0.;
		if (x < M_l_)
			f = -4. / (M_l_ * M_l_) * (pow(x, 3) / 3. - 0.5 * M_l_ * pow(x, 2));
		else {
			double dl = M_l - M_l_;
			x -= M_l_;
			double a1 = M_l_ / dl;
			f = 4 * a1 / (dl * dl) * (pow(x, 3) / 3. - 0.5 * dl * pow(x, 2))
				+ 2. / 3. * a1 * M_l_;
		}
		return f;
	};
	auto Period_f_sin = [](double M_l, double x) {
		return M_l / (2 * pi) * sin(2 * pi * x / M_l);
		};

	auto A_mag = [=, &Period_f_1](Eigen::Vector3d r) -> Eigen::Vector3d {
		return Eigen::Vector3d{ -Bz * Period_f_sin(Moire_ly, r(1)), 
			Bz * Period_f_1(Moire_lx, Moire_lx_, r(0) - r(1) / sqrt(3)), 0.};
		};
	auto V_r = [=, v = 2 * Bz, &Period_f_1](Eigen::Vector3d r) -> double {
		return v * Period_f_1(Moire_lx, Moire_lx_, r(0));
		};
	double Bz0 = 2 * pi / Moire_lx;
	// std::cout << "Bz0 = " << Bz0 * 205102.0 / (2 * pi) << std::endl;
	auto A_mag_uniform = [Bz = Bz0, Moire_lx](Eigen::Vector3d r) -> Eigen::Vector3d {
		return Bz * Eigen::Vector3d{ 0., -r(0), 0.};
		};
	auto V_TBG = [
		G1 = (4 * pi / sqrt(3) / Moire_lx) * Vector3d{ 1, 0, 0 },
		G3 = (4 * pi / sqrt(3) / Moire_lx) * Vector3d{ cos(2 * pi / 3), sin(2 * pi / 3), 0 },
		G5 = (4 * pi / sqrt(3) / Moire_lx) * Vector3d{ cos(4 * pi / 3), sin(4 * pi / 3), 0 },
		V0 = 0.1
	](Eigen::Vector3d r) -> double {
		return 2 * V0 * (cos(G1.dot(r)) + cos(G3.dot(r)) + cos(G5.dot(r)));
	};
	
	d.add_hopping("A", On_Site, 0.).add_hopping("B", On_Site, 0.);
	auto X = PartialLattice{ d, { Moire_Nx, Moire_Ny } };
	// X.gauge_field = A_mag;
	X.set_scalar_field(V_TBG);

	auto K = utils::linspace(K_point - K_point / 1, K_point, 1000) | std::ranges::to<std::vector>();
	auto Kx = utils::linspace(-0.5, 0.5, 201, true) | std::ranges::to<std::vector>();
	auto Ky = utils::linspace(-0.5, 0.5, 201, true) | std::ranges::to<std::vector>();
	
	auto K_line = K | std::views::transform([=](double k) -> Eigen::Vector3d {
		return { k, 0, 0 };
		}) | std::ranges::to<std::vector>();

	std::vector<Eigen::Vector3d> K_samples;
	// K_point = 2.0;
	for (auto&& kx : Kx) {
		for (auto&& ky : Ky) {
			K_samples.push_back({ kx + K_point, ky, 0 });
		}
	}
	// auto E_k = d.get_band_dispersion_as_vecs(K_line);
	const string filename = "C:/Users/zchende/OneDrive - HKUST Connect/Research/Projects/Tight Binding/test data/LL_B40_xy_dense.txt";
	// const string filename = "D:/documents/OneDrive - HKUST Connect/Research/Projects/Tight Binding/test data/LL_B0_xy.txt";
	std::vector<std::vector<double>> E_k(X.get_total_number());
	for (auto&& k : K_line) {
		auto H_k = X.get_Hamiltonian_periodic_extend(k, { 1, 1 });
		 Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(H_k);
		 const Eigen::VectorXd& e = es.eigenvalues().real();
		/*auto e = numerical::linalg::sparse_matrix_n_eigen_closest_to(
			numerical::linalg::complex_matrix_convert_to_real_sparse(H_k.sparseView()),
			0.001, 24);*/
		cout << k(0) << " " << k(1) << "\n";
		for (int i = 0; i < e.size(); i++) {
			E_k[i].push_back(e(i));
		}
	}
	utils::output_to_file(filename, K, utils::transposed_nested_vector(E_k));
	plt::figure(true);
	for (const auto& e : E_k) {
		plt::plot(K, e)->color("black");
		plt::hold(true);
	}
	// plt::plot({ K_point, K_point }, { -3, 3 })->color("red").line_style("--");
	// plt::ylim({ -0.2, 0.2 });
	plt::show();
}

void test_lanczos() {
	Eigen::MatrixXcd H(8, 8);
	H = Eigen::MatrixXcd::Random(8, 8);
	Eigen::MatrixXcd H_k = H + H.adjoint();
	std::cout << H_k << std::endl;
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(H_k);
	const Eigen::VectorXd& e_ref = es.eigenvalues().real();
	std::cout << e_ref.transpose() << std::endl;
	auto e = numerical::linalg::sparse_matrix_n_eigen_closest_to(
		numerical::linalg::complex_matrix_convert_to_real_sparse(H_k.sparseView()),
		0.01, 3);
	std::cout << e.real().transpose() << std::endl;
}

void test_1D_moire() {
	auto p = models::Chain;
	p.add_hopping("A", 1_NN, -1.0);
	// p.add_hopping("A", On_Site, 0.0); // to enable scalar field
	int LN = 8;
	double Lm = LN * 1.0;
	auto V = [V0 = 1, Lm](Vector3d r) -> double {
		return V0 * cos(2 * pi * r(0) / Lm);
	};
	auto X = PartialLattice<1>{ p, LatticeVector<1>{ LN } };
	X.set_scalar_field(V);
	auto K = utils::linspace(-pi, pi, 1000) | std::ranges::to<std::vector>();
	auto K_line = K | std::views::transform([](double k) -> Eigen::Vector3d {
		return { k, 0, 0 };
		}) | std::ranges::to<std::vector>();
	std::vector<std::vector<double>> E_k(X.get_total_number());
	for (auto&& k : K_line) {
		auto H_k = X.get_Hamiltonian_periodic_extend(k, LatticeVector<1>{ 1 });
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(H_k);
		const Eigen::VectorXd& e = es.eigenvalues().real();
		cout << k(0) << "\n";
		for (int i = 0; i < e.size(); i++) {
			E_k[i].push_back(e(i));
		}
	}
	plt::figure(true);
	for (const auto& e : E_k) {
		plt::plot(K, e)->color("black");
		plt::hold(true);
	}
	plt::show();
}