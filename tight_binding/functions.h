#pragma once

#include "lattice.h"

auto density_of_states_function(auto&& G) {
	Eigen::MatrixXcd kets = numerical::linalg::bases_of_uniform_ampl_with_random_phases(G.scaled_HE.rows(), 4);
	auto f = G.imag(kets, kets);
	return [f, &G](double E) -> double {
		Complex omega_t = G.scale(E + 0e-8i, 1.0i); // This imaginary part controls the spread
		double d = -1.0 / 4 / pi * f(omega_t);
		return d;
		};
}

template <ScalarType Scalar, int Dim>
auto density_of_states_seq(PartialLattice<Dim>& X, int N, const std::vector<double>& E_list, double E_min, double E_max) -> std::vector<double>
{
	// auto H = X.get_Hamiltonian_bare_sparse<double>();
	auto H = X.get_Hamiltonian_periodic_extend_bare_sparse<Scalar>({ 1, 1 });
	auto G = PartialLattice<1>::GreenFunction<Scalar>(N, H, E_min - 1e-8, E_max + 1e-8);
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