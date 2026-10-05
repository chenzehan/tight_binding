#ifndef PYTHON_LIB

#define _SILENCE_CXX23_DENORM_DEPRECATION_WARNING
// #define USE_CUDA
#define EIGEN_USE_MKL_ALL
#undef EIGEN_USE_MKL_ALL
#include <iostream> 

#include "unit_cell.h"
#include "lattice.h"
#include "models.h"

#include <cmath>
#include <numbers>
#include <print>
#include <chrono>
using namespace std;
using numbers::pi;

#include "snippets.h"
int main()
{
	//auto p = models::Square;
	//double t = 0.1;
	//p.add_hopping("A", 1_NN, t);
	//int Size = 400;
	//PartialLattice<2> X{ p, LatticeVector<2>{ Size, Size } };
	///*auto E = utils::linspace(-1, +1, 2000) | std::ranges::to<std::vector>();
	//auto DoS = DoS_test_seq(X, 1024, E);
	//plt::figure(true);
	//plt::plot(E, DoS);
	//plt::show();*/
	//// Eigen::MatrixXcd kets = numerical::linalg::bases_of_uniform_ampl_with_random_phases(X.get_total_number(), Size);
	//int ChebN = 1024;
	//// conductivity_spectrum_test(X.get_Hamiltonian_sparse<Complex>(), X.op_position(0), X.op_position(0), ChebN);
	//conductivity_Kubo_Bastin_test(X.get_Hamiltonian_sparse<Complex>(), X.op_position(0), X.op_position(0), ChebN);

	// Bilayer_Graphene_PMF_DoS();
	// DoS_of_Graphene();
	// test_graphene_PBC();
	// test_lanczos();
	// test_1D_moire();
	test_conductivity_Kubo_xy();
	
	return 0;
}

#endif