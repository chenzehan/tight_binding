#pragma once

#define _SILENCE_CXX23_DENORM_DEPRECATION_WARNING
#include <Eigen/Sparse>
#include <fftw3.h>
#include <iostream>
#include <print>
#include <random>
#include <algorithm>

// The Spectra library included after the matplot library may cause a conflict
// the conflict is possibly due to the macro of min and max in the Spectra library
// This should be fixed if we undef the min and max macros after including the Spectra library
#include "numerical.h"
#include "unit_cell.h"
#ifdef USE_CUDA
#include "numerics_cuda.cuh"
#endif

using numerical::ScalarType;

template <int Dim> requires (is_lattice_dim(Dim))
class PartialLattice {
public:
	PartialLattice(const UnitCell<Dim>& cell, LatticeVector<Dim> span_size);

	// index system: i_v: lattice vector index, i_a: atom index, i_b: bare index, i_c: compressed true index
	int index_bare(LatticeVector<Dim> i_v, int i_a) const { return i_v.dot(span_s) + i_a; };
	LatticeVector<Dim> index_bare_to_i_v(int i_b) const { return i_b / span_s.array(); };
	int index_comp(int i_b) const { return index_c[i_b]; };
	int index_comp(LatticeVector<Dim> i_v, int i_a) const { return index_c[index_bare(i_v, i_a)]; }
	int index_comp_to_i_b(int i_c) const;
	bool index_inside_lattice(LatticeVector<Dim> i) const;
	bool index_inside_lattice_periodic_extend(LatticeVector<Dim> i, LatticeVector<Dim> extend) const;
	bool index_bare_is_removed(int i_b) const { return index_c[i_b] == -1; };

	Eigen::MatrixXcd get_Hamiltonian_bare() const;
	Eigen::MatrixXcd get_Hamiltonian() const;
	Eigen::MatrixXcd get_Hamiltonian_periodic_extend(Vector3d k, LatticeVector<Dim> extend) const;
	template <ScalarType Scalar>
	Eigen::SparseMatrix<Scalar> get_Hamiltonian_periodic_extend_bare_sparse(LatticeVector<Dim> extend);
	template <ScalarType Scalar>
	Eigen::SparseMatrix<Scalar> get_Hamiltonian_bare_sparse();
	template <ScalarType Scalar>
	Eigen::SparseMatrix<Scalar> get_Hamiltonian_sparse();
	Eigen::MatrixXcd get_Hamiltonian_dense();

	Eigen::SparseMatrix<double> op_position(int axis);
	Eigen::SparseMatrix<double> op_velocity(int axis);

	int get_total_number() const { return total_number; };
	int get_cell_number() const { return cell_number; };

	// void remove_atom(const std::string& atom_name, LatticeVector<Dim> i);
	void remove_atoms(const std::string& atom_name, const std::vector<LatticeVector<Dim>>& i);
	void remove_atoms_random(const std::string& atom_name, int n);

	static std::vector<int> generate_random_int_sequence(int n, int N);
	// std::vector<lattice_vector<Dim>> generate_random_index_sequence(int n) const;
	// struct BlockView;

	// only real supported. complex should be cast to real
	template<ScalarType Scalar> // TODO: implement complex version
	struct GreenFunction; // finite imaginary part Chebyshev expansion
	struct GreenFunction_Dense;

	/*GreenFunction get_GreenFunction(int N, const Eigen::SparseMatrix<double>& H_ref, double epsilon = 0.01) {
		return GreenFunction(N, get_Hamiltonian_sparse<double>(), H_ref, epsilon);
	}*/

	inline static bool auto_info = true;

	std::function<Eigen::Vector3d(Eigen::Vector3d)> gauge_field;
	void set_scalar_field(auto&& f);

private:
	void index_comp_construct();
	void index_comp_init();
	auto hopping_of_each_atom_view_filter(const auto& filter) const;
	auto hopping_of_each_atom_view() const;
	// auto index_sequence_view() const;
	template <int... ints>
	auto index_sequence_tuple(std::integer_sequence<int, ints...>) const
	{
		return std::views::cartesian_product(std::views::iota(0, span_v(ints))...);
	}
	auto index_sequence_view() const
	{
		return index_sequence_tuple(std::make_integer_sequence<int, Dim>{})
			| std::views::transform([](auto&& t) { return UnitCell<Dim>::get_vector_from_tuple(t); });
	}

	UnitCell<Dim> cell_;
	LatticeVector<Dim> span_v, span_s;
	int cell_number, total_number;

	std::unordered_map<std::string, std::vector<int>> removed_atoms;
	std::vector<int> index_c;

	// you need a "typename" here
	std::unordered_map<Hopping<Dim>, std::function<Complex()>, typename UnitCell<Dim>::HoppingHash> alter_hoppings;
	std::function<double(Eigen::Vector3d)> scalar_field;
};

template<int Dim> requires (is_lattice_dim(Dim))
inline PartialLattice<Dim>::PartialLattice(const UnitCell<Dim>& cell, LatticeVector<Dim> span_size)
	: cell_(cell), span_v(span_size), cell_number(1)
{
	for (int i = 0; i < Dim; i++) cell_number *= span_v(i);
	total_number = cell_number * cell.get_atom_number();
	// generate span_s
	span_s(Dim - 1) = cell_.get_atom_number();
	if constexpr (Dim > 1)
		for (int i = Dim - 2; i >= 0; i--) span_s(i) = span_s(i + 1) * span_v(i + 1);
	if (auto_info) std::cout << "PartialLattice generated with size: " << span_v.transpose()
							 << " and total number: " << total_number << std::endl;
}

template<int Dim> requires (is_lattice_dim(Dim))
inline int PartialLattice<Dim>::index_comp_to_i_b(int i_c) const
{
	if (i_c >= index_c.size() || i_c < 0) {
		std::cerr << "index_comp_to_i_b: Index out of range." << std::endl;
		return -1;
	}
	int i = index_c[i_c];
	for (; i != i_c; i += index_c[i] < 0 ? 1 : (i - index_c[i]))
		if (i >= index_c.size()) {
			std::cerr << "index_comp_to_i_b: Index out of range." << std::endl;
			return -1;
		}
	return i;
}

template<int Dim> requires (is_lattice_dim(Dim))
inline bool PartialLattice<Dim>::index_inside_lattice(LatticeVector<Dim> i) const
{
	auto&& Cond_1 = i.array() >= 0;
	auto&& Cond_2 = i.array() < span_v.array();
	return std::ranges::all_of(Cond_1 && Cond_2, [](auto&& x) { return x; });
}

template<int Dim> requires (is_lattice_dim(Dim))
inline bool PartialLattice<Dim>::index_inside_lattice_periodic_extend(LatticeVector<Dim> i, LatticeVector<Dim> extend) const
{
	auto&& Cond_1 = i.array() >= 0;
	auto&& Cond_2 = i.array() < span_v.array();
	return std::ranges::all_of(extend.array().cast<bool>() || (Cond_1 && Cond_2), [](auto&& x) { return x; });
}

template<int Dim> requires (is_lattice_dim(Dim))
inline Eigen::MatrixXcd PartialLattice<Dim>::get_Hamiltonian_bare() const
{
	Eigen::MatrixXcd H(total_number, total_number);
	H.setZero();
	auto each_atom_view = hopping_of_each_atom_view();
	for (auto&& [i_v, to_v, from, to, t] : each_atom_view) {
		int row = index_bare(i_v, from);
		int col = index_bare(to_v, to);
		H(row, col) = t;
		if (row == col) continue;
		H(col, row) = std::conj(t);
	}
	return H;
}

template<int Dim> requires (is_lattice_dim(Dim))
inline Eigen::MatrixXcd PartialLattice<Dim>::get_Hamiltonian() const
{
	Eigen::MatrixXcd H(total_number, total_number);
	H.setZero();
	auto each_atom_view = hopping_of_each_atom_view();
	auto view_t = hopping_of_each_atom_view_filter([](auto&& t) { return true; });
	for (auto&& [i_v, to_v, from, to, t_] : each_atom_view) {
		int row = index_bare(i_v, from);
		int col = index_bare(to_v, to);
		auto t = t_;
		if (gauge_field) {
			auto hop_v = cell_.displacement_of_hopping({ from, to, to_v - i_v });
			t *= exp(1.0i * gauge_field(cell_.atom_position(from, i_v)).dot(hop_v));
		}
		if (scalar_field && row == col) t += scalar_field(cell_.atom_position(from, i_v));
		H(row, col) = t;
		if (row == col) continue;
		H(col, row) = std::conj(t);
	}
	return H;
}

template<int Dim> requires (is_lattice_dim(Dim))
inline Eigen::MatrixXcd PartialLattice<Dim>::get_Hamiltonian_periodic_extend(Vector3d k, LatticeVector<Dim> extend) const
{
	Eigen::MatrixXcd H = get_Hamiltonian();
	auto mod_arr = [](auto&& x, auto&& y) { return x - (y * (x / y)); };
	auto mod_arr_u = [&mod_arr](auto&& x, auto&& y) { return mod_arr(mod_arr(x, y) + y, y); };
	auto each_atom_view = hopping_of_each_atom_view_filter([this, &extend](auto&& tp) {
		auto&& to_v = std::get<1>(tp);
		return index_inside_lattice_periodic_extend(to_v, extend) && !index_inside_lattice(to_v);
		}
	);
	for (auto&& [i_v, to_v, from, to, t_] : each_atom_view) {
		LatticeVector<Dim> to_v_m = mod_arr_u(to_v.array(), span_v.array()).matrix();
		Eigen::Vector3d hop_v = cell_.displacement_of_hopping({ from, to, to_v - i_v }); // hp = { from, to, to_v - i_v }
		Eigen::Vector3d k_ = k;
		if (gauge_field) k_ += gauge_field(cell_.atom_position(from, i_v));
		auto t = t_ * std::exp(Complex(0., k_.dot(hop_v)));
		int row = index_bare(i_v, from);
		int col = index_bare(to_v_m, to);
		if (scalar_field && row == col) t += scalar_field(cell_.atom_position(from, i_v));
		H(row, col) += t;
		H(col, row) += std::conj(t);
	}
	return H;
}

template<int Dim> requires (is_lattice_dim(Dim))
template <ScalarType Scalar>
inline Eigen::SparseMatrix<Scalar>
PartialLattice<Dim>::get_Hamiltonian_periodic_extend_bare_sparse(LatticeVector<Dim> extend)
{
	std::vector<Eigen::Triplet<Scalar>> triplet_list;
	auto mod_arr = [](auto&& x, auto&& y) { return x - (y * (x / y)); };
	auto mod_arr_u = [&mod_arr](auto&& x, auto&& y) { return mod_arr(mod_arr(x, y) + y, y); };
	auto each_atom_view = hopping_of_each_atom_view_filter([this, &extend](auto&& tp) {
		auto&& to_v = std::get<1>(tp);
		return index_inside_lattice_periodic_extend(to_v, extend) || index_inside_lattice(to_v);
		}
	);
	for (auto&& [i_v, to_v, from, to, t_] : each_atom_view) {
		LatticeVector<Dim> to_v_m = mod_arr_u(to_v.array(), span_v.array()).matrix();
		// Vector3d hop_v = cell_.displacement_of_hopping({ from, to, to_v - i_v }); // hp = { from, to, to_v - i_v }
		int row = index_bare(i_v, from);
		int col = index_bare(to_v_m, to);
		auto t = t_;
		if (gauge_field) {
			auto hop_v = cell_.displacement_of_hopping({ from, to, to_v - i_v });
			t *= exp(1.0i * gauge_field(cell_.atom_position(from, i_v)).dot(hop_v));
		}
		if (scalar_field && row == col) t += scalar_field(cell_.atom_position(from, i_v));
		if constexpr (std::is_same_v<Scalar, Complex>) {
			triplet_list.push_back(Eigen::Triplet<Scalar>(row, col, t));
			if (row == col) continue;
			triplet_list.push_back(Eigen::Triplet<Scalar>(col, row, std::conj(t)));
		}
		else {
			if (abs(t.imag()) > 1e-10)
				std::cerr << "Warning: Imaginary part of hopping is discarded." << std::endl;
			triplet_list.push_back(Eigen::Triplet<Scalar>(row, col, t.real()));
			if (row == col) continue;
			triplet_list.push_back(Eigen::Triplet<Scalar>(col, row, t.real()));
		}
	}
	Eigen::SparseMatrix<Scalar> H(total_number, total_number);
	H.setFromTriplets(triplet_list.begin(), triplet_list.end());
	if (auto_info)
		std::cout << "Periodic Sparse Hamiltonian generated with number of nonzero: " << H.nonZeros() << std::endl;
	return H;
}

// Get the complete Hamiltonian including impurities in sparse form
// This is recommended to be used for large systems
template<int Dim> requires (is_lattice_dim(Dim))
template <ScalarType Scalar>
inline Eigen::SparseMatrix<Scalar> PartialLattice<Dim>::get_Hamiltonian_sparse()
{
	if (index_c.empty()) index_comp_init();
	std::vector<Eigen::Triplet<Scalar>> triplet_list;
	auto each_atom_view = hopping_of_each_atom_view_filter([this](auto&& tp) {
		auto&& [i_v, to_v, from, to, t] = tp;
		return index_inside_lattice(to_v) 
			&& !index_bare_is_removed(index_bare(i_v, from))
			&& !index_bare_is_removed(index_bare(to_v, to));
		});
	for (auto&& [i_v, to_v, from, to, t_] : each_atom_view) {
		int row = index_comp(i_v, from);
		int col = index_comp(to_v, to);
		auto t = t_;
		if (!alter_hoppings.empty()) {
			auto hp = Hopping<Dim>{ from, to, to_v - i_v };
			if (alter_hoppings.contains(hp)) t = alter_hoppings[hp]();
		}
		if (gauge_field) {
			auto hop_v = cell_.displacement_of_hopping({ from, to, to_v - i_v });
			t *= exp(1.0i * gauge_field(cell_.atom_position(from, i_v)).dot(hop_v));
		}
		if (scalar_field && row == col) t += scalar_field(cell_.atom_position(from, i_v));
		if constexpr (std::is_same_v<Scalar, Complex>) {
			triplet_list.push_back(Eigen::Triplet<Scalar>(row, col, t));
			if (row == col) continue;
			triplet_list.push_back(Eigen::Triplet<Scalar>(col, row, std::conj(t)));
		}
		else {
			if (abs(t.imag()) > 1e-10)
				std::cerr << "Warning: Imaginary part of hopping is discarded." << std::endl;
			triplet_list.push_back(Eigen::Triplet<Scalar>(row, col, t.real()));
			if (row == col) continue;
			triplet_list.push_back(Eigen::Triplet<Scalar>(col, row, t.real()));
		}
	}
	Eigen::SparseMatrix<Scalar> H(total_number, total_number);
	H.setFromTriplets(triplet_list.begin(), triplet_list.end());
	if (auto_info)
		std::cout << "Sparse Hamiltonian generated with number of nonzero: " << H.nonZeros() << std::endl;
	return H;
}

template<int Dim> requires (is_lattice_dim(Dim))
inline Eigen::MatrixXcd PartialLattice<Dim>::get_Hamiltonian_dense()
{
	return get_Hamiltonian_sparse().toDense();
}

// bare version
template<int Dim> requires (is_lattice_dim(Dim))
inline Eigen::SparseMatrix<double> PartialLattice<Dim>::op_position(int axis)
{
	if (axis >= Dim) {
		std::cerr << "Error: axis out of range." << std::endl;
		axis = 0;
	}
	if (index_c.empty()) index_comp_init();
	std::vector<Eigen::Triplet<double>> triplet_list;
	auto index_view = index_sequence_view();
	for (LatticeVector<Dim>&& i : index_view) {
		for (auto&& [a, a_pos] : std::views::enumerate(cell_.atom_pos)) {
			int row = index_comp(i, a);
			if (row == -1) continue; // removed atom (index_c[i] == -1)
			int col = row;
			double pos = a_pos(axis) + cell_.displacement_of_lattice(i)(axis);
			triplet_list.push_back(Eigen::Triplet<double>(row, col, pos));
		}
	}
	Eigen::SparseMatrix<double> op(total_number, total_number);
	op.setFromTriplets(triplet_list.begin(), triplet_list.end());
	return op;
}

// discard i (imaginary unit) for now
template<int Dim> requires (is_lattice_dim(Dim))
inline Eigen::SparseMatrix<double> PartialLattice<Dim>::op_velocity(int axis)
{
	auto op_x = op_position(axis);
	auto H = get_Hamiltonian_sparse<double>();
	return (op_x * H - H * op_x);
}

template<int Dim> requires (is_lattice_dim(Dim))
template <ScalarType Scalar>
inline Eigen::SparseMatrix<Scalar> PartialLattice<Dim>::get_Hamiltonian_bare_sparse()
{
	auto index_view = index_sequence_view();
	std::vector<Eigen::Triplet<Scalar>> triplet_list;
	for (LatticeVector<Dim>&& i : index_view) {
		for (auto&& [hp, t] : cell_.hopping_view()) {
			auto&& [from, to, lat_v] = hp;
			auto&& to_i = i + lat_v;
			if (!index_inside_lattice(to_i)) continue;
			int row = index_bare(i, from);
			int col = index_bare(to_i, to);
			if constexpr (std::is_same_v<Scalar, Complex>) {
				triplet_list.push_back(Eigen::Triplet<Scalar>(row, col, t));
				if (row == col) continue;
				triplet_list.push_back(Eigen::Triplet<Scalar>(col, row, std::conj(t)));
			}
			else {
				if (abs(t.imag()) > 1e-10)
					std::cerr << "Warning: Imaginary part of hopping is discarded." << std::endl;
				triplet_list.push_back(Eigen::Triplet<Scalar>(row, col, t.real()));
				if (row == col) continue;
				triplet_list.push_back(Eigen::Triplet<Scalar>(col, row, t.real()));
			}
		}
	}
	Eigen::SparseMatrix<Scalar> H(total_number, total_number);
	H.setFromTriplets(triplet_list.begin(), triplet_list.end());
	return H;
}

//template<int Dim> requires (is_lattice_dim(Dim))
//inline void PartialLattice<Dim>::remove_atom(const std::string& atom_name, LatticeVector<Dim> i)
//{
//	if (!cell_.atom_exists(atom_name)) {
//		std::cerr << "Atom " << atom_name << " does not exist in the unit cell." << std::endl;
//		return;
//	}
//	removed_atoms[atom_name].push_back(index_bare(i, cell_.get_atom_index(atom_name)));
//}

template<int Dim> requires (is_lattice_dim(Dim))
inline void PartialLattice<Dim>::remove_atoms(const std::string& atom_name, const std::vector<LatticeVector<Dim>>& i)
{
	if (!cell_.atom_exists(atom_name)) {
		std::cerr << "Atom " << atom_name << " does not exist in the unit cell." << std::endl;
		return;
	}
	int atom_index = cell_.get_atom_index(atom_name);
	auto&& to_int = i
		| std::views::transform([this, atom_index](auto&& x) { return index_bare(x, atom_index); });
	removed_atoms[atom_name].insert(removed_atoms[atom_name].end(), to_int.begin(), to_int.end());
	// THIS can be optimized, since no need to compute the whole index_c every time, copilot says XD
	index_comp_construct();
}

template<int Dim> requires (is_lattice_dim(Dim))
inline void PartialLattice<Dim>::remove_atoms_random(const std::string& atom_name, int n)
{
	if (!cell_.atom_exists(atom_name)) {
		std::cerr << "Atom " << atom_name << " does not exist in the unit cell." << std::endl;
		return;
	}
	int atom_index = cell_.get_atom_index(atom_name);
	int cell_atom_number = cell_.get_atom_number();
	auto&& to_int = generate_random_int_sequence(n, cell_number)
		| std::views::transform([=](auto&& x) { return x * cell_atom_number + atom_index; });
	removed_atoms[atom_name].insert(removed_atoms[atom_name].end(), to_int.begin(), to_int.end());
	index_comp_construct();
}

template<int Dim> requires (is_lattice_dim(Dim))
inline std::vector<int> PartialLattice<Dim>::generate_random_int_sequence(int n, int N)
{
	if (n > N) {
		std::cerr << "The random selection is larger than the total number." << std::endl;
		return {};
	}
	if (n > N * 0.3) {
		std::vector<int> sequence(N);
		std::iota(sequence.begin(), sequence.end(), 0);
		std::random_device rd;
		std::mt19937 gen(rd());
		std::shuffle(sequence.begin(), sequence.end(), gen);
		return std::vector(sequence.begin(), sequence.begin() + n);
	}
	else {
		std::unordered_set<int> sequence_set;
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<int> dis(0, N - 1);
		while (sequence_set.size() < n) {
			sequence_set.insert(dis(gen));
		}
		return std::vector(sequence_set.begin(), sequence_set.end());
	}
}

template<int Dim> requires (is_lattice_dim(Dim))
inline void PartialLattice<Dim>::index_comp_construct()
{
	index_c = std::vector<int>(total_number);
	for (auto&& [atom_name, index_list] : removed_atoms) {
		for (int i : index_list) index_c[i] = -1;
	}
	int count = 0;
	for (int i = 0; i < total_number; i++) {
		if (index_c[i] != -1) index_c[i] = count++;
	}
}

template<int Dim> requires (is_lattice_dim(Dim))
inline void PartialLattice<Dim>::index_comp_init()
{
	index_c = std::vector<int>(total_number);
	std::iota(index_c.begin(), index_c.end(), 0);
}

// an iteration view of every possible hopping terms with filter condition
// return as [i_v, to_v, from, to, t]; i_v, to_v: LatticeVector<Dim>; from, to: int; t: complex
template<int Dim> requires (is_lattice_dim(Dim))
inline auto PartialLattice<Dim>::hopping_of_each_atom_view_filter(const auto& filter) const
{
	auto hopping_view = cell_.hopping_view();
	if (std::empty(hopping_view)) std::cerr << "Warning: no hopping is added!" << std::endl;
	auto index_view = index_sequence_view();
	return std::views::cartesian_product(index_view, hopping_view)
		| std::views::transform([this](auto&& tp) -> std::tuple<LatticeVector<Dim>, LatticeVector<Dim>, int, int, Complex> {
			auto&& [i_v, hp_t] = tp;
			auto&& [hp, t] = hp_t;
			auto&& [from, to, lat_v] = hp;
			decltype(i_v)&& to_v = i_v + lat_v; // No auto, i_v + lat_v returns type binaryOp<Vec, Vec>, causing leakage
			return std::make_tuple(i_v, to_v, from, to, t);
			})
		| std::views::filter(filter);
}

// return as [i_v, to_v, from, to, t]
template<int Dim> requires (is_lattice_dim(Dim))
inline auto PartialLattice<Dim>::hopping_of_each_atom_view() const
{
	return hopping_of_each_atom_view_filter([this](auto&& tp) {
		auto&& to_v = std::get<1>(tp);
		return index_inside_lattice(to_v);
		});
}

template<int Dim> requires (is_lattice_dim(Dim))
inline void PartialLattice<Dim>::set_scalar_field(auto&& f)
{
	scalar_field = f;
	for (int i = 0; i < cell_.get_atom_number(); i++) {
		if (cell_.hopping_exists({ i, i, LatticeVector<Dim>::Zero() })) {
			continue;
		}
		cell_.add_hopping_direct({ i ,i, LatticeVector<Dim>::Zero() }, 0.0);
	}
}

//template<int Dim> requires (is_lattice_dim(Dim))
//inline auto PartialLattice<Dim>::index_sequence_view() const
//{
//	return BlockView{ this, LatticeVector<Dim>::Zero(), span_v }.index_sequence();
//}

//template<int Dim> requires (is_lattice_dim(Dim))
//struct PartialLattice<Dim>::BlockView {
//	const PartialLattice<Dim>* self;
//	LatticeVector<Dim> corner_v, span_v;
//
//	template <int... ints>
//	auto index_sequence_tuple(std::integer_sequence<int, ints...>) const
//	{
//		return std::views::cartesian_product(std::views::iota(corner_v(ints), span_v(ints))...);
//	}
//	auto index_sequence() const
//	{
//		return index_sequence_tuple(std::make_integer_sequence<int, Dim>{})
//			| std::views::transform([](auto&& t) { return UnitCell<Dim>::get_vector_from_tuple(t); });
//	}
//	BlockView(const PartialLattice<Dim>* self, LatticeVector<Dim> corner_v, LatticeVector<Dim> span_v)
//		: self(self), corner_v(corner_v), span_v(span_v) {}
//};

template<int Dim> requires (is_lattice_dim(Dim))
template<ScalarType Scalar>
struct PartialLattice<Dim>::GreenFunction {
	using Mat = Eigen::SparseMatrix<Scalar>;
	using Ket = Eigen::VectorXcd;
	using Kets = Eigen::MatrixXcd;
	using Bras = Eigen::MatrixXcd;

	int N;
	double a, b;
	double E_min, E_max;
	Mat scaled_HE;
	std::vector<Mat> Tn_EH;
	static Mat identity_of_size(const Mat& A) {
		Mat id(A.rows(), A.cols());
		id.setIdentity();
		return id;
	}
	template <typename T> auto scale(const T& x, T&& id) -> T { return (x - b * id) / a; }
	// HE: H - E_0, ref_HE is a reference Hamiltonian for computing the spectral range, should be small
	// epsilon is the contraction of scaling
	GreenFunction(int N, const Mat& HE, double E_min, double E_max)
		: N(N), E_min(E_min), E_max(E_max)
	{
		a = (E_max - E_min) / 2;
		b = (E_max + E_min) / 2;
		scaled_HE = scale(HE, identity_of_size(HE));
	}
	GreenFunction(int N, const Mat& HE, const Mat& ref_HE, double epsilon = 0.01)
		: N(N)
	{
		std::tie(E_min, E_max) = numerical::linalg::matrix_spectral_range_sparse(ref_HE);
		a = (E_max - E_min) / (2 - epsilon);
		b = (E_max + E_min) / 2;
		scaled_HE = scale(HE, identity_of_size(HE));
		if (auto_info)
			std::cout << "GreenFunction generated, energy range is " << E_min << ", " << E_max << std::endl;
	}
	// for small systems
	GreenFunction(int N, const Mat& HE)
		: GreenFunction(N, HE, HE)
	{
		if (HE.rows() < 5000)
			Tn_EH = numerical::Chebyshev::Cheb_first_kind_seq(scaled_HE, N, identity_of_size(scaled_HE));
		else if (auto_info)
			std::cout << "GreenFunctionKP refuses to generate Chebyt seq of H (H is too big)." << std::endl;
	}

	Mat direct_eval(Complex z, int sgn = 1) {
		if (Tn_EH.empty()) std::cerr << "Warning: Too big to implement direct eval." << std::endl;
		Mat Gf(scaled_HE.rows(), scaled_HE.cols());
		Gf.setZero();
		z = scale(z, Complex(1.0, 0.0));
		auto alpha = numerical::Chebyshev::expn_coef_alpha_seq(z, N);
		for (auto&& [an, Tn] : std::views::zip(alpha, Tn_EH)) Gf += an.imag() * Tn; // TODO, imag may be removed
		return Gf / a;
	}
	// z = omega + i * eta, omega is the energy, eta corresponds to the imaginary part of energy
	Mat direct_eval(double omega, double eta = 0.01) {
		int sgn = eta > 0 ? 1 : -1;
		return direct_eval(Complex(omega, eta), sgn);
	}
	// return pair of E_min and E_max
	std::pair<double, double> spectral_range() {
		return { E_min, E_max };
	}

	// template<ScalarType Scalar>
	struct GreenFunctionImag {
		GreenFunction<Scalar>* G;
		std::vector<double> moments;
		std::vector<double> g; // kernel
		// a functor of e, [<bras| Im(G(¡¤) |kets>](e)
		GreenFunctionImag(GreenFunction<Scalar>* G, const Kets& bras_T, const Kets& kets)
			: G(G)
		{
#ifdef USE_CUDA
			moments = cuda::Cheb_first_kind_seq_braket(G->scaled_HE, G->N, bras_T, kets);
#else
			moments = numerical::Chebyshev::Cheb_first_kind_seq_braket(G->scaled_HE, G->N, bras_T, kets);
#endif
			g = numerical::kernels::kernel_seq(G->N, numerical::kernels::Lorentz(G->N, 4));
			if (auto_info) std::cout << "GreenFunctionImag Chebyt moments generated." << std::endl;
		}
		static double weight(double x) { return 1. / std::sqrt(1. - x * x); }
		// directly evaluate the expansion sum at e, sgn = ¡À1
		double operator()(Complex e) const {
			auto alpha = numerical::Chebyshev::expn_coef_alpha_seq(e, G->N);
			double sum = 0;
			for (auto&& [gn, an, mu] : std::views::zip(g, alpha, moments)) sum += gn * an.imag() * mu;
			// for (auto&& [an, mu] : std::views::zip(alpha, moments)) sum += an.imag() * mu;
			return sum;
		}
		auto fft(int n = 1) -> std::pair<std::vector<double>, std::vector<double>> const {
			int N = G->N * n;
			auto Theta = utils::range(0.5 * pi / N, pi / N, N);
			std::vector<double> mu_tilde(N), fft_result(N);
			for (int i = 0; i < N; i++) mu_tilde[i] = g[i] * moments[i];
			fftw_plan plan = fftw_plan_r2r_1d(N, mu_tilde.data(), fft_result.data(), FFTW_REDFT01, FFTW_ESTIMATE);
			fftw_execute(plan);
			fftw_destroy_plan(plan);
			std::vector<double> E_list(N), result(N);
			for (int i = 0; i < N; i++) {
				E_list[i] = std::cos(Theta[i]);
				result[i] = fft_result[i] * weight(E_list[i]);
				std::cout << i << " " << E_list[i] << " " << (result[i]) << std::endl;
			}
			return { E_list, result };
		}
		auto slice_range(double e_min, double e_max, int n = 1) -> std::pair<int, int> const {
			int N = G->N * n;
			double theta_right = std::acos(e_min);
			double theta_left = std::acos(e_max);
			return { static_cast<int>(round(theta_left / pi * N)), static_cast<int>(round(theta_right / pi * N)) };
		}
	};
	// returns a functor of e, [<bras| Im(G(¡¤) |kets>](e), e is the energy
	GreenFunctionImag imag(const Kets& bras_T, const Kets& kets) {
		return GreenFunctionImag(this, bras_T, kets);
	}
};

template<int Dim> requires (is_lattice_dim(Dim))
struct PartialLattice<Dim>::GreenFunction_Dense {
	int N;
	double a, b;
	double E_min, E_max;
	Eigen::MatrixXcd HE, scaled_HE;
	std::vector<Eigen::MatrixXcd> Tn_EH;
	static Eigen::MatrixXcd identity_of_size(const Eigen::MatrixXcd& A) {
		return Eigen::MatrixXcd::Identity(A.rows(), A.cols());
	}
	template <typename T> auto scale(const T& x, T&& id) -> T { return (x - b * id) / a; }
	// HE: H - E_0, ref_HE is a reference Hamiltonian for computing the spectral range, should be small
	GreenFunction_Dense(int N, const Eigen::MatrixXcd& HE, double epsilon = 0.01)
		: N(N), HE(HE)
	{
		std::tie(E_min, E_max) = numerical::linalg::matrix_spectral_range_dense(HE);
		a = (E_max - E_min) / (2 - epsilon);
		b = (E_max + E_min) / 2;
		scaled_HE = scale(HE, identity_of_size(HE));
		Tn_EH = numerical::Chebyshev::Cheb_first_kind_seq(scaled_HE, N, identity_of_size(scaled_HE));
	}
	GreenFunction_Dense(int N, const Eigen::MatrixXcd& HE, double E_min, double E_max)
		: N(N), HE(HE), E_min(E_min), E_max(E_max)
	{
		a = (E_max - E_min) / 2;
		b = (E_max + E_min) / 2;
		scaled_HE = scale(HE, identity_of_size(HE));
		Tn_EH = numerical::Chebyshev::Cheb_first_kind_seq(scaled_HE, N, identity_of_size(scaled_HE));
	}

	Eigen::MatrixXcd eval(Complex e) {
		Eigen::MatrixXcd Gf = Eigen::MatrixXcd::Zero(scaled_HE.rows(), scaled_HE.cols());
		e = scale(e, Complex(1.0, 0.0));
		auto alpha = numerical::Chebyshev::expn_coef_alpha_seq(e, N);
		for (auto&& [an, Tn] : std::views::zip(alpha, Tn_EH)) Gf += an * Tn;
		// Gf = (-HE + z * identity_of_size(HE)).inverse();
		return Gf / a;
	}
	// e = omega + i * eta
	Eigen::MatrixXcd eval(double omega, double eta = 0.01) {
		int sgn = eta > 0 ? 1 : -1;
		return eval(Complex(omega, eta), sgn);
	}
	std::pair<double, double> spectral_range() {
		return { E_min, E_max };
	}
};
