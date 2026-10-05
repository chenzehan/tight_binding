#pragma once

#define _SILENCE_CXX23_DENORM_DEPRECATION_WARNING
#include <vector>
#include <unordered_map>
#include <iostream>
#include <tuple>
#include <queue>
#include <ranges>
#include <unordered_set>
#include <complex>
#include <numbers>
#include <functional>

#include "utils.h"
#include "spatial_geometry.h"

#include <Eigen/Dense>

#ifndef PYTHON_LIB
	#include <matplot/matplot.h>
	namespace plt = matplot;
#else
	#include "pybind_plot.h"
	namespace plt = pybind_plot;
#endif

using std::numbers::pi;
using Vector3d = Eigen::Vector3d;
using Complex = Eigen::dcomplex;
constexpr std::complex<double> operator"" _j(long double imaginaryPart) {
	return std::complex<double>(0.0, imaginaryPart);
}
std::string to_string_complex(Complex c) {
	if (c.imag() == 0) return std::format("{:.1f}", c.real());
	if (c.real() == 0) return std::format("{:.1f}i", c.imag());
	return std::format("{:.1f}{}{:.1f}i", c.real(), (c.imag() > 0 ? '+' : '-'), abs(c.imag()));
}

constexpr bool is_lattice_dim(int dim) // allow only 1, 2, 3 dimensions
{
	return dim == 1 || dim == 2 || dim == 3;
}

template<int Dim> requires (is_lattice_dim(Dim))
using LatticeVector = Eigen::Matrix<int, Dim, 1>;

template<int Dim> requires (is_lattice_dim(Dim))
using Hopping = std::tuple<int, int, LatticeVector<Dim>>;

using HoppingType = unsigned long long int;
constexpr HoppingType On_Site = 0;

HoppingType operator""_NN(unsigned long long int n) { return static_cast<int>(n); }

template<int Dim> requires (is_lattice_dim(Dim))
class UnitCellNN;

template<int Dim> requires (is_lattice_dim(Dim))
class PartialLattice;

template<int Dim> requires (is_lattice_dim(Dim))
using NN_info = std::tuple<double, Hopping<Dim>>;

template<int Dim> requires (is_lattice_dim(Dim))
class UnitCell
{
public:
	UnitCell(std::vector<Vector3d>&& lattice_vectors, std::vector<std::pair<std::string, Vector3d>>&& atoms);
	UnitCell<Dim>& add_hopping(const std::string& from, const std::string& to, LatticeVector<Dim> lattice_vector, Complex hopping_strength);
	UnitCell<Dim>& add_hopping(const std::string& from, HoppingType NN, Complex hopping_strength);
	// std::vector<hopping<Dim>> get_hoppings_sorted() const;

	double distance_of_hopping(const Hopping<Dim>& h) const;
	double distance_between_lattices(LatticeVector<Dim> a) const;
	double distance_between_lattices(LatticeVector<Dim> a, LatticeVector<Dim> b) const;

	Vector3d displacement_of_lattice(LatticeVector<Dim> a) const;
	Vector3d displacement_of_hopping(const Hopping<Dim>& h) const;

	static Hopping<Dim> hopping_hc(const Hopping<Dim>& h); // Hermitian conjugate of hopping
	Eigen::MatrixXcd Hamiltonian_k(Vector3d k) const; // get Hamiltonian_k matrix
	Eigen::MatrixXcd Hamiltonian_k_lower_tri(Vector3d k) const;
	Eigen::VectorXd Hamiltonian_k_eigenvalues(Vector3d k) const;
	std::vector<Eigen::VectorXd> Hamiltonian_k_eigenvalues(const std::vector<Vector3d>& k_vec) const;
	std::vector<std::vector<double>> get_band_dispersion_as_vecs(const std::vector<Vector3d>& k_vec) const;

	Vector3d atom_position(const std::string& atom) const { return atom_pos.at(atom_no.at(atom)); }
	Vector3d atom_position(const std::string& atom, LatticeVector<Dim> lat_v) const 
		{ return atom_pos.at(atom_no.at(atom)) + displacement_of_lattice(lat_v); }
	Vector3d atom_position(int atom_no_, LatticeVector<Dim> lat_v) const
		{ return atom_pos.at(atom_no_) + displacement_of_lattice(lat_v); }
	void draw_primitive_cell();
	// void draw_primitive_cell_3D();
	void draw_hoppings();
	void draw_band_dispersion() const;
	void draw_k_space() const;

	auto get_first_Brillouin_zone() const;
	int get_atom_number() const { return static_cast<int>(atom_pos.size()); }
	int get_atom_index(const std::string& name) const { return atom_no.at(name); }
	bool atom_exists(const std::string& name) const { return atom_no.find(name) != atom_no.end(); }
	auto hopping_view() const { return hoppings | std::views::all; }

	Eigen::Matrix<double, 3, Dim> bases;
	Eigen::Matrix<double, Dim, Dim> k_bases; // column vectors are the reciprocal lattice vectors

	const inline static LatticeVector<Dim> Intra_Cell = LatticeVector<Dim>::Zero();
	const inline static LatticeVector<Dim> Left_NN = -LatticeVector<Dim>::Unit(0);
	const inline static LatticeVector<Dim> Right_NN = LatticeVector<Dim>::Unit(0);

	inline static bool auto_show = true;
	struct HoppingHash;
private:
	// provide hash function for hopping, used in std::unordered_map
	// template<int Dim> requires (is_lattice_dim(Dim))

	friend class UnitCellNN<Dim>;
	friend class PartialLattice<Dim>;
	std::unordered_map<std::string, int> atom_no; // atom number in the cell
	std::vector<std::string> atom_name;
	std::vector<Vector3d> atom_pos;
	std::unordered_map<Hopping<Dim>, Complex, HoppingHash> hoppings;

	std::unordered_map<int, UnitCellNN<Dim>> hopping_gen;

	std::function<Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>(Vector3d)> H_k;

	void add_hopping_direct(Hopping<Dim>&& h, Complex t);
	void draw_atomic_sites(plt::axes_handle ax, bool draw_NN, int mesh = 3);
	void draw_band_dispersion_1D() const;
	void draw_band_dispersion_2D() const;
	void draw_k_space_2D() const;
	void draw_k_space_3D() const;
	bool hopping_exists(const Hopping<Dim>& h) const;

	static std::vector<LatticeVector<Dim>> get_NN_vectors(const auto& nn);
	inline static std::vector<LatticeVector<Dim>> NN_vectors = get_NN_vectors(std::array{ -1, 0, 1 });

	template <int N, typename T, typename... Args>
	constexpr static auto make_cartesian(const T& x, const Args&... args) // make_cartesian<3>(x) returns cartesian product of (x, x, x)
	{
		if constexpr (N == 1)
			return std::views::cartesian_product(x, args...);
		else
			return make_cartesian<N - 1>(x, x, args...);
	}
public:
	template<typename tuple_t>
	constexpr static LatticeVector<Dim> get_vector_from_tuple(tuple_t&& tuple)
	{
		constexpr auto get_vector = [](auto&& ... x) { return LatticeVector<Dim>{ std::forward<decltype(x)>(x) ... }; };
		return std::apply(get_vector, std::forward<tuple_t>(tuple));
	}
	// constexpr static auto nn = std::array{ -1, 0, 1 };
};

using UnitCell_1D = UnitCell<1>;
using UnitCell_2D = UnitCell<2>;
using UnitCell_3D = UnitCell<3>;

template<int Dim> requires (is_lattice_dim(Dim))
class UnitCellNN
{
private:
	friend class UnitCell<Dim>;

	int A; // atom number
	UnitCell<Dim>* cell_ptr;

	// provide hash function for lattice_vector, used in std::unordered_map
	struct LatticeVectorHash;

	std::priority_queue <NN_info<Dim>, std::vector<NN_info<Dim>>, 
		decltype([](const NN_info<Dim>& n, const NN_info<Dim>& m) { return std::get<0>(n) > std::get<0>(m); })> NN_atoms_to_add;
	std::unordered_set<LatticeVector<Dim>, LatticeVectorHash> NN_lattice_registered;
	double current_NN_distance;
	std::vector<std::vector<Hopping<Dim>>> NN_solved;

	UnitCellNN(UnitCell<Dim>* cell, int atom_no)
		:cell_ptr(cell), A(atom_no), current_NN_distance(0.0)
	{}

	void generate_hoppings_to(HoppingType NN);
	void add_cell(LatticeVector<Dim> a);
};

template<int Dim> requires (is_lattice_dim(Dim))
UnitCell<Dim>::UnitCell(std::vector<Vector3d>&& lattice_vectors, std::vector<std::pair<std::string, Vector3d>>&& atoms)
	: atom_pos(atoms.size())
{
	if (lattice_vectors.size() != Dim)
		throw std::runtime_error("Number of lattice vectors must be equal to the dimension of the unit cell");
	for (auto&& v: lattice_vectors) {
		if (v.isZero()) throw std::runtime_error("Lattice vectors must not be zero");
	}
	if (atoms.size() == 0)
		throw std::runtime_error("Unit cell must contain at least one atom");
	for (int i = 0; i < Dim; i++) {
		bases.col(i) = lattice_vectors[i];
	}
	for (auto&& [i, x]: std::views::enumerate(atoms)) {
		if (atom_no.find(x.first) != atom_no.end())
			throw std::runtime_error("Atom " + x.first + " is already defined");
		atom_no.insert({ x.first, i });
		atom_name.push_back(x.first);
		atom_pos[i] = x.second;
	}
	k_bases = 2*pi * bases.block(0, 0, Dim, Dim).inverse().transpose();
}

template<int Dim> requires (is_lattice_dim(Dim))
inline UnitCell<Dim>& UnitCell<Dim>::add_hopping(const std::string& from, const std::string& to, LatticeVector<Dim> lattice_vector, Complex hopping_strength)
{
	auto from_ptr = atom_no.find(from);
	if (from_ptr == atom_no.end())
		throw std::runtime_error("Atom " + from + " not found in the unit cell");
	auto to_ptr = atom_no.find(to);
	if (to_ptr == atom_no.end())
		throw std::runtime_error("Atom " + to + " not found in the unit cell");
	if (from == to && lattice_vector.isZero() && std::imag(hopping_strength) != 0) {
		std::cerr << "On-site hopping must be real" << std::endl;
		return *this;
	}
	auto&& hp = std::make_tuple(from_ptr->second, to_ptr->second, lattice_vector);
	add_hopping_direct(std::move(hp), hopping_strength);
	return *this;
}

template<int Dim> requires (is_lattice_dim(Dim))
inline UnitCell<Dim>& UnitCell<Dim>::add_hopping(const std::string& from, HoppingType NN, Complex hopping_strength)
{
	auto from_ptr = atom_no.find(from);
	if (from_ptr == atom_no.end()) {
		std::cerr << "Atom " << from << " not found in the unit cell" << std::endl;
		return *this;
	}
	int from_no = from_ptr->second;
	if (NN == On_Site) {
		if (std::imag(hopping_strength) != 0) {
			std::cerr << "On-site hopping must be real" << std::endl;
			return *this;
		}
		auto hp = std::make_tuple(from_no, from_no, LatticeVector<Dim>::Zero());
		add_hopping_direct(std::move(hp), hopping_strength);
		return *this;
	}
	auto it = hopping_gen.find(from_no);
	if (it == hopping_gen.end())
		it = hopping_gen.insert({ from_no, UnitCellNN<Dim>{ this, from_no } }).first;
	it->second.generate_hoppings_to(NN);
	for (auto hp : it->second.NN_solved[NN]) {
		add_hopping_direct(std::move(hp), hopping_strength);
	}
	return *this;
}

//template<int Dim> requires (is_lattice_dim(Dim))
//inline std::vector<hopping<Dim>> UnitCell<Dim>::get_hoppings_sorted() const
//{
//	std::vector<hopping<Dim>> hoppings_sorted;
//	for (const auto& [h, _] : hoppings) {
//		hoppings_sorted.push_back(h);
//	}
//	return hoppings_sorted;
//}

template<int Dim> requires (is_lattice_dim(Dim))
inline Vector3d UnitCell<Dim>::displacement_of_hopping(const Hopping<Dim>& h) const
{
	auto&& [from, to, lattice_vector] = h;
	return -atom_pos.at(from) + displacement_of_lattice(lattice_vector) + atom_pos.at(to);
}

template<int Dim> requires (is_lattice_dim(Dim))
inline double UnitCell<Dim>::distance_of_hopping(const Hopping<Dim>& h) const
{
	return displacement_of_hopping(h).norm();
}

template<int Dim> requires (is_lattice_dim(Dim))
inline double UnitCell<Dim>::distance_between_lattices(LatticeVector<Dim> a) const
{
	return displacement_of_lattice(a).norm();
}

template<int Dim> requires (is_lattice_dim(Dim))
inline double UnitCell<Dim>::distance_between_lattices(LatticeVector<Dim> a, LatticeVector<Dim> b) const
{
	return distance_between_lattices(a - b);
}

template<int Dim> requires (is_lattice_dim(Dim))
inline Vector3d UnitCell<Dim>::displacement_of_lattice(LatticeVector<Dim> a) const
{
	return bases * a.cast<double>();
}

template<int Dim> requires (is_lattice_dim(Dim))
inline Hopping<Dim> UnitCell<Dim>::hopping_hc(const Hopping<Dim>& h)
{
	return std::make_tuple(std::get<1>(h), std::get<0>(h), -std::get<2>(h));
}

template<int Dim> requires (is_lattice_dim(Dim))
inline Eigen::MatrixXcd UnitCell<Dim>::Hamiltonian_k(Vector3d k) const
{
	Eigen::MatrixXcd H(atom_pos.size(), atom_pos.size());
	H.setZero();
	for (auto&& [hp, t] : hoppings) {
		auto&& [row, col, lat_v] = hp;
		auto&& r = displacement_of_hopping(hp); // May involve repeating calculation
		if (row == col) H(row, col) += t.real() * 2. * std::cos(k.dot(r));
		else {
			H(row, col) += t * std::exp(Complex(0, 1) * k.dot(r));
			H(col, row) += std::conj(t * std::exp(Complex(0, 1) * k.dot(r)));
		}
	}
	return H;
}

// get Hamiltonian_k matrix only lower triangular part
template<int Dim> requires (is_lattice_dim(Dim))
inline Eigen::MatrixXcd UnitCell<Dim>::Hamiltonian_k_lower_tri(Vector3d k) const
{
	Eigen::MatrixXcd H(atom_pos.size(), atom_pos.size());
	H.setZero();
	for (auto&& [hp, t] : hoppings) {
		auto&& [row, col, lat_v] = hp;
		auto&& r = displacement_of_hopping(hp);
		if (row == col) H(row, col) += t * 2. * std::cos(k.dot(r));
		else H(row, col) += t * std::exp(Complex(0, 1) * k.dot(r));
	}
	return H;
}

template<int Dim> requires (is_lattice_dim(Dim))
inline Eigen::VectorXd UnitCell<Dim>::Hamiltonian_k_eigenvalues(Vector3d k) const
{
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(Hamiltonian_k_lower_tri(k));
	return es.eigenvalues();
}

template<int Dim> requires (is_lattice_dim(Dim))
inline std::vector<Eigen::VectorXd> UnitCell<Dim>::Hamiltonian_k_eigenvalues(const std::vector<Vector3d>& k_vec) const
{
	return k_vec
		| std::views::transform([this](Vector3d k) { return Hamiltonian_k_eigenvalues(k); })
		| std::ranges::to<std::vector>();
}

template<int Dim> requires (is_lattice_dim(Dim))
inline std::vector<std::vector<double>> UnitCell<Dim>::get_band_dispersion_as_vecs(const std::vector<Vector3d>& k_vec) const
{
	if (k_vec.empty()) {
		std::cerr << "k_vec is empty" << std::endl;
		return {};
	}
	auto eigenvalues = Hamiltonian_k_eigenvalues(k_vec);
	std::vector<std::vector<double>> band_dispersion(atom_pos.size());
	for (auto&& e : eigenvalues) {
		for (int i = 0; i < e.size(); i++) {
			band_dispersion[i].push_back(e(i));
		}
	}
	return band_dispersion;
}

template<int Dim> requires (is_lattice_dim(Dim))
inline void UnitCell<Dim>::draw_primitive_cell()
{
	plt::figure(true);
	plt::hold(true);

	draw_atomic_sites(plt::gca(), true, 6 - Dim);
	
	if constexpr (Dim != 3)
	for (int i = 0; i < Dim; i++) {
		plt::arrow(0, 0, bases(0, i), bases(1, i));
	}

	if (auto_show) plt::show();
}

// draw primitive cell and hoppings (including on-site), only hoppings from the primitive cell atoms are drawn
template<int Dim> requires (is_lattice_dim(Dim))
inline void UnitCell<Dim>::draw_hoppings()
{
	plt::figure(true);
	plt::hold(true);

	draw_atomic_sites(plt::gca(), false);
	for (auto&& [h, t] : hoppings) {
		auto&& [from, to, lattice_vector] = h;
		auto from_pos = atom_pos[from];
		auto to_pos = atom_pos[to] + displacement_of_lattice(lattice_vector);
		// draw hopping arrow
		plt::arrow(from_pos(0), from_pos(1), to_pos(0), to_pos(1));
		// write hopping strength on the middle of the arrow
		plt::text((from_pos(0) + to_pos(0)) / 2, (from_pos(1) + to_pos(1)) / 2, to_string_complex(t)); 
	}

	if (auto_show) plt::show();
}

template<int Dim> requires (is_lattice_dim(Dim))
inline void UnitCell<Dim>::draw_band_dispersion() const
{
	if constexpr (Dim > 2) {
		std::cerr << "Band dispersion plot is only available for 1D and 2D lattices" << std::endl;
		return;
	}
	if constexpr (Dim == 2) {
		draw_band_dispersion_2D();
		return;
	}
	else draw_band_dispersion_1D();
}

template<int Dim> requires (is_lattice_dim(Dim))
inline void UnitCell<Dim>::draw_k_space() const
{
	if constexpr (Dim == 1) {
		std::cerr << "k-space plot is only available for 2D and 3D lattices" << std::endl;
	}
	else if constexpr (Dim == 2) {
		draw_k_space_2D();
	}
	else draw_k_space_3D();
}

template<int Dim> requires (is_lattice_dim(Dim))
inline void UnitCell<Dim>::draw_k_space_2D() const
{
	auto lattice_mesh = make_cartesian<Dim>(std::views::iota(-5, 5 + 1)) 
		| std::views::transform([](auto&& t) { return get_vector_from_tuple(t); });
	auto lattice_mesh_xy = lattice_mesh | std::views::transform([this](auto&& v) { return k_bases * v.cast<double>(); });
	auto X = lattice_mesh_xy | std::views::transform([](auto&& v) { return v(0); }) | std::ranges::to<std::vector>();
	auto Y = lattice_mesh_xy | std::views::transform([](auto&& v) { return v(1); }) | std::ranges::to<std::vector>();
	
	plt::figure(true);
	plt::hold(true);

	plt::plot(X, Y, "o");
	plt::arrow(0, 0, k_bases(0, 0), k_bases(1, 0));
	plt::arrow(0, 0, k_bases(0, 1), k_bases(1, 1));

	auto [BZ_1x, BZ_1y] = get_first_Brillouin_zone();
	BZ_1x.push_back(BZ_1x[0]);
	BZ_1y.push_back(BZ_1y[0]);
	plt::plot(BZ_1x, BZ_1y, "k-")->line_width(2).color("red");

	plt::xlabel("k_x");
	plt::ylabel("k_y");
	double plot_range_x = abs(k_bases(0, 0)) * 2.8;
	plt::xlim({ -plot_range_x, plot_range_x });
	plt::ylim({ -plot_range_x, plot_range_x });
	if (auto_show) plt::show();
}

template<int Dim> requires (is_lattice_dim(Dim))
inline void UnitCell<Dim>::draw_k_space_3D() const
{
	plt::figure(true);
	plt::hold(true);

	auto BZ_1 = get_first_Brillouin_zone();
	for (auto& polygon : BZ_1) {
		auto& [X, Y, Z] = polygon;
		X.push_back(X[0]);
		Y.push_back(Y[0]);
		Z.push_back(Z[0]);
		plt::plot3(X, Y, Z)->line_width(2).color("red");
	}
	if (auto_show) plt::show();
}

template<int Dim> requires (is_lattice_dim(Dim))
inline auto UnitCell<Dim>::get_first_Brillouin_zone() const
{
	using namespace spatial_geometry;
	if constexpr (Dim == 1) {
		return std::tuple{ -k_bases(0, 0) / 2., k_bases(0, 0) / 2. };
	}
	else if constexpr (Dim == 2) {
		auto NN_k_points = NN_vectors | std::views::transform([this](auto&& v) { return k_bases * v.cast<double>(); });
		auto NN_affine_points = NN_k_points
			| std::views::transform([](auto&& v) { return Vector3d{ v(0), v(1), 1. }; });
		Vector3d O{ 0., 0., 1. };
		auto affine_bisects = NN_affine_points
			| std::views::transform([O](auto&& v) { return affine_2D::perp_bisector(O, v); })
			| std::ranges::to<std::vector>();
		auto BZ_1_affine = minimal_complex::minimal_polygon(O, affine_bisects,
			affine_2D::distance_to_line, affine_2D::project_point,
			affine_2D::crossing_point, [](const auto& L1, const auto& L2) { return abs(affine_2D::crossing_point(L1, L2)(2)) < 1e-6; },
			[](const auto& v1, const auto& v2) { return v1.cross(v2)(2) > 0; });
		// auto BZ_1 = BZ_1_affine | std::views::transform([](auto v) { return Vector2d(v.head<2>()); }) | std::ranges::to<std::vector>();
		auto BZ_1x = BZ_1_affine | std::views::transform([](auto v) { return v(0); }) | std::ranges::to<std::vector>();
		auto BZ_1y = BZ_1_affine | std::views::transform([](auto v) { return v(1); }) | std::ranges::to<std::vector>();
		return std::make_tuple(BZ_1x, BZ_1y);
	}
	else if constexpr (Dim == 3) {
		// auto NN_k_points = NN_vectors | std::views::transform([this](auto&& v) { return k_bases * v.cast<double>(); });
		std::vector<LatticeVector<Dim>> NN_k_points_5 = UnitCell<Dim>::get_NN_vectors(std::array{ -2,-1,0,1,2 });
		auto NN_k_points = NN_k_points_5 | std::views::transform([this](auto&& v) { return k_bases * v.cast<double>(); });
		auto NN_affine_points = NN_k_points
			| std::views::transform([](auto&& v) { return Vector4d{ v(0), v(1), v(2), 1. }; });
		Vector4d O{ 0., 0., 0., 1. };
		auto affine_bisect = NN_affine_points
			| std::views::transform([O](auto&& v) { return affine_3D::perp_bisector(O, v); })
			| std::ranges::to<std::vector>();
		return minimal_complex::minimal_3D_complex(O, affine_bisect);
	}
}

//template<int Dim> requires (is_lattice_dim(Dim))
//inline void UnitCell<Dim>::add_hopping_to_hopping_compressed(const hopping<Dim>& h, Complex t)
//{
//	auto&& [row, col, lattice_vector] = h;
//	if (row < col) { // add only lower triangular part of the Hamiltonian
//		hopping_compressed.emplace_back(col, row, -displacement_of_hopping(h), std::conj(t));
//		return;
//	}
//	hopping_compressed.emplace_back(row, col, displacement_of_hopping(h), t);
//}

template<int Dim> requires (is_lattice_dim(Dim))
inline bool UnitCell<Dim>::hopping_exists(const Hopping<Dim>& h) const
{
	return hoppings.find(h) != hoppings.end() || hoppings.find(hopping_hc(h)) != hoppings.end();
}

template<int Dim> requires (is_lattice_dim(Dim))
inline void UnitCell<Dim>::add_hopping_direct(Hopping<Dim>&& h, Complex t)
{
	if (hopping_exists(h)) {
		std::cerr << "Hopping already exists" << std::endl;
		return;
	}
	auto& [from, to, lattice_vector] = h;
	if (from == to && lattice_vector.isZero() && std::imag(t) != 0) {
		std::cerr << "On-site hopping must be real" << std::endl;
		return;
	}
	if (from < to) hoppings.insert({ hopping_hc(h), std::conj(t) }); // to make row >= col
	else hoppings.insert({ std::forward<decltype(h)>(h), t});
}

template<int Dim> requires (is_lattice_dim(Dim))
inline void UnitCell<Dim>::draw_atomic_sites(plt::axes_handle ax, bool draw_NN, int mesh)
{
	// lattice mesh is a list of lattice vectors to be drawn, it meshs [-mesh, mesh] x [-mesh, mesh] primitive cells
	/*auto lattice_mesh = [=]() {
		if constexpr (Dim == 3) {
			auto x = std::views::iota(-mesh, mesh + 1);
			return std::views::cartesian_product(x, x, { 0. });
		}
		else return make_cartesian<Dim>(std::views::iota(-mesh, mesh + 1));
		}() | std::views::transform([](auto&& t) { return get_vector_from_tuple(t); });*/
	auto lattice_mesh = make_cartesian<Dim>(std::views::iota(-mesh, mesh + 1))
		| std::views::transform([](auto&& t) { return get_vector_from_tuple(t); });
	// lattice_mesh_xy converts lattice_mesh to a list of real space positions
	auto lattice_mesh_xy = lattice_mesh | std::views::transform([this](auto&& v) { return displacement_of_lattice(v); });
	// to_xy_vector converts lattice_mesh_xy to a pair of X and Y coordinates
	auto to_xy_vector = [](auto&& xy) {
		auto X = xy | std::views::transform([](auto&& v) { return v(0); }) | std::ranges::to<std::vector>();
		auto Y = xy | std::views::transform([](auto&& v) { return v(1); }) | std::ranges::to<std::vector>();
		return std::make_pair(X, Y);
	};

	for (auto&& [i_, a] : std::views::enumerate(atom_pos)) { // visit all atoms in the primitive cell
		int i = i_;
		auto a_xy = lattice_mesh_xy | std::views::transform([a](auto&& v) { return v + a; });
		auto [X, Y] = to_xy_vector(a_xy);
		if constexpr (Dim != 3) {
			ax->plot(X, Y, "o");
			ax->text(a(0), a(1), atom_name[i]);
		}
		else {
			auto Z = a_xy | std::views::transform([](auto&& v) { return v(2); }) | std::ranges::to<std::vector>();
			ax->plot3(X, Y, Z, "o");
		}

		if (draw_NN) { // if draw_NN is true, draw nearest neighbor lines
			auto NN_it = hopping_gen.find(i);
			if (NN_it == hopping_gen.end()) {
				NN_it = hopping_gen.insert({ i, UnitCellNN<Dim>{ this, i } }).first;
			}
			auto& NN = NN_it->second;
			NN.generate_hoppings_to(1_NN); // generate NN infos to 1_NN to find nearest neighbors
			for (auto&& h : NN.NN_solved[1_NN]) {
				auto&& [from, to, lattice_vector] = h;

				for (auto&& lat_v: lattice_mesh) { // draw nearest neighbor lines for each lattice mesh
					auto from_pos = atom_pos[from] + displacement_of_lattice(lat_v);
					auto to_pos = atom_pos[to] + displacement_of_lattice(lattice_vector + lat_v);
					if constexpr (Dim != 3) {
						ax->plot({ from_pos(0), to_pos(0) }, { from_pos(1), to_pos(1) }, "k-")
							->line_width(0.5).color("C0");
					}
					else {
						ax->plot3({ from_pos(0), to_pos(0) }, { from_pos(1), to_pos(1) }, { from_pos(2), to_pos(2) }, "k-")
							->line_width(0.5).color("C0");
					}
				}
			}
		}
	}

	double plot_range_x = bases.col(0).norm() * (4.2 - Dim * 0.8);
	double plot_range_y = plot_range_x;

	ax->xlim({ -plot_range_x, plot_range_x });
	ax->ylim({ -plot_range_y, plot_range_y });
	if constexpr (Dim == 3) ax->zlim({ -plot_range_x, plot_range_x });
	// plt::axis("equal"); // ?
}

template<int Dim> requires (is_lattice_dim(Dim))
inline void UnitCell<Dim>::draw_band_dispersion_1D() const
{
	plt::figure(true);
	plt::hold(plt::on);

	auto K_x = utils::linspace(-pi / bases(0, 0), pi / bases(0, 0), 201, true) | std::ranges::to<std::vector>();
	auto E = get_band_dispersion_as_vecs(K_x
		| std::views::transform([](double k) { return Vector3d(k, 0., 0.); })
		| std::ranges::to<std::vector>());

	for (auto&& e : E) {
		plt::plot(K_x, e);
	}

	plt::xlabel("k");
	plt::ylabel("E");
	if (auto_show) plt::show();
}

template<int Dim> requires (is_lattice_dim(Dim))
inline void UnitCell<Dim>::draw_band_dispersion_2D() const
{
	plt::figure(true);
	plt::hold(plt::on);

	auto K_x = utils::linspace(-pi * 1.1, pi * 1.1, 50) | std::ranges::to<std::vector>();
	auto K_y = utils::linspace(-pi * 1.1, pi * 1.1, 50) | std::ranges::to<std::vector>();

	auto [K_X, K_Y] = utils::meshgrid(K_x, K_y);
	auto E 
		= utils::nested_vector_transform_load([this](double kx, double ky) { return Hamiltonian_k_eigenvalues({ kx, ky, 0. }); },
			[](Eigen::VectorXd v, int i) { return v(i); },
		atom_pos.size(),
		K_X, K_Y);

	for (auto&& e : E) {
		plt::surf(K_X, K_Y, e)->face_alpha(0.5).line_width(0);
	}

	plt::xlabel("k_x");
	plt::ylabel("k_y");
	plt::zlabel("E");
	if (auto_show) plt::show();
}

template<int Dim> requires (is_lattice_dim(Dim))
inline std::vector<LatticeVector<Dim>> UnitCell<Dim>::get_NN_vectors(const auto& nn)
{
	auto NN = make_cartesian<Dim>(nn);
	std::vector<LatticeVector<Dim>> NN_vectors;
	for (const auto& v_tuple : NN) {
		auto v = get_vector_from_tuple(v_tuple);
		if (v.isZero()) continue;
		NN_vectors.emplace_back(v);
	}
	return NN_vectors;
}

// generate hoppings up to the NN-th nearest neighbors
template<int Dim> requires (is_lattice_dim(Dim))
inline void UnitCellNN<Dim>::generate_hoppings_to(HoppingType NN)
{
	if (NN_lattice_registered.empty()) add_cell(LatticeVector<Dim>::Zero());
	while (NN_solved.size() <= NN) {
		auto [d, hp] = NN_atoms_to_add.top();
		double d0 = d;
		std::vector<Hopping<Dim>> one_NN_hoppings;
		while (abs(d - d0) < 1e-7) {
			for (auto&& Delta_x: UnitCell<Dim>::NN_vectors) add_cell(std::get<2>(hp) + Delta_x); // get<2> gets the lat vec
			one_NN_hoppings.emplace_back(hp);
			NN_atoms_to_add.pop();
			std::tie(d, hp) = NN_atoms_to_add.top();
		}
		NN_solved.emplace_back(one_NN_hoppings);
	}
}

// if an atom is processed, add the NN cells wrt the current cell to the candidate queue
template<int Dim> requires (is_lattice_dim(Dim))
inline void UnitCellNN<Dim>::add_cell(LatticeVector<Dim> a)
{
	if (NN_lattice_registered.find(a) != NN_lattice_registered.end()) return;
	NN_lattice_registered.insert(a);
	for (int B: std::views::iota(0, cell_ptr->get_atom_number())) {
		auto&& hp = std::make_tuple(A, B, a);
		NN_atoms_to_add.emplace(cell_ptr->distance_of_hopping(hp), hp);
	}
}

template<int Dim> requires (is_lattice_dim(Dim))
struct UnitCellNN<Dim>::LatticeVectorHash
{
	std::size_t operator()(const LatticeVector<Dim>& v) const
	{
		std::size_t seed = 0;
		for (int i = 0; i < Dim; i++)
		{
			const std::size_t h = std::hash<int>{}(v(i));
			seed ^= h + 0x9e3779b9 + (seed << 6) + (seed >> 2);
		}
		return seed;
	}
};

template<int Dim> requires (is_lattice_dim(Dim))
struct UnitCell<Dim>::HoppingHash
{
	std::size_t operator()(const Hopping<Dim>& h) const
	{
		std::size_t seed = 0;
		auto&& [from, to, lattice_vector] = h;
		seed ^= std
			::hash<int>{}(from)+0x9e3779b9 + (seed << 6) + (seed >> 2);
		seed ^= std::hash<int>{}(to)+0x9e3779b9 + (seed << 6) + (seed >> 2);
		for (int i = 0; i < Dim; i++)
		{
			const std::size_t h = std::hash<int>{}(lattice_vector(i));
			seed ^= h + 0x9e3779b9 + (seed << 6) + (seed >> 2);
		}
		return seed;
	}
};