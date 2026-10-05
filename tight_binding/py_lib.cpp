#ifdef PYTHON_LIB

#define _SILENCE_CXX23_DENORM_DEPRECATION_WARNING
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

#include "unit_cell.h"
#include "models.h"
#include "pybind_plot.h"

template <int Dim> requires (is_lattice_dim(Dim))
void templ_pybind_unit_cell(py::module& m, const std::string& typestr) {
	py::class_<UnitCell<Dim>>(m, typestr.c_str())
		.def(py::init<std::vector<Vector3d>&&, std::unordered_map<std::string, Vector3d>&&>())
		.def("add_hopping", py::overload_cast<
			const std::string&, const std::string&, lattice_vector<Dim>, Complex>(&UnitCell<Dim>::add_hopping))
		.def("add_hopping", py::overload_cast<
			const std::string&, hopping_type, Complex>(&UnitCell<Dim>::add_hopping))
		/*.def("atom_position",	py::overload_cast<
			const std::string&>(&UnitCell<Dim>::atom_position))
		.def("atom_position",	py::overload_cast<
			const std::string&, lattice_vector<Dim>>(&UnitCell<Dim>::atom_position))
		.def("distance_between_lattices", py::overload_cast<
			lattice_vector<Dim>>(&UnitCell<Dim>::distance_between_lattices))
		.def("distance_between_lattices", py::overload_cast<
			lattice_vector<Dim>, lattice_vector<Dim>>(&UnitCell<Dim>::distance_between_lattices))*/
		.def("distance_of_hopping",			&UnitCell<Dim>::distance_of_hopping)
		.def("displacement_of_lattice",		&UnitCell<Dim>::displacement_of_lattice)
		.def("displacement_of_hopping",		&UnitCell<Dim>::displacement_of_hopping)
		.def("Hamiltonian_k",				&UnitCell<Dim>::Hamiltonian_k)
		.def("Hamiltonian_k_lower_tri",		&UnitCell<Dim>::Hamiltonian_k_lower_tri)
		//.def("Hamiltonian_k_eigenvalues",	&UnitCell<Dim>::Hamiltonian_k_eigenvalues)
		.def("get_band_dispersion_as_vecs", &UnitCell<Dim>::get_band_dispersion_as_vecs)
		.def("get_atom_number",				&UnitCell<Dim>::get_atom_number)
		.def("get_atom_index",				&UnitCell<Dim>::get_atom_index)
		//.def("atom_position",				&UnitCell<Dim>::atom_position)
		.def("atom_exists",					&UnitCell<Dim>::atom_exists)
		.def("draw_hoppings",				&UnitCell<Dim>::draw_hoppings)
		.def("draw_band_dispersion",		&UnitCell<Dim>::draw_band_dispersion)
		.def("get_first_Brillouin_zone",	&UnitCell<Dim>::get_first_Brillouin_zone)
		.def("draw_k_space",				&UnitCell<Dim>::draw_k_space);
}

void test_plot(const std::vector<double>& X, const std::vector<double>& Y, const std::vector<double>& Z) {
	pybind_plot::plot3(X, Y, Z)->color("gray").line_width(2);
}

PYBIND11_MODULE(tight_binding, m) {

	templ_pybind_unit_cell<1>(m, "UnitCell1D");
	templ_pybind_unit_cell<2>(m, "UnitCell2D");
	templ_pybind_unit_cell<3>(m, "UnitCell3D");

	py::module models = m.def_submodule("models");
	models.attr("graphene")			= py::cast(&models::graphene);
	models.attr("Honeycomb")		= py::cast(&models::Honeycomb);
	models.attr("FCC")				= py::cast(&models::FCC);
	models.attr("stub_1d_X")		= py::cast(&models::stub_1d_X);
	models.attr("sawtooth_chain_X") = py::cast(&models::sawtooth_chain_X);

	models.def("stub_1d_p",			&models::stub_1d_p);
	models.def("sawtooth_chain_p",	&models::sawtooth_chain_p);

	m.def("test_plot", &test_plot);
}

#endif