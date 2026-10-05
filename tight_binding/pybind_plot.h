#pragma once
#ifdef PYTHON_LIB

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <memory>

namespace pybind_plot {
	namespace py = pybind11;

	struct PlotArgs {
		py::dict kwargs;
		PlotArgs& line_width(double width) {
			kwargs["linewidth"] = width;
			return *this;
		}
		PlotArgs& color(std::string&& color) {
			kwargs["color"] = color;
			return *this;
		}
		PlotArgs& marker(std::string&& marker) {
			kwargs["marker"] = marker;
			return *this;
		}
		PlotArgs& linestyle(std::string&& linestyle) {
			kwargs["linestyle"] = linestyle;
			return *this;
		}
		PlotArgs& legend_string(std::string&& label) {
			kwargs["label"] = label;
			return *this;
		}
		PlotArgs& face_alpha(double alpha) {
			kwargs["alpha"] = alpha;
			return *this;
		}
		PlotArgs& edge_color(std::string&& color) {
			kwargs["edge_color"] = color;
			return *this;
		}
	};

	struct plot_;
	struct scatter_;
	struct arrow_;
	struct text_;
	struct plot3_;
	struct surf_;
	using NestedVec2D = std::vector<std::vector<double>>;
	py::array_t<double> convertToNumPyArray(const NestedVec2D& data);

	class Plotter {
	public:
		py::object handle_;
		inline static py::module plt;
		inline static py::object ca;
		inline static py::object ax_3d;

		Plotter() {
			if (!plt) plt = py::module::import("matplotlib.pyplot");
			handle_ = plt.attr("figure")();
		}
		Plotter(py::object h) {
			if (!plt) plt = py::module::import("matplotlib.pyplot");
			handle_ = h;
		}
		static py::module& get() {
			if (!plt) plt = py::module::import("matplotlib.pyplot");
			return plt;
		}
		static py::object& gca() {
			if (!ca) ca = get().attr("gca")();
			return ca;
		}
		static py::object& get_axes_3d() {
			if (!ax_3d) ax_3d = get().attr("axes")(py::arg("projection") = "3d");
			return ax_3d;
		}
		std::unique_ptr<plot_> plot(const std::vector<double>& X, const std::vector<double>& Y, std::string&& prop = "") {
			auto plt_ptr = std::make_unique<plot_>(X, Y, std::forward<std::string>(prop), handle_);
			return plt_ptr;
		}
		std::unique_ptr<arrow_> arrow(double x0, double y0, double x1, double y1, std::string&& prop = "") {
			auto plt_ptr = std::make_unique<arrow_>(x0, y0, x1, y1, std::forward<std::string>(prop), handle_);
			return plt_ptr;
		}
		std::unique_ptr<text_> text(double x, double y, const std::string& s) {
			auto plt_ptr = std::make_unique<text_>(x, y, s, handle_);
			return plt_ptr;
		}
		std::unique_ptr<scatter_> scatter(const std::vector<double>& X, const std::vector<double>& Y, std::string&& prop = "") {
			auto plt_ptr = std::make_unique<scatter_>(X, Y, std::forward<std::string>(prop), handle_);
			return plt_ptr;
		}

		std::unique_ptr<plot3_> plot3(const std::vector<double>& X, const std::vector<double>& Y, const std::vector<double>& Z, std::string&& prop = "") {
			auto plt_ptr = std::make_unique<plot3_>(X, Y, Z, std::forward<std::string>(prop), handle_);
			return plt_ptr;
		}
		std::unique_ptr<surf_> surf(const NestedVec2D& X, const NestedVec2D& Y, const NestedVec2D& Z) {
			auto plt_ptr = std::make_unique<surf_>(X, Y, Z, handle_);
			return plt_ptr;
		}

		void xlim(std::pair<double, double> lim) const {
			handle_.attr("set_xlim")(lim.first, lim.second);
		}
		void ylim(std::pair<double, double> lim) const {
			handle_.attr("set_ylim")(lim.first, lim.second);
		}
		void zlim(std::pair<double, double> lim) const {
			handle_.attr("set_zlim")(lim.first, lim.second);
		}
		void xlabel(std::string&& label) const {
			handle_.attr("set_xlabel")(label);
		}
		void ylabel(std::string&& label) const {
			handle_.attr("set_ylabel")(label);
		}
		void zlabel(std::string&& label) const {
			handle_.attr("set_zlabel")(label);
		}
		void title(std::string&& title) const {
			handle_.attr("set_title")(title);
		}
	};

	using axes_handle = std::unique_ptr<Plotter>;

	struct plot_ : PlotArgs {
		std::vector<double> X, Y, Z;
		std::string prop;
		py::object handle_;
		plot_(const std::vector<double>& X, const std::vector<double>& Y, std::string&& prop, py::object ax) :
			PlotArgs(), X(X), Y(Y), prop(prop), handle_(ax) {}
		~plot_() {
			handle_.attr("plot")(X, Y, prop, **kwargs);
		}
	};
	std::unique_ptr<plot_> plot(const std::vector<double>& X, const std::vector<double>& Y, std::string&& prop = "") {
		auto plt_ptr = std::make_unique<plot_>(X, Y, std::forward<std::string>(prop), Plotter::gca());
		return plt_ptr;
	}

	struct scatter_ : PlotArgs {
		std::vector<double> X, Y;
		std::string prop;
		py::object handle_;
		scatter_(const std::vector<double>& X, const std::vector<double>& Y, std::string&& prop, py::object ax) :
			PlotArgs(), X(X), Y(Y), prop(prop), handle_(ax) {}
		~scatter_() {
			handle_.attr("scatter")(X, Y, prop, **kwargs);
		}
	};
	std::unique_ptr<scatter_> scatter(const std::vector<double>& X, const std::vector<double>& Y, std::string&& prop = "") {
		auto plt_ptr = std::make_unique<scatter_>(X, Y, std::forward<std::string>(prop), Plotter::gca());
		return plt_ptr;
	}

	struct arrow_ : PlotArgs {
		double x0, y0, x1, y1;
		std::string prop;
		py::object handle_;
		arrow_(double x0, double y0, double x1, double y1, std::string&& prop, py::object ax) :
			PlotArgs(), x0(x0), y0(y0), x1(x1), y1(y1), prop(prop), handle_(ax) {}
		~arrow_() {
			handle_.attr("arrow")(x0, y0, x1 - x0, y1 - y0, **kwargs);
		}
	};
	std::unique_ptr<arrow_> arrow(double x0, double y0, double x1, double y1, std::string&& prop = "") {
		auto plt_ptr = std::make_unique<arrow_>(x0, y0, x1, y1, std::forward<std::string>(prop), Plotter::gca());
		return plt_ptr;
	}

	struct text_ : PlotArgs {
		double x, y;
		std::string s;
		py::object handle_;
		text_(double x, double y, const std::string& s, py::object ax) :
			PlotArgs(), x(x), y(y), s(s), handle_(ax) {}
		~text_() {
			handle_.attr("text")(x, y, s, **kwargs);
		}
	};
	std::unique_ptr<text_> text(double x, double y, const std::string& s) {
		auto plt_ptr = std::make_unique<text_>(x, y, s, Plotter::gca());
		return plt_ptr;
	}

	void figure(bool t) {}
	void hold(bool t) {}
	bool on = true;

	std::unique_ptr<Plotter> gca() {
		auto plt_ptr = std::make_unique<Plotter>(Plotter::gca());
		return plt_ptr;
	}

	struct plot3_ : PlotArgs {
		std::vector<double> X, Y, Z;
		std::string prop;
		py::object handle_;
		plot3_(const std::vector<double>& X, const std::vector<double>& Y, const std::vector<double>& Z, std::string&& prop, py::object ax) :
			PlotArgs(), X(X), Y(Y), Z(Z), prop(prop), handle_(ax) {}
		~plot3_(){
			handle_.attr("plot")(X, Y, Z, prop, **kwargs);
		}
	};
	std::unique_ptr<plot3_> plot3(const std::vector<double>& X, const std::vector<double>& Y, const std::vector<double>& Z, std::string&& prop = "") {
		auto plt_ptr = std::make_unique<plot3_>(X, Y, Z, std::forward<std::string>(prop), Plotter::get_axes_3d());
		return plt_ptr;
	}

	struct surf_ : PlotArgs {
		NestedVec2D X, Y, Z;
		py::object handle_;
		surf_(const NestedVec2D& X, const NestedVec2D& Y, const NestedVec2D& Z, py::object ax) :
			PlotArgs(), X(X), Y(Y), Z(Z), handle_(ax) {}
		~surf_() {
			handle_.attr("plot_surface")(convertToNumPyArray(X), convertToNumPyArray(Y), convertToNumPyArray(Z), **kwargs);
		}
	};
	std::unique_ptr<surf_> surf(const NestedVec2D& X, const NestedVec2D& Y, const NestedVec2D& Z) {
		auto plt_ptr = std::make_unique<surf_>(X, Y, Z, Plotter::get_axes_3d());
		return plt_ptr;
	}

	void show() {
		Plotter::get().attr("show")();
	}
	void cla() {
		Plotter::gca().attr("cla")();
	}
	void clf() {
		Plotter::get().attr("clf")();
	}
	void xlim(std::pair<double, double> lim) {
		// Plotter::get().attr("xlim")(lim.first, lim.second);
		gca()->xlim(lim);
	}
	void ylim(std::pair<double, double> lim) {
		// Plotter::get().attr("ylim")(lim.first, lim.second);
		gca()->ylim(lim);
	}
	void zlim(std::pair<double, double> lim) {
		gca()->zlim(lim);
	}
	void xlabel(std::string&& label) {
		gca()->xlabel(std::move(label));
	}
	void ylabel(std::string&& label) {
		gca()->ylabel(std::move(label));
	}
	void zlabel(std::string&& label) {
		gca()->zlabel(std::move(label));
	}
	void title(std::string&& title) {
		gca()->title(std::move(title));
	}

	void axis(std::string&& axis) {
		Plotter::gca().attr("axis")(axis);
		if (axis == "equal") {
			Plotter::gca().attr("set_aspect")("equal");
		}
	}

	py::array_t<double> convertToNumPyArray(const NestedVec2D& data) {

		size_t rows = data.size();
		size_t cols = (rows > 0) ? data[0].size() : 0;

		py::array_t<double> npArray({ rows, cols });
		auto npArrayPtr = npArray.mutable_unchecked<2>();

		for (size_t i = 0; i < rows; ++i) {
			for (size_t j = 0; j < cols; ++j) {
				npArrayPtr(i, j) = data[i][j];
			}
		}

		return npArray;
	}
}

#endif