#pragma once

#include "unit_cell.h"

namespace models {
    auto Honeycomb = UnitCell<2>{
        {
            { 3. / 2, -sqrt(3) / 2, 0},
            { 3. / 2,  sqrt(3) / 2, 0 }
        },
        {
            {"A", {0, 0, 0}},
            {"B", {1, 0, 0}}
        }
    };

    auto stub_1d_X = UnitCell<1>{
        {
            { 1, 0, 0 }
        },
        {
            {"A", {0, 0, 0 }},
            {"C", {0.5, 0, 0}},
            {"B", {0, 0.6, 0}}
        }
    };

	auto sawtooth_chain_X = UnitCell<1>{
		{
			{ 1, 0, 0 }
		},
		{
			{"A", {0, 0, 0 }},
			{"B", {0, 1, 0}}
		}
	};

    auto Lieb = UnitCell<2>{
		{
			{ 1, 0, 0 },
			{ 0, 1, 0 }
		},
		{
			{"A", {0, 0, 0}},
			{"B", {0.5, 0, 0}},
			{"C", {0, 0.5, 0}}
		}
	};

	auto Kagome = UnitCell<2>{
		{
			{ 1, 0, 0 },
			{ 0.5, sqrt(3) / 2, 0 }
		},
		{
			{"A", {0, 0, 0}},
			{"B", {0.5, 0, 0}},
			{"C", {0.25, sqrt(3) / 4, 0}}
		}
	};

	auto Square = UnitCell<2>{
		{
			{ 1, 0, 0 },
			{ 0, 1, 0 }
		},
		{
			{"A", {0, 0, 0}}
		}
	};

	auto Chain = UnitCell<1>{
		{
			{ 1, 0, 0 }
		},
		{
			{"A", {0, 0, 0}}
		}
	};

	auto FCC = UnitCell<3>{
		{
			{ 0.5, 0.5, 0 },
			{ 0.5, 0, 0.5 },
			{ 0, 0.5, 0.5 }
		},
		{
			{"A", {0, 0, 0}}
		}
	};

	auto Cubic = UnitCell<3>{
		{
			{ 1, 0, 0 },
			{ 0, 1, 0 },
			{ 0, 0, 1 }
		},
		{
			{"A", {0, 0, 0}}
		}
	};

	using namespace std::complex_literals;
	auto TI_sp = [] {
		auto p = UnitCell<2> {
		   {
			   {1, 0, 0},
			   {0, 1, 0}
		   },
		   {
			   {"s", {0, 0, 0}},
			   {"p", {0, 0, 0}}
		   }
		};
		Complex ep = 0.6, es = -ep, t_ss = 0.4, t_pp = -t_ss, t_sp = 0.2;
		p.add_hopping("s", "s", p.Intra_Cell, es);
		p.add_hopping("p", "p", p.Intra_Cell, ep);
		p.add_hopping("s", "s", { 1, 0 }, t_ss);
		p.add_hopping("s", "s", { 0, 1 }, t_ss);
		p.add_hopping("p", "p", { 1, 0 }, t_pp);
		p.add_hopping("p", "p", { 0, 1 }, t_pp);
		p.add_hopping("s", "p", { 1, 0 }, 1. * t_sp);
		p.add_hopping("s", "p", { 0, 1 }, 1.i * t_sp);
		p.add_hopping("s", "p", { -1, 0 }, -1. * t_sp);
		p.add_hopping("s", "p", { 0, -1 }, -1.i * t_sp);
		return p;
	}();

	auto graphene = [] {
		UnitCell<2> p = Honeycomb;
		p.add_hopping("A", 1_NN, 1.0);
		return p;
	}();

	auto stub_1d_p(double t, double t_) -> UnitCell<1> {
		UnitCell<1> p = stub_1d_X;
		p.add_hopping("A", "C", p.Intra_Cell, t);
		p.add_hopping("A", "C", p.Left_NN, t);
		p.add_hopping("A", "B", p.Intra_Cell, t_);
		return p;
	}

	// t_ = sqrt(2) * t leads to flat band
	auto sawtooth_chain_p(double t, double t_) -> UnitCell<1> {
		UnitCell<1> p = sawtooth_chain_X;
		p.add_hopping("A", "A", p.Right_NN, t);
		p.add_hopping("A", "B", p.Intra_Cell, t_);
		p.add_hopping("B", "A", p.Right_NN, t_);
		return p;
	}
}