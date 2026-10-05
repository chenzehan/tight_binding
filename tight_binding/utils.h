#pragma once

#include <vector>
#include <ranges>
#include <iostream>
#include <fstream>
#include <algorithm>

namespace utils {
	auto range(double start, double step, int n)
	{
		return std::views::iota(0, n) | std::views::transform([=](auto i) { return start + i * step; });
	}

	auto linspace(double start, double end, int n, bool endpoint = false)
	{
		if (n <= 1) std::cerr << "linspace n should be larger than 1\n";
		double step = (end - start) / n;
		if (endpoint) step = (end - start) / (n - 1);
		return range(start, step, n);
	}

	auto arange(double start, double end, double step)
	{
		int n = static_cast<int>(floor((end - start) / step));
		return range(start, step, n);
	}

	template <typename T>
	auto transposed_nested_vector(const std::vector<std::vector<T>>& vec) {
		std::vector<std::vector<T>> result;
		if (vec.empty()) return result;
		result.resize(vec[0].size());
		for (auto&& v : vec) {
			for (int i = 0; i < v.size(); ++i) {
				result[i].push_back(v[i]);
			}
		}
		return result;
	}

	template<typename... Containers>
	void output_to_file(const std::string& filename, const Containers&... vectors) {
		std::vector<std::size_t> vec_size_vec{ vectors.size()... };
		auto ref_size = vec_size_vec[0];
		// get size
		for (auto i : vec_size_vec) if (i != ref_size) {
			std::cerr << "Not equal size\n";
			return;
		}

		std::ofstream out(filename);
		if (!out.is_open()) {
			std::cerr << "Fail to open " << filename << std::endl;
			return;
		}
		auto output_element = [&out](auto&& v) {
			if constexpr (std::ranges::input_range<decltype(v)>)
				for (auto&& e : v) out << e << " ";
			else
				out << v << " ";
		};
		// output as columns
		for (std::size_t i = 0; i < ref_size; ++i) {
			(output_element(vectors[i]), ...);
			out << std::endl;
		}
		out.close();
	}

	template <typename T>
	struct NestedVectorDepth {
		static constexpr int value = 0;
		using Type = T;
	};

	template <typename T>
	struct NestedVectorDepth<std::vector<T>> {
		static constexpr int value = NestedVectorDepth<T>::value + 1;
		using Type = typename NestedVectorDepth<T>::Type;
	};

	template <int N, typename T>
	struct NestedVectorType {
		using Type = std::vector<typename NestedVectorType<N - 1, T>::Type>;
	};

	template <typename T>
	struct NestedVectorType<0, T> {
		using Type = T;
	};

	template <int N, typename T>
	using NestedVectorType_t = typename NestedVectorType<N, T>::Type;

	template<int N, typename ...Args>
	auto nested_vector_transform_no_check(const auto& func, const Args&... args)
	{
		if constexpr (N == 0) {
			return func(args...);
		}
		else if constexpr (N == 1) {
			return std::views::zip(args...) | std::views::transform([&](auto&& t) { return std::apply(func, t); }) | std::ranges::to<std::vector>();
		}
		else return std::views::zip(args...) // to vector<tuple<vector, vector, ...>>
			| std::views::transform([&](auto&& t) { // t is tuple<vector, vector, ...>
			return std::apply(
				[&](auto&& ...args) { // Currying the function
					return nested_vector_transform_no_check<N - 1>(func, args...);
				}, t);
				}) | std::ranges::to<std::vector>();
	}

	// transform a list of vector<vector<...>> to vector<vector<...>> using func(args...)
	template<typename ...Args>
	auto nested_vector_transform(const auto& func, const Args&... args)
	{
		constexpr std::array<int, sizeof...(Args)> depths = { NestedVectorDepth<Args>::value... };
		static_assert(std::all_of(depths.begin(), depths.end(), [&](int d) { return d == depths[0]; }), "Depths are not equal");
		static_assert(std::is_invocable_v<decltype(func), typename NestedVectorDepth<Args>::Type...>, "Function is not invocable, type mismatch");
		constexpr int depth = depths[0];
		return nested_vector_transform_no_check<depth>(func, args...);
	}

	// no check. Another version of nested_vector_transform_no_check,
	// split vector<vector<...>> of some (x, y, z) to [X, Y, Z], X is vector<vector<...<T>>>
	template<int N, typename T, typename ...Args>
	auto nested_vector_transform_load_no_check(const auto& func, const auto& get_i, int n, const Args&... args)
	{
		int vec_size = std::array{ args.size()... }[0];
		if constexpr (N == 0) {
			auto&& expl_result = func(args...);
			std::vector<T> result(n);
			for (int i = 0; i < n; i++) {
				result[i] = get_i(expl_result, i);
			}
			return result;
		}
		else if constexpr (N == 1) {
			auto expl_result = std::views::zip(args...) | std::views::transform([&](auto&& t) { return std::apply(func, t); });
			typename NestedVectorType<N + 1, T>::Type result(n);
			for (auto& r: result) r.reserve(vec_size);
			for (auto&& v : expl_result) {
				for (int i = 0; i < n; i++) {
					result[i].emplace_back(get_i(v, i));
				}
			}
			return result;
		}
		else {
			auto expl_result = std::views::zip(args...) // to vector<tuple<vector, vector, ...>>
				| std::views::transform([&](auto&& t) { // t is tuple<vector, vector, ...>
				return std::apply(
					[&](auto&& ...args) { // Currying the function
						return nested_vector_transform_load_no_check<N - 1, T>(func, get_i, n, args...);
					}, t);
					});
			typename NestedVectorType<N + 1, T>::Type result(n);
			for (auto& r : result) r.reserve(vec_size);
			for (auto&& v : expl_result) {
				for (int i = 0; i < n; i++) {
					result[i].emplace_back(v[i]);
				}
			}
			return result;
		}
	}

	// Another version of nested_vector_transform_no_check,
	// split vector<vector<...>> of some (x, y, z) to [X, Y, Z], X is vector<vector<...<T>>>
	// auotmatically retrieve the type of T and check the get_i function
	template<typename ...Args>
	auto nested_vector_transform_load(const auto& func, const auto& get_i, int n, const Args&... args)
	{
		constexpr std::array<int, sizeof...(Args)> depths = { NestedVectorDepth<Args>::value... };
		static_assert(std::all_of(depths.begin(), depths.end(), [&](int d) { return d == depths[0]; }), "Depths are not equal");
		static_assert(std::is_invocable_v<decltype(func), typename NestedVectorDepth<Args>::Type...>, "Function is not invocable, type mismatch");
		using F_t = std::invoke_result<decltype(func), typename NestedVectorDepth<Args>::Type...>::type;
		static_assert(std::is_invocable_v<decltype(get_i), F_t, int>, "get_i is not invocable, type mismatch");
		constexpr int depth = depths[0];
		using T = std::invoke_result<decltype(get_i), F_t, int>::type;
		return nested_vector_transform_load_no_check<depth, T>(func, get_i, n, args...);
	}

	auto meshgrid(const std::vector<double>& x, const std::vector<double>& y)
	{
		std::vector<std::vector<double>> X, Y;
		for (auto&& xi : x) {
			X.emplace_back(y.size(), xi);
			Y.push_back(y);
		}
		return std::make_pair(X, Y);
	}
}