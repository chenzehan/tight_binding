#pragma once

#define _SILENCE_CXX23_DENORM_DEPRECATION_WARNING
#include <vector>
#include <ranges>
#include <limits>
#include <queue>

#include <Eigen/Dense>

namespace spatial_geometry {
	using namespace Eigen;

	namespace affine_3D {
		// P & Q are affine points, last element is 1
		RowVector4d perp_bisector(const Vector4d& P, const Vector4d& Q)
		{
			Vector4d PQ = Q - P;
			return PQ - Vector4d(0., 0., 0., 0.5 * (P + Q).dot(PQ));
		}
		// 3D cross product, returns the crossing point of the 3 affine planes using ( (i,j,k,w), alpha, beta, gamma )^T
		Vector4d crossing_point(const RowVector4d& alpha, const RowVector4d& beta, const RowVector4d& gamma)
		{
			Eigen::Matrix<double, 3, 4> M;
			M.row(0) = alpha;
			M.row(1) = beta;
			M.row(2) = gamma;
			auto cofactor = [&M](int col) {
				Eigen::Matrix<double, 3, 3> subM;
				subM << M.leftCols(col), M.rightCols(M.cols() - col - 1);
				return subM.determinant();
				};
			auto&& P = Vector4d(cofactor(0), -cofactor(1), cofactor(2), -cofactor(3));
			if (abs(P(3)) < 1e-8) return P;
			return P / P(3);
		}
		double distance_to_plane(const Vector4d& P, const RowVector4d& alpha)
		{
			// assume PQ perp to ¦Á, Q on ¦Á
			// ¦Á[0:3] // (P - Q), so ¦Ë * ¦Á[0:3] = P - Q
			// also ¦Á ¡¤ Q = 0
			// so ¦Ë * ¦Á[0:3] ^ 2 + ¦Á ¡¤ P = 0
			// |PQ| = ¦Ë * |¦Á[0:3]| = sqrt(¦Ë^2 * |¦Á[0:3]|^2) = |¦Á ¡¤ Q| / |¦Á[0:3]|
			return abs(alpha.dot(P)) / alpha.head<3>().norm();
		}
		Vector4d project_point(const Vector4d& P, const RowVector4d& alpha)
		{
			double lambda = -alpha.dot(P) / alpha.head<3>().squaredNorm();
			RowVector4d a(alpha);
			a(3) = 0;
			return P + lambda * a.transpose();
		}
		bool on_the_plane(const Vector4d& P, const RowVector4d& alpha)
		{
			return abs(alpha.transpose().dot(P)) < 1e-8;
		}
		Vector4d to_affine(const Vector3d& P)
		{
			return Vector4d(P(0), P(1), P(2), 1);
		}
		Vector3d to_euclidean(const Vector4d& P)
		{
			return P.head<3>() / P(3);
		}
	}
	
	namespace affine_2D {
		RowVector3d perp_bisector(const Vector3d& P, const Vector3d& Q)
		{
			Vector3d PQ = Q - P;
			return (PQ - Vector3d(0., 0., 0.5 * (P + Q).dot(PQ))).transpose();
		}
		Vector3d crossing_point(const RowVector3d& alpha, const RowVector3d& beta)
		{
			auto&& P = alpha.cross(beta).transpose();
			if (abs(P(2)) < 1e-10) return P;
			return P / P(2);
		}
		RowVector3d crossing_line(const Vector3d& P, const Vector3d& Q)
		{
			return P.cross(Q).transpose();
		}
		double distance_to_line(const Vector3d& P, const RowVector3d& alpha)
		{
			return abs(alpha.dot(P)) / alpha.head<2>().norm();
		}
		Vector3d project_point(const Vector3d& P, const RowVector3d& alpha)
		{
			double lambda = -alpha.dot(P) / alpha.head<2>().squaredNorm();
			RowVector3d a(alpha);
			a(2) = 0;
			return P + lambda * a.transpose();
		}
		Vector3d to_affine(const Vector2d& P)
		{
			return Vector3d(P(0), P(1), 1);
		}
		Vector2d to_euclidean(const Vector3d& P)
		{
			return P.head<2>() / P(2);
		}
		double cross_2D(const Vector2d& P, const Vector2d& Q)
		{
			return P(0) * Q(1) - P(1) * Q(0);
		}
		bool is_inf_point(const Vector3d& P)
		{
			return abs(P(2)) < 1e-8;
		}
		bool on_the_line(const Vector2d& P, const RowVector3d& alpha)
		{
			return abs(alpha.transpose().dot(to_affine(P))) < 1e-8;
		}
		bool same_side(const Vector3d& P, const Vector3d& Q, const RowVector3d& alpha)
		{
			return alpha.transpose().dot(P) * alpha.transpose().dot(Q) > 0;
		}
		Vector2d two_point_direction(const Vector3d& P, const Vector3d& Q)
		{
			if (is_inf_point(P)) return -P.head<2>();
			if (is_inf_point(Q)) return Q.head<2>();
			return (Q - P).head<2>();
		}
		bool is_parallel(const RowVector3d& alpha, const RowVector3d& beta)
		{
			return abs(cross_2D(alpha.head<2>(), beta.head<2>())) < 1e-8;
		}
	}

	namespace minimal_complex {
		struct ConvexPolygonAffine {
			Vector3d center;
			std::vector<Vector3d> vertices; // counter-clockwise
			// -1: inside, 0: on the edge, 1: outside
			int point_relation(Vector3d P) const
			{
				for (int i = 0; i < vertices.size(); ++i) {
					Vector3d v1 = vertices[i];
					Vector3d v2 = vertices[(i + 1) % vertices.size()];
					double line_relation = 
						affine_2D::cross_2D(affine_2D::two_point_direction(v1, v2), affine_2D::two_point_direction(v1, P));
					if (abs(line_relation) < 1e-8) return 0;
					if (line_relation < 0) return 1;
				}
				return -1;
			}
			//// add affine line, requires that at most one point in the origianal polygon removed
			//// return false if failed
			//bool add_edge_affine(RowVector3d alpha) {
			//	double center_alpha_rel = alpha.transpose().dot(center);
			//	for (auto it = vertices.begin(); it != vertices.end(); it++) {
			//		Vector3d v = *it;
			//		if (v.dot(alpha.transpose()) * center_alpha_rel > -1e-8) continue; // v and center on the same side of alpha
			//		// now v must be on the opposite side of center, to be removed
			//		Vector3d v_next = (it + 1 != vertices.end() ? *(it + 1) : vertices.front());
			//		Vector3d v_before = (it != vertices.begin() ? *(it - 1) : vertices.back());
			//		Vector3d P = affine_2D::crossing_point(alpha, affine_2D::crossing_line(v_before, v));
			//		Vector3d Q = affine_2D::crossing_point(alpha, affine_2D::crossing_line(v, v_next));
			//		*it = Q;
			//		vertices.insert(it, P);
			//		return true;
			//	}
			//	return false;
			//}
			auto get_with_closed_end() -> std::tuple<std::vector<double>, std::vector<double>> const {
				std::vector<double> x, y;
				x.reserve(vertices.size() + 1);
				y.reserve(vertices.size() + 1);
				for (auto&& v : vertices) {
					x.push_back(v(0));
					y.push_back(v(1));
				}
				//x.push_back(x.front());
				//y.push_back(y.front());
				std::cout << x.size() << std::endl;
				return { x, y };
			}
			ConvexPolygonAffine(Vector3d center, const std::vector<RowVector3d>& sorted_lines)
				: center(center)
			{
			}
		};
		ConvexPolygonAffine minimal_polygon_affine(Vector3d O, std::vector<RowVector3d>& lines)
		{
			std::sort(lines.begin(), lines.end(), [&O](const auto& L1, const auto& L2) {
				return affine_2D::distance_to_line(O, L1) < affine_2D::distance_to_line(O, L2);
				});
			return ConvexPolygonAffine(O, lines);
		}
		// Not stable algorithm
		template<typename Point, typename Line>
		std::vector<Point> minimal_polygon(const Point& O, const std::vector<Line>& lines,
			const auto& distance_to_line, const auto& project_to_line,
			const auto& crossing_point, const auto& parallel, const auto& right_direction, std::vector<int>* line_recorder = nullptr)
		{
			using PointVec = Point;
			auto starting_line = std::ranges::min_element(lines, [&O, &distance_to_line](const auto& L1, const auto& L2) {
				return distance_to_line(O, L1) < distance_to_line(O, L2);
				});
			Point S = project_to_line(O, *starting_line);
			PointVec dragging_vec = S - O;
			std::vector<Point> polygon;

			auto current_line = starting_line;
			do {
				// find next point and line
				Point P_next{};
			    PointVec new_dragging_vec{};
				double current_dis = std::numeric_limits<double>::infinity();
				auto line_it = lines.begin();
				auto next_line = lines.end();
				for (; line_it < lines.end(); ++line_it) {
					if (parallel(*current_line, *line_it)) continue;
					Point P = crossing_point(*current_line, *line_it);
					PointVec new_vec = P - S;
					if (!right_direction(dragging_vec, new_vec)) continue;
					if (double new_dis = new_vec.norm();
						(new_dis < current_dis - 1e-8 || /* some of the conditions are to avoid coincide of lines*/
							(abs(new_dis - current_dis) < 1e-8 && distance_to_line(O, *line_it) < distance_to_line(O, *next_line))
						)
						&& new_dis > 1e-8) { // must be finite step
						current_dis = new_dis;
						P_next = P;
			            new_dragging_vec = new_vec;
						next_line = line_it;
					}
				}
				if (next_line == lines.end()) break;
				polygon.push_back(P_next);
			    dragging_vec = new_dragging_vec;
				S = P_next;
				current_line = next_line;
				if (polygon.size() > lines.size() + 1) break;
				if (line_recorder) line_recorder->push_back(std::distance(lines.begin(), current_line));
			} while (current_line != starting_line);
			return polygon;
		}

		// NOT stable algorithm
		auto minimal_3D_complex(const Vector4d& O, std::vector<RowVector4d>& planes)
		{
			std::sort(planes.begin(), planes.end(), [&O](const auto& L1, const auto& L2) {
				return affine_3D::distance_to_plane(O, L1) < affine_3D::distance_to_plane(O, L2);
				});
			auto starting_plane = planes.begin();
			using D_Vec = std::vector<double>;
			std::vector<std::tuple<D_Vec, D_Vec, D_Vec>> result;
			std::queue<int> q;
			std::vector<bool> visited(planes.size(), false);
			q.push(std::distance(planes.begin(), starting_plane));
			visited[q.front()] = true;
			while (!q.empty()) {
				auto& current_plane = planes[q.front()];
				q.pop();
				auto O_on_plane = affine_3D::project_point(O, current_plane);
				std::vector<int> next_planes;
				auto distance_to_line = [&current_plane](const Vector4d& P, const RowVector4d& alpha) -> double {
					if (RowVector3d(alpha.head<3>() - current_plane.head<3>()).squaredNorm() < 1e-8) return std::numeric_limits<double>::infinity();
					RowVector4d P_proj_af = affine_3D::project_point(P, alpha) - P;
					Vector3d P_proj = P_proj_af.head<3>();
					RowVector3d normal = current_plane.head<3>();
					double cos_theta = P_proj.transpose().dot(normal);
					cos_theta /= P_proj.norm();
					cos_theta /= normal.norm();
					return P_proj.norm() / sqrt(1 - cos_theta * cos_theta);
					};
				auto project_to_line = [&current_plane](const Vector4d& P, const RowVector4d& alpha) -> Vector4d {
					RowVector4d P_proj_af = affine_3D::project_point(P, alpha) - P;
					Vector3d P_proj = P_proj_af.head<3>();
					RowVector3d normal = current_plane.head<3>();
					Vector3d line_direction = normal.transpose().cross(P_proj);
					double w = -line_direction.dot(P.head<3>()) / P(3);
					RowVector4d perp_plane;
					perp_plane << line_direction.transpose(), w;
					return affine_3D::crossing_point(current_plane, alpha, perp_plane);
					};
				auto polygon = minimal_polygon(O_on_plane, planes, distance_to_line, project_to_line,
					[&current_plane](const RowVector4d& L1, const RowVector4d& L2)
						{ return affine_3D::crossing_point(current_plane, L1, L2); },
					[&current_plane](const RowVector4d& L1, const RowVector4d& L2) 
					{ return abs(affine_3D::crossing_point(current_plane, L1, L2)(3)) < 1e-8; },
					[&current_plane](const Vector4d& d1, const Vector4d& d2) {
						Eigen::Matrix<double, 3, 3> M;
						M << current_plane.transpose().head<3>(), d1.head<3>(), d2.head<3>();
						return M.determinant() > 0;
						},
					&next_planes
				);
				for (auto i : next_planes) {
					if (!visited[i]) {
						q.push(i);
						visited[i] = true;
					}
				}
				if (polygon.empty()) continue;
				auto&& polygon_x = polygon | std::views::transform([](const auto& P) { return P(0); }) | std::ranges::to<std::vector>();
				auto&& polygon_y = polygon | std::views::transform([](const auto& P) { return P(1); }) | std::ranges::to<std::vector>();
				auto&& polygon_z = polygon | std::views::transform([](const auto& P) { return P(2); }) | std::ranges::to<std::vector>();
				result.push_back({ polygon_x, polygon_y, polygon_z });
			}
			return result;
		}

	}
}