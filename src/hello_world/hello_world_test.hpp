// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <complex>
#include <vector>

void ccm_functions11(std::vector<double>& re_h, std::vector<double>& im_h,
                   std::vector<double>& dt_cauchy_x,
                   std::vector<double>& dt_cauchy_y,
                   std::vector<double>& dt_cauchy_z,
                   std::vector<double>& dt_inertial_x,
                   std::vector<double>& dt_inertial_y,
                   std::vector<double>& dt_inertial_z,
                   std::vector<double>& re_psi3, std::vector<double>& im_psi3,
                   std::vector<double>& dt_u_scri, const size_t l_max,
                   const size_t number_of_radial_points,
                   const std::vector<std::vector<double>>& spacetime_metric,
                   const std::vector<std::vector<double>>& pi,
                   const std::vector<std::vector<std::vector<double>>>& phi,
                   const double radius, const std::vector<double>& re_j,
                   const std::vector<double>& im_j,
                   const std::vector<std::vector<double>>& cauchy_cart,
                   const std::vector<std::vector<double>>& inertial_cart);

// struct test {
//   using a = tmpl::list<>;
// };
