// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <complex>
#include <vector>

void myprint();
int mynewfunction(int a);
void print_data_vector();
size_t get_vector_size(const size_t l_max);
std::vector<double> transpose_wt_data(const std::vector<double>& data,
                                      const size_t l_max);

void initialize_j(std::vector<double>& re_j, std::vector<double>& im_j,
                  std::vector<double>& cauchy_x, std::vector<double>& cauchy_y,
                  std::vector<double>& cauchy_z,
                  std::vector<double>& inertial_x,
                  std::vector<double>& inertial_y,
                  std::vector<double>& inertial_z, const size_t l_max,
                  const size_t number_of_radial_points,
                  const std::vector<std::vector<double>>& spacetime_metric,
                  const std::vector<std::vector<double>>& pi,
                  const std::vector<std::vector<std::vector<double>>>& phi,
                  const double radius);
void ccm_functions(std::vector<double>& re_h, std::vector<double>& im_h,
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

namespace spectre {

struct MyScriPlusInterpolationManager;

struct InterpolationInterface {
 public:
  InterpolationInterface();
  ~InterpolationInterface();

  void clear();

  void insert_data(std::vector<double> data);

 private:
  MyScriPlusInterpolationManager* my_scri_plus_interpolation_manager_;
};
}  // namespace spectre

// struct test {
//   using a = tmpl::list<>;
// };
