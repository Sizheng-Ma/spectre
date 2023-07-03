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
                  const size_t l_max, const size_t number_of_radial_points,
                  const std::vector<std::vector<double>>& spacetime_metric,
                  const std::vector<std::vector<double>>& pi,
                  const std::vector<std::vector<std::vector<double>>>& phi);
void ccm_functions(std::vector<double>& re_h, std::vector<double>& im_h,
                   const size_t l_max, const size_t number_of_radial_points,
                   const std::vector<std::vector<double>>& spacetime_metric,
                   const std::vector<std::vector<double>>& pi,
                   const std::vector<std::vector<std::vector<double>>>& phi);

// struct test {
//   using a = tmpl::list<>;
// };
