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
                  const std::vector<std::complex<double>>& bondi_beta_spec,
                  const std::vector<std::complex<double>>& bondi_dr_j_spec,
                  const std::vector<std::complex<double>>& bondi_du_r_spec,
                  const std::vector<std::complex<double>>& bondi_h_spec,
                  const std::vector<std::complex<double>>& bondi_j_spec,
                  const std::vector<std::complex<double>>& bondi_q_spec,
                  const std::vector<std::complex<double>>& bondi_r_spec,
                  const std::vector<std::complex<double>>& bondi_u_spec,
                  const std::vector<std::complex<double>>& bondi_w_spec,
                  const size_t l_max, const size_t number_of_radial_points);
void ccm_functions(std::vector<double>& re_h, std::vector<double>& im_h,
                   const std::vector<std::complex<double>>& bondi_beta_spec,
                   const std::vector<std::complex<double>>& bondi_dr_j_spec,
                   const std::vector<std::complex<double>>& bondi_du_r_spec,
                   const std::vector<std::complex<double>>& bondi_h_spec,
                   const std::vector<std::complex<double>>& bondi_j_spec,
                   const std::vector<std::complex<double>>& bondi_q_spec,
                   const std::vector<std::complex<double>>& bondi_r_spec,
                   const std::vector<std::complex<double>>& bondi_u_spec,
                   const std::vector<std::complex<double>>& bondi_w_spec,
                   const size_t l_max, const size_t number_of_radial_points);

// struct test {
//   using a = tmpl::list<>;
// };
