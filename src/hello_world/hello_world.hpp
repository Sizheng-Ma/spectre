// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <complex>
#include <vector>

void myprint();
int mynewfunction(int a);
void print_data_vector();
void initialize_j(std::vector<double>& re_j, std::vector<double>& im_j,
                  const std::vector<std::complex<double>>& bondi_beta_spec,
                  const std::vector<std::complex<double>>& bondi_dr_j_spec,
                  const std::vector<std::complex<double>>& bondi_du_r_spec,
                  const std::vector<std::complex<double>>& bondi_h_spec,
                  const std::vector<std::complex<double>>& bondi_j_spec,
                  const std::vector<std::complex<double>>& bondi_q_spec,
                  const std::vector<std::complex<double>>& bondi_r_spec,
                  const std::vector<std::complex<double>>& bondi_u_spec,
                  const std::vector<std::complex<double>>& bondi_w_spec);
void ccm_functions(std::vector<double>& re_h, std::vector<double>& im_h,
                   const std::vector<double>& bondi_beta_spec,
                   const std::vector<double>& bondi_dr_j_spec,
                   const std::vector<double>& bondi_du_r_spec,
                   const std::vector<double>& bondi_h_spec,
                   const std::vector<double>& bondi_j_spec,
                   const std::vector<double>& bondi_q_spec,
                   const std::vector<double>& bondi_r_spec,
                   const std::vector<double>& bondi_u_spec,
                   const std::vector<double>& bondi_w_spec);

// struct test {
//   using a = tmpl::list<>;
// };
