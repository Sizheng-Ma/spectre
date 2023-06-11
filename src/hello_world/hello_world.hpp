// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <vector>

void myprint();
int mynewfunction(int a);
void print_data_vector();
void ccm_functions(std::vector<double>& psi0,
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
