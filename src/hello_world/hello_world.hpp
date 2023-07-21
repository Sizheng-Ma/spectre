// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <complex>
#include <deque>
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
void ccm_functions(
    std::vector<double>& re_h, std::vector<double>& im_h,
    std::vector<double>& dt_cauchy_x, std::vector<double>& dt_cauchy_y,
    std::vector<double>& dt_cauchy_z, std::vector<double>& dt_inertial_x,
    std::vector<double>& dt_inertial_y, std::vector<double>& dt_inertial_z,
    std::vector<double>& re_psi3, std::vector<double>& im_psi3,
    std::vector<std::complex<double>>& psi4, std::vector<double>& dt_u_scri,
    const size_t l_max, const size_t number_of_radial_points,
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
  InterpolationInterface(size_t target_number_of_points, size_t l_max,
                         size_t scri_output_density, size_t observation_l_max);
  ~InterpolationInterface();

  void clear();

  std::deque<std::pair<double, double>> get_u_bondi_ranges();

  void InsertInterpolationScriData(
      const double delta_time_spec, std::vector<double>& inertial_time,
      std::vector<std::complex<double>>& psi0,
      std::vector<std::complex<double>>& psi1,
      std::vector<std::complex<double>>& psi2,
      std::vector<std::complex<double>>& psi3,
      std::vector<std::complex<double>>& psi4,
      std::vector<std::complex<double>>& strain,
      std::vector<std::complex<double>>& news,
      std::vector<std::complex<double>>& eth_inertial_retarded_time);

  void ScriObserveInterpolated(
      std::vector<double>& eth_inertial_retarded_time_to_write,
      std::vector<double>& psi0_to_write, std::vector<double>& psi1_to_write,
      std::vector<double>& psi2_to_write, std::vector<double>& psi3_to_write,
      std::vector<double>& psi4_to_write, std::vector<double>& strain_to_write,
      std::vector<double>& news_to_write);

 private:
  MyScriPlusInterpolationManager* my_scri_plus_interpolation_manager_;
  size_t scri_output_density_, l_max_, observation_l_max_;
};
}  // namespace spectre

// struct test {
//   using a = tmpl::list<>;
// };
