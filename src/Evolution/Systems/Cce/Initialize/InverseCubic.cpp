// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/Initialize/InverseCubic.hpp"

#include <cstddef>
#include <memory>
#include <type_traits>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshInterpolation.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce::InitializeJ {

void spin_weight_1_coord_perturbation_heuristic1(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> gauge_c_step,
    const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> gauge_d_step,
    const SpinWeighted<ComplexDataVector, 0>& full_omega,
    const SpinWeighted<ComplexDataVector, 0>& omega_filtered,
    const SpinWeighted<ComplexDataVector, 0>& target_omega,
    const SpinWeighted<ComplexDataVector, 2>& /*gauge_c*/,
    const SpinWeighted<ComplexDataVector, 0>& gauge_d, const size_t l_max) {
  SpinWeighted<ComplexDataVector, 1> jacobian_supplement_f{full_omega.size()};

  // The alteration in each of the spin-weighted Jacobian factors determined
  // by linearizing the system in small \Delta \omega
  gauge_d_step->data() = full_omega.data() *
                         (target_omega.data() - omega_filtered.data()) /
                         gauge_d.data();
  Spectral::Swsh::angular_derivatives<
      tmpl::list<Spectral::Swsh::Tags::InverseEthbar>>(
      l_max, 1, make_not_null(&jacobian_supplement_f), *gauge_d_step);
  Spectral::Swsh::angular_derivatives<tmpl::list<Spectral::Swsh::Tags::Eth>>(
      l_max, 1, gauge_c_step, jacobian_supplement_f);
}

std::unique_ptr<InitializeJ<true>> InverseCubic<true>::get_clone() const {
  return std::make_unique<InverseCubic>();
}
std::unique_ptr<InitializeJ<false>> InverseCubic<false>::get_clone() const {
  return std::make_unique<InverseCubic>();
}

void InverseCubic<true>::apply(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_cauchy_coordinates,
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_inertial_coordinates,
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_inertial_coordinates,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_dr_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& r,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& beta, const size_t l_max,
    const size_t number_of_radial_points) {
  // use_input_modes_ = False use_beta_integral_estimate_ = False
  // optimize_l_0_mode_ = True
  double angular_coordinate_tolerance = 1.0e-13;
  size_t max_iterations = 1000;
  bool require_convergence = false;
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  Variables<tmpl::list<::Tags::TempSpinWeightedScalar<0, 2>,
                       ::Tags::TempSpinWeightedScalar<1, 2>,
                       ::Tags::TempSpinWeightedScalar<2, 2>,
                       ::Tags::TempSpinWeightedScalar<3, 0>,
                       ::Tags::TempSpinWeightedScalar<4, 0>,
                       ::Tags::TempSpinWeightedScalar<5, 0>,
                       ::Tags::TempSpinWeightedScalar<6, 0>,
                       ::Tags::TempSpinWeightedScalar<7, 0>,
                       ::Tags::TempSpinWeightedScalar<8, 0>,
                       ::Tags::TempSpinWeightedScalar<9, 2>,
                       ::Tags::TempSpinWeightedScalar<10, 2>,
                       ::Tags::TempSpinWeightedScalar<11, 2>>>
      buffers{number_of_angular_points};
  auto& surface_j_buffer = get<::Tags::TempSpinWeightedScalar<0, 2>>(buffers);
  auto& surface_dr_j_buffer =
      get<::Tags::TempSpinWeightedScalar<1, 2>>(buffers);
  auto& input_j_buffer =
      get(get<::Tags::TempSpinWeightedScalar<2, 2>>(buffers));
  auto& gauge_omega = get<::Tags::TempSpinWeightedScalar<4, 0>>(buffers);
  auto& filtered_gauge_omega =
      get(get<::Tags::TempSpinWeightedScalar<5, 0>>(buffers));
  auto& target_omega = get(get<::Tags::TempSpinWeightedScalar<6, 0>>(buffers));
  auto& interpolated_target_gauge_omega =
      get(get<::Tags::TempSpinWeightedScalar<7, 0>>(buffers));
  auto& surface_r_buffer = get<::Tags::TempSpinWeightedScalar<8, 0>>(buffers);

  auto& one_minus_y_coefficient =
      get(get<::Tags::TempSpinWeightedScalar<9, 2>>(buffers));
  auto& one_minus_y_cubed_coefficient =
      get(get<::Tags::TempSpinWeightedScalar<10, 2>>(buffers));
  auto& one_minus_y_fourth_coefficient =
      get(get<::Tags::TempSpinWeightedScalar<11, 2>>(buffers));

  Variables<tmpl::list<::Tags::ModalTempSpinWeightedScalar<0, 2>,
                       ::Tags::ModalTempSpinWeightedScalar<1, 0>>>
      modal_buffers{Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)};
  auto& input_j_libsharp_modes =
      get(get<::Tags::ModalTempSpinWeightedScalar<0, 2>>(modal_buffers));
  auto& gauge_omega_transform_buffer =
      get(get<::Tags::ModalTempSpinWeightedScalar<1, 0>>(modal_buffers));

  SpinWeighted<ComplexModalVector, 2> goldberg_modes{square(l_max + 1)};

  target_omega.data() = exp(2.0 * get(beta).data());

  void (*iteration_heuristic_function)(
      const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*>,
      const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*>,
      const SpinWeighted<ComplexDataVector, 0>&,
      const SpinWeighted<ComplexDataVector, 0>&,
      const SpinWeighted<ComplexDataVector, 0>&,
      const SpinWeighted<ComplexDataVector, 2>&,
      const SpinWeighted<ComplexDataVector, 0>&, size_t) = nullptr;

  iteration_heuristic_function = &spin_weight_1_coord_perturbation_heuristic1;

  auto iteration_function =
      [&iteration_heuristic_function, &filtered_gauge_omega, &gauge_omega,
       &target_omega, &interpolated_target_gauge_omega,
       &gauge_omega_transform_buffer, &l_max, &surface_r_buffer,
       &input_j_buffer,
       &r](const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
               gauge_c_step,
           const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
               gauge_d_step,
           const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
           const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
           const Spectral::Swsh::SwshInterpolator& iteration_interpolator) {
        get(gauge_omega).data() =
            0.5 * sqrt(get(gauge_d).data() * conj(get(gauge_d).data()) -
                       get(gauge_c).data() * conj(get(gauge_c).data()));
        iteration_interpolator.interpolate(
            make_not_null(&interpolated_target_gauge_omega), target_omega);
        filtered_gauge_omega = get(gauge_omega);
        double max_error = max(abs(filtered_gauge_omega.data() -
                                   interpolated_target_gauge_omega.data()));
        iteration_heuristic_function(make_not_null(&get(*gauge_c_step)),
                                     make_not_null(&get(*gauge_d_step)),
                                     get(gauge_omega), filtered_gauge_omega,
                                     interpolated_target_gauge_omega,
                                     get(gauge_c), get(gauge_d), l_max);
        return max_error;
      };

  auto finalize_function =
      [&gauge_omega, &l_max, &surface_dr_j_buffer, &boundary_dr_j, &boundary_j,
       &surface_j_buffer, &surface_r_buffer,
       &r](const Scalar<SpinWeighted<ComplexDataVector, 2>>& gauge_c,
           const Scalar<SpinWeighted<ComplexDataVector, 0>>& gauge_d,
           const tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>&
           /*angular_cauchy_coordinates*/,
           const Spectral::Swsh::SwshInterpolator& interpolator) {
        get(gauge_omega).data() =
            0.5 * sqrt(get(gauge_d).data() * conj(get(gauge_d).data()) -
                       get(gauge_c).data() * conj(get(gauge_c).data()));
        GaugeAdjustedBoundaryValue<Tags::Dr<Tags::BondiJ>>::apply(
            make_not_null(&surface_dr_j_buffer), boundary_dr_j, boundary_j,
            gauge_c, gauge_d, gauge_omega, interpolator, l_max);
        GaugeAdjustedBoundaryValue<Tags::BondiJ>::apply(
            make_not_null(&surface_j_buffer), boundary_j, gauge_c, gauge_d,
            gauge_omega, interpolator);
        GaugeAdjustedBoundaryValue<Tags::BondiR>::apply(
            make_not_null(&surface_r_buffer), r, gauge_omega, interpolator);
      };

  detail::iteratively_adapt_angular_coordinates(
      cartesian_cauchy_coordinates, angular_cauchy_coordinates, l_max,
      angular_coordinate_tolerance, max_iterations, 1.0e-2, iteration_function,
      require_convergence, finalize_function);
  Spectral::Swsh::create_angular_and_cartesian_coordinates(
      cartesian_inertial_coordinates, angular_inertial_coordinates, l_max);

  const DataVector one_minus_y_collocation =
      1.0 - Spectral::collocation_points<Spectral::Basis::Legendre,
                                         Spectral::Quadrature::GaussLobatto>(
                number_of_radial_points);
  one_minus_y_coefficient =
      0.25 * (3.0 * get(surface_j_buffer) +
              get(surface_r_buffer) * get(surface_dr_j_buffer));
  one_minus_y_cubed_coefficient =
      -0.0625 * (get(surface_j_buffer) +
                 get(surface_r_buffer) * get(surface_dr_j_buffer));
  for (size_t i = 0; i < number_of_radial_points; i++) {
    ComplexDataVector angular_view_j{
        get(*j).data().data() + get(boundary_j).size() * i,
        get(boundary_j).size()};
    angular_view_j =
        one_minus_y_collocation[i] * one_minus_y_coefficient.data() +
        pow<3>(one_minus_y_collocation[i]) *
            one_minus_y_cubed_coefficient.data();
  }
}

void InverseCubic<true>::operator()(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_cauchy_coordinates,
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_inertial_coordinates,
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_inertial_coordinates,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_dr_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& r,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& /*beta*/,
    const size_t l_max, const size_t number_of_radial_points,
    const gsl::not_null<Parallel::NodeLock*> /*hdf5_lock*/) const {
  const DataVector one_minus_y_collocation =
      1.0 - Spectral::collocation_points<Spectral::Basis::Legendre,
                                         Spectral::Quadrature::GaussLobatto>(
                number_of_radial_points);
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  Spectral::Swsh::SpinWeightedSphericalHarmonic y_22{2, 2_st, 0};
  const auto& collocation_metadata =
      Spectral::Swsh::cached_collocation_metadata<
          Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
  SpinWeighted<ComplexDataVector, 2> perturbed_j{number_of_angular_points};
  for (const auto collocation_point : collocation_metadata) {
    const std::complex<double> y_22_factor =
        y_22.evaluate(collocation_point.theta, collocation_point.phi);
    perturbed_j.data()[collocation_point.offset] = y_22_factor;
  }

  for (size_t i = 0; i < number_of_radial_points; i++) {
    ComplexDataVector angular_view_j{
        get(*j).data().data() + get(boundary_j).size() * i,
        get(boundary_j).size()};
    // auto is acceptable here as these two values are only used once in the
    // below computation. `auto` causes an expression template to be
    // generated, rather than allocating.
    const auto one_minus_y_coefficient =
        0.25 * (3.0 * get(boundary_j).data() +
                get(r).data() * get(boundary_dr_j).data());
    const auto one_minus_y_cubed_coefficient =
        -0.0625 *
        (get(boundary_j).data() + get(r).data() * get(boundary_dr_j).data());
    angular_view_j =
        one_minus_y_collocation[i] * one_minus_y_coefficient +
        pow<3>(one_minus_y_collocation[i]) * one_minus_y_cubed_coefficient;

    double ycenter = -0.;
    double ymin = -0.9;
    double ymax = 0.9;
    double width = 0.15;
    if (one_minus_y_collocation[i] >= (1. - ymax) &&
        one_minus_y_collocation[i] <= (1. - ymin)) {
      angular_view_j +=
          perturbed_j.data() * 0 *
          exp(-pow(1.0 - one_minus_y_collocation[i] - ycenter, 2.0) / width /
              width);
    }
  }
  Spectral::Swsh::create_angular_and_cartesian_coordinates(
      cartesian_cauchy_coordinates, angular_cauchy_coordinates, l_max);
  // Same as the Cauchy coordinates
  Spectral::Swsh::create_angular_and_cartesian_coordinates(
      cartesian_inertial_coordinates, angular_inertial_coordinates, l_max);
}

void InverseCubic<false>::operator()(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_cauchy_coordinates,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_dr_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& r,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& /*beta*/,
    const size_t l_max, const size_t number_of_radial_points,
    const gsl::not_null<Parallel::NodeLock*> /*hdf5_lock*/) const {
  const DataVector one_minus_y_collocation =
      1.0 - Spectral::collocation_points<Spectral::Basis::Legendre,
                                         Spectral::Quadrature::GaussLobatto>(
                number_of_radial_points);
  for (size_t i = 0; i < number_of_radial_points; i++) {
    ComplexDataVector angular_view_j{
        get(*j).data().data() + get(boundary_j).size() * i,
        get(boundary_j).size()};
    // auto is acceptable here as these two values are only used once in the
    // below computation. `auto` causes an expression template to be
    // generated, rather than allocating.
    const auto one_minus_y_coefficient =
        0.25 * (3.0 * get(boundary_j).data() +
                get(r).data() * get(boundary_dr_j).data());
    const auto one_minus_y_cubed_coefficient =
        -0.0625 *
        (get(boundary_j).data() + get(r).data() * get(boundary_dr_j).data());
    angular_view_j =
        one_minus_y_collocation[i] * one_minus_y_coefficient +
        pow<3>(one_minus_y_collocation[i]) * one_minus_y_cubed_coefficient;
  }
  Spectral::Swsh::create_angular_and_cartesian_coordinates(
      cartesian_cauchy_coordinates, angular_cauchy_coordinates, l_max);
}

void InverseCubic<true>::pup(PUP::er& /*p*/) {}
void InverseCubic<false>::pup(PUP::er& /*p*/) {}

PUP::able::PUP_ID InverseCubic<true>::my_PUP_ID = 0;
PUP::able::PUP_ID InverseCubic<false>::my_PUP_ID = 0;
}  // namespace Cce::InitializeJ
