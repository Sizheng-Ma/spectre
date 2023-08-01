// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/Initialize/InitializeSTPsi.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshInterpolation.hpp"

namespace Cce::ScalarTensor {
void InitializeSTPsi::apply(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> bondi_st_psi,
    const size_t l_max, const size_t number_of_radial_points) {
  const DataVector one_minus_y_collocation =
      1.0 - Spectral::collocation_points<Spectral::Basis::Legendre,
                                         Spectral::Quadrature::GaussLobatto>(
                number_of_radial_points);

  const size_t boundary_size =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  Spectral::Swsh::SpinWeightedSphericalHarmonic y_22{0, 2_st, 0};
  const auto& collocation_metadata =
      Spectral::Swsh::cached_collocation_metadata<
          Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
  SpinWeighted<ComplexDataVector, 2> perturbed_j{boundary_size};
  for (const auto collocation_point : collocation_metadata) {
    const std::complex<double> y_22_factor =
        y_22.evaluate(collocation_point.theta, collocation_point.phi);
    perturbed_j.data()[collocation_point.offset] = y_22_factor;
  }

  for (size_t i = 0; i < number_of_radial_points; i++) {
    ComplexDataVector angular_view_scalar_tensor_psi{
        get(*bondi_st_psi).data().data() + boundary_size * i, boundary_size};

    double ycenter = -0.0;
    double ymin = -0.8;
    double ymax = 0.8;
    double width = 0.15;
    angular_view_scalar_tensor_psi =
        perturbed_j.data() * one_minus_y_collocation[i] * 0.01;
    if (one_minus_y_collocation[i] >= (1. - ymax) &&
        one_minus_y_collocation[i] <= (1. - ymin)) {
      angular_view_scalar_tensor_psi +=
          perturbed_j.data() * 0.001 *
          exp(-pow(1.0 - one_minus_y_collocation[i] - ycenter, 2.0) / width /
              width) *
          (one_minus_y_collocation[i] - 1.0 + ymax) *
          (1. - one_minus_y_collocation[i] - ymin) * 4.0 /
          pow((ymax - ymin), 2.0);
    }
  }
}
}  // namespace Cce::ScalarTensor
