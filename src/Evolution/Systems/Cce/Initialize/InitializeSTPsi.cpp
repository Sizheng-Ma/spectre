// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/Initialize/InitializeSTPsi.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"

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
  for (size_t i = 0; i < number_of_radial_points; i++) {
    ComplexDataVector angular_view_scalar_tensor_psi{
        get(*bondi_st_psi).data().data() + boundary_size * i, boundary_size};

    angular_view_scalar_tensor_psi = one_minus_y_collocation[i];
  }
}
}  // namespace Cce::ScalarTensor
