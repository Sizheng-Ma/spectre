// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/Initialize/InitializeSTPsi.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshInterpolation.hpp"

namespace Cce {

namespace detail2 {
ComplexDataVector a0(ComplexDataVector u) { return sin(u); }
DataVector a0dot(DataVector u) { return cos(u); }

ComplexDataVector a2(ComplexDataVector u) { return -0.5 * cos(u); }
DataVector a2dot(DataVector u) { return 0.5 * sin(u); }

ComplexDataVector a3(ComplexDataVector u) { return 0.5 * sin(u); }
DataVector a3dot(DataVector u) { return 0.5 * cos(u); }

ComplexDataVector a4(ComplexDataVector u) {
  return 0.75 * cos(u) - 9. / 8. * sin(u);
}
DataVector a4dot(DataVector u) { return -0.75 * sin(u) - 9. / 8. * cos(u); }

ComplexDataVector a5(ComplexDataVector u) {
  return -77. / 20 * cos(u) - 1.5 * sin(u);
}
DataVector a5dot(DataVector u) { return 77. / 20 * sin(u) - 1.5 * cos(u); }

ComplexDataVector a6(ComplexDataVector u) {
  return 15. / 16 * cos(u) + 51. / 4. * sin(u);
}
DataVector a6dot(DataVector u) { return 15. / 16 * cos(u) + 51. / 4. * sin(u); }

ComplexDataVector a7(ComplexDataVector u) {
  return (1287. * cos(u)) / 28. - (1809. * sin(u)) / 80.;
}
DataVector a7dot(DataVector u) {
  return -(1287. * sin(u)) / 28. - (1809. * cos(u)) / 80.;
}

ComplexDataVector a8(ComplexDataVector u) {
  return -(12579. * cos(u) / 80.) - (19857. * sin(u)) / 128.;
}
DataVector a8dot(DataVector u) {
  return (12579. * sin(u) / 80.) - (19857. * cos(u)) / 128.;
}

ComplexDataVector a9(ComplexDataVector u) {
  return -(73557. * cos(u) / 160) + (133813. * sin(u)) / 140.;
}
DataVector a9dot(DataVector u) {
  return (73557. * sin(u) / 160) + (133813. * cos(u)) / 140.;
}

ComplexDataVector a10(ComplexDataVector u) {
  return (49797063. * cos(u)) / 8960. + (1272267. * sin(u)) / 1600.;
}
DataVector a10dot(DataVector u) {
  return -(49797063. * sin(u)) / 8960. + (1272267. * cos(u)) / 1600.;
}

ComplexDataVector a11(ComplexDataVector u) {
  return -((116136241. * cos(u)) / 24640) - (57286503. * sin(u)) / 1792.;
}
DataVector a11dot(DataVector u) {
  return ((116136241. * sin(u)) / 24640) - (57286503. * cos(u)) / 1792.;
}

ComplexDataVector a12(ComplexDataVector u) {
  return -((16472195091. * cos(u)) / 89600) + (419653067. * sin(u)) / 5120.;
}

ComplexDataVector a13(ComplexDataVector u) {
  return ((197057851611. * cos(u)) / 232960) + (517853793843. * sin(u)) / 492800.;
}

ComplexDataVector a14(ComplexDataVector u) {
  return ((23027722022071. * cos(u)) / 3942400) - (2420206444191. * sin(u)) / 313600.;
}

ComplexDataVector a15(ComplexDataVector u) {
  return -((166944468961581. * cos(u)) / 2464000) - (218435295225339. * sin(u)) / 7321600.;
}
void inverse_r_dot(DataVector& deriv, const int n, const DataVector r,
                   const DataVector drdt) {
  deriv = -n / pow(r, n + 1) * drdt;
}

void bc_psi(ComplexDataVector& theta, const ComplexDataVector u,
            const ComplexDataVector r) {
  theta = a0(u) * r;

  ComplexDataVector radial_profile = square(r)*r; 
  theta += a2(u) * radial_profile;
  radial_profile*=r;
  theta += a3(u) * radial_profile;
  radial_profile*=r;
  theta += a4(u) * radial_profile;
  radial_profile*=r;
  theta += a5(u) * radial_profile;
  radial_profile*=r;
  theta += a6(u) * radial_profile;
  radial_profile*=r;
  theta += a7(u) * radial_profile;
  radial_profile*=r;
  theta += a8(u) * radial_profile;
  radial_profile*=r;
  theta += a9(u) * radial_profile;
  radial_profile*=r;
  theta += a10(u) * radial_profile;
  radial_profile*=r;
  theta += a11(u) * radial_profile;
  radial_profile*=r;
  theta += a12(u) * radial_profile;
  radial_profile*=r;
  theta += a13(u) * radial_profile;
  radial_profile*=r;
  theta += a14(u) * radial_profile;
  radial_profile*=r;
  theta += a15(u) * radial_profile;
}
}  // namespace detail2
}  // namespace Cce
namespace Cce::ScalarTensor {
void InitializeSTPsi::apply(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> bondi_st_psi,
    const size_t l_max, const size_t number_of_radial_points,
    const Scalar<SpinWeighted<ComplexDataVector, 0>> st_psi_boundary,
    const Scalar<SpinWeighted<ComplexDataVector, 0>> bondi_r) {
  const DataVector one_minus_y_collocation =
      1.0 - Spectral::collocation_points<Spectral::Basis::Legendre,
                                         Spectral::Quadrature::GaussLobatto>(
                number_of_radial_points);

  const size_t boundary_size =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  Spectral::Swsh::SpinWeightedSphericalHarmonic y_22{0, 2, 0};
  const auto& collocation_metadata =
      Spectral::Swsh::cached_collocation_metadata<
          Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
  SpinWeighted<ComplexDataVector, 0> perturbed_j{boundary_size};
  for (const auto collocation_point : collocation_metadata) {
    const std::complex<double> y_22_factor =
        y_22.evaluate(collocation_point.theta, collocation_point.phi);
    perturbed_j.data()[collocation_point.offset] = y_22_factor;
  }

  for (size_t i = 0; i < number_of_radial_points; i++) {
    ComplexDataVector angular_view_scalar_tensor_psi{
        get(*bondi_st_psi).data().data() + boundary_size * i, boundary_size};

    // double ycenter = -0.0;
    // double ymin = -0.8;
    // double ymax = 0.8;
    // double width = 0.15;
    const double u0 = 10;
    const double sigma0 = 1;
    double psi_boundary = exp(-0.5 * square(u0) / square(sigma0));
    // angular_view_scalar_tensor_psi =
    //     get(st_psi_boundary).data() * one_minus_y_collocation[i] / 2.;

    auto u0new = -get(bondi_r).data() - 4 * log(get(bondi_r).data() / 2 - 1.);

    detail2::bc_psi(angular_view_scalar_tensor_psi, u0new,
                    one_minus_y_collocation[i] / 2. / get(bondi_r).data());

    // if (one_minus_y_collocation[i] >= (1. - ymax) &&
    //     one_minus_y_collocation[i] <= (1. - ymin)) {
    //   angular_view_scalar_tensor_psi +=
    //       perturbed_j.data() * 0.001 *
    //       exp(-pow(1.0 - one_minus_y_collocation[i] - ycenter, 2.0) / width /
    //           width) *
    //       (one_minus_y_collocation[i] - 1.0 + ymax) *
    //       (1. - one_minus_y_collocation[i] - ymin) * 4.0 /
    //       pow((ymax - ymin), 2.0);
    // }
  }
}
}  // namespace Cce::ScalarTensor
