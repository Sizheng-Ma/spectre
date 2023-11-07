// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/AnalyticSolutions/BouncingBlackHole.hpp"

#include <cstddef>
#include <memory>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/BouncingBlackHole.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Cce {

namespace detail1 {
DataVector a0(DataVector u) { return sin(u); }
DataVector a0dot(DataVector u) { return cos(u); }

DataVector a2(DataVector u) { return -0.5 * cos(u); }
DataVector a2dot(DataVector u) { return 0.5 * sin(u); }

DataVector a3(DataVector u) { return 0.5 * sin(u); }
DataVector a3dot(DataVector u) { return 0.5 * cos(u); }

DataVector a4(DataVector u) { return 0.75 * cos(u) - 9. / 8. * sin(u); }
DataVector a4dot(DataVector u) { return -0.75 * sin(u) - 9. / 8. * cos(u); }

DataVector a5(DataVector u) { return -77. / 20 * cos(u) - 1.5 * sin(u); }
DataVector a5dot(DataVector u) { return 77. / 20 * sin(u) - 1.5 * cos(u); }

DataVector a6(DataVector u) { return 15. / 16 * cos(u) + 51. / 4. * sin(u); }
DataVector a6dot(DataVector u) { return -15. / 16 * sin(u) + 51. / 4. * cos(u); }

DataVector a7(DataVector u) {
  return (1287. * cos(u)) / 28. - (1809. * sin(u)) / 80.;
}
DataVector a7dot(DataVector u) {
  return -(1287. * sin(u)) / 28. - (1809. * cos(u)) / 80.;
}

DataVector a8(DataVector u) {
  return -(12579. * cos(u) / 80.) - (19857. * sin(u)) / 128.;
}
DataVector a8dot(DataVector u) {
  return (12579. * sin(u) / 80.) - (19857. * cos(u)) / 128.;
}

DataVector a9(DataVector u) {
  return -(73557. * cos(u) / 160) + (133813. * sin(u)) / 140.;
}
DataVector a9dot(DataVector u) {
  return (73557. * sin(u) / 160) + (133813. * cos(u)) / 140.;
}

DataVector a10(DataVector u) {
  return (49797063. * cos(u)) / 8960. + (1272267. * sin(u)) / 1600.;
}
DataVector a10dot(DataVector u) {
  return -(49797063. * sin(u)) / 8960. + (1272267. * cos(u)) / 1600.;
}

DataVector a11(DataVector u) {
  return -((116136241. * cos(u)) / 24640) - (57286503. * sin(u)) / 1792.;
}
DataVector a11dot(DataVector u) {
  return ((116136241. * sin(u)) / 24640) - (57286503. * cos(u)) / 1792.;
}

DataVector a12(DataVector u) {
  return -((16472195091. * cos(u)) / 89600) + (419653067. * sin(u)) / 5120.;
}
DataVector a12dot(DataVector u) {
  return ((16472195091. * sin(u)) / 89600) + (419653067. * cos(u)) / 5120.;
}

DataVector a13(DataVector u) {
  return ((197057851611. * cos(u)) / 232960) + (517853793843. * sin(u)) / 492800.;
}
DataVector a13dot(DataVector u) {
  return -((197057851611. * sin(u)) / 232960) + (517853793843. * cos(u)) / 492800.;
}

DataVector a14(DataVector u) {
  return ((23027722022071. * cos(u)) / 3942400) - (2420206444191. * sin(u)) / 313600.;
}
DataVector a14dot(DataVector u) {
  return -((23027722022071. * sin(u)) / 3942400) - (2420206444191. * cos(u)) / 313600.;
}

DataVector a15(DataVector u) {
  return -((166944468961581. * cos(u)) / 2464000) - (218435295225339. * sin(u)) / 7321600.;
}
DataVector a15dot(DataVector u) {
  return ((166944468961581. * sin(u)) / 2464000) - (218435295225339. * cos(u)) / 7321600.;
}

void inverse_r_dot(DataVector& deriv, const int n, const DataVector r,
                   const DataVector drdt) {
  deriv = -n / pow(r, n + 1) * drdt;
}

void bc_psi(ComplexDataVector& theta, const DataVector u, const DataVector r) {
  theta = a0(u) / r;

  theta += a2(u) / pow(r, 3);
  theta += a3(u) / pow(r, 4);
  theta += a4(u) / pow(r, 5);
  theta += a5(u) / pow(r, 6);
  theta += a6(u) / pow(r, 7);
  theta += a7(u) / pow(r, 8);
  theta += a8(u) / pow(r, 9);
  theta += a9(u) / pow(r, 10);
  theta += a10(u) / pow(r, 11);
  theta += a11(u) / pow(r, 12);
  theta += a12(u) / pow(r, 13);
  theta += a13(u) / pow(r, 14);
  theta += a14(u) / pow(r, 15);
  theta += a15(u) / pow(r, 16);
}

void bc_theta(ComplexDataVector& theta, const DataVector u,
              const DataVector drdt, const DataVector r) {
  auto dudt = 1. + drdt - 2 * drdt / (1 - 2. / r);
  theta = a0dot(u) * dudt / r - (0 + 1) / square(r) * a0(u) * drdt;

  int iii = 2;
  theta += a2dot(u) * dudt / pow(r, iii + 1) -
           (iii + 1) / pow(r, iii + 2) * a2(u) * drdt;

  iii = 3;
  theta += a3dot(u) * dudt / pow(r, iii + 1) -
           (iii + 1) / pow(r, iii + 2) * a3(u) * drdt;

  iii = 4;
  theta += a4dot(u) * dudt / pow(r, iii + 1) -
           (iii + 1) / pow(r, iii + 2) * a4(u) * drdt;

  iii = 5;
  theta += a5dot(u) * dudt / pow(r, iii + 1) -
           (iii + 1) / pow(r, iii + 2) * a5(u) * drdt;

  iii = 6;
  theta += a6dot(u) * dudt / pow(r, iii + 1) -
           (iii + 1) / pow(r, iii + 2) * a6(u) * drdt;

  iii = 7;
  theta += a7dot(u) * dudt / pow(r, iii + 1) -
           (iii + 1) / pow(r, iii + 2) * a7(u) * drdt;

  iii = 8;
  theta += a8dot(u) * dudt / pow(r, iii + 1) -
           (iii + 1) / pow(r, iii + 2) * a8(u) * drdt;

  iii = 9;
  theta += a9dot(u) * dudt / pow(r, iii + 1) -
           (iii + 1) / pow(r, iii + 2) * a9(u) * drdt;

  iii = 10;
  theta += a10dot(u) * dudt / pow(r, iii + 1) -
           (iii + 1) / pow(r, iii + 2) * a10(u) * drdt;

  iii = 11;
  theta += a11dot(u) * dudt / pow(r, iii + 1) -
           (iii + 1) / pow(r, iii + 2) * a11(u) * drdt;

  iii = 12;
  theta += a12dot(u) * dudt / pow(r, iii + 1) -
           (iii + 1) / pow(r, iii + 2) * a12(u) * drdt;

  iii = 13;
  theta += a13dot(u) * dudt / pow(r, iii + 1) -
           (iii + 1) / pow(r, iii + 2) * a13(u) * drdt;

  iii = 14;
  theta += a14dot(u) * dudt / pow(r, iii + 1) -
           (iii + 1) / pow(r, iii + 2) * a14(u) * drdt;

  iii = 15;
  theta += a15dot(u) * dudt / pow(r, iii + 1) -
           (iii + 1) / pow(r, iii + 2) * a15(u) * drdt;
}
}  // namespace detail1
}  // namespace Cce

namespace Cce::Solutions {

BouncingBlackHole::BouncingBlackHole(const double amplitude,
                                     const double extraction_radius,
                                     const double mass, const double period)
    : WorldtubeData(extraction_radius),
      amplitude_{amplitude},
      mass_{mass},
      frequency_{2.0 * M_PI / period} {}

std::unique_ptr<WorldtubeData> BouncingBlackHole::get_clone() const {
  return std::make_unique<BouncingBlackHole>(*this);
}

void BouncingBlackHole::variables_impl(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> st_psi,
    size_t output_l_max, double time,
    tmpl::type_<Tags::BondiSTPsi> /*meta*/) const {
  const auto& cartesian_coordinates =
      cache_or_compute<Tags::CauchyCartesianCoords>(output_l_max, time);
  const DataVector adjusted_x_coordinate =
      amplitude_ * pow<4>(sin(frequency_ * time)) +
      get<0>(cartesian_coordinates);
  const DataVector r = sqrt(square(adjusted_x_coordinate) +
                            square(get<1>(cartesian_coordinates)) +
                            square(get<2>(cartesian_coordinates)));

  auto rs = r + 2 * mass_ * log(r / 2. - 1.);

  detail1::bc_psi(get(*st_psi).data(), time + r - 2 * rs, r);
  //    = sin(time - r) / r;
}
void BouncingBlackHole::variables_impl(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> st_psi,
    size_t output_l_max, double time,
    tmpl::type_<Tags::BondiSTTheta> /*meta*/) const {
  const auto& cartesian_coordinates =
      cache_or_compute<Tags::CauchyCartesianCoords>(output_l_max, time);
  const double dt_adjusted_x_coordinate = 4.0 * amplitude_ * frequency_ *
                                          cos(frequency_ * time) *
                                          pow<3>(sin(frequency_ * time));
  const DataVector adjusted_x_coordinate =
      amplitude_ * pow<4>(sin(frequency_ * time)) +
      get<0>(cartesian_coordinates);
  const DataVector r = sqrt(square(adjusted_x_coordinate) +
                            square(get<1>(cartesian_coordinates)) +
                            square(get<2>(cartesian_coordinates)));

  auto drdt = adjusted_x_coordinate / r * dt_adjusted_x_coordinate;

  auto rs = r + 2 * mass_ * log(r / 2. - 1.);

  detail1::bc_theta(get(*st_psi).data(), time + r - 2 * rs, drdt, r);

  //   get(*st_psi).data() =
  //       cos(time - r) / r * (1 - drdt) - sin(time - r) / square(r) * drdt;
}

void BouncingBlackHole::variables_impl(
    const gsl::not_null<tnsr::aa<DataVector, 3>*> spacetime_metric,
    const size_t l_max, const double time,
    tmpl::type_<gr::Tags::SpacetimeMetric<DataVector, 3>> /*meta*/) const {
  const auto& cartesian_coordinates =
      cache_or_compute<Tags::CauchyCartesianCoords>(l_max, time);

  const double dt_adjusted_x_coordinate = 4.0 * amplitude_ * frequency_ *
                                          cos(frequency_ * time) *
                                          pow<3>(sin(frequency_ * time));

  const DataVector adjusted_x_coordinate =
      amplitude_ * pow<4>(sin(frequency_ * time)) +
      get<0>(cartesian_coordinates);

  const DataVector r = sqrt(square(adjusted_x_coordinate) +
                            square(get<1>(cartesian_coordinates)) +
                            square(get<2>(cartesian_coordinates)));

  const DataVector inverse_r_cubed = 1.0 / pow<3>(r);

  get<0, 0>(*spacetime_metric) =
      -1.0 + 2.0 * mass_ / r +
      square(dt_adjusted_x_coordinate) *
          (1.0 +
           2.0 * mass_ * square(adjusted_x_coordinate) * inverse_r_cubed) +
      4.0 * mass_ * dt_adjusted_x_coordinate * adjusted_x_coordinate /
          square(r);

  get<0, 1>(*spacetime_metric) =
      dt_adjusted_x_coordinate +
      2.0 * mass_ * adjusted_x_coordinate / square(r) +
      2.0 * mass_ * dt_adjusted_x_coordinate * square(adjusted_x_coordinate) *
          inverse_r_cubed;

  for (size_t i = 1; i < 3; ++i) {
    spacetime_metric->get(0, i + 1) =
        2.0 * mass_ * (dt_adjusted_x_coordinate * adjusted_x_coordinate + r) *
        cartesian_coordinates.get(i) * inverse_r_cubed;

    spacetime_metric->get(i + 1, i + 1) =
        1.0 +
        2.0 * mass_ * square(cartesian_coordinates.get(i)) * inverse_r_cubed;

    spacetime_metric->get(1, i + 1) = 2.0 * mass_ * adjusted_x_coordinate *
                                      cartesian_coordinates.get(i) *
                                      inverse_r_cubed;
  }

  get<1, 1>(*spacetime_metric) =
      1.0 + 2.0 * mass_ * square(adjusted_x_coordinate) * inverse_r_cubed;

  get<2, 3>(*spacetime_metric) = 2.0 * mass_ * get<1>(cartesian_coordinates) *
                                 get<2>(cartesian_coordinates) *
                                 inverse_r_cubed;
}

void BouncingBlackHole::variables_impl(
    const gsl::not_null<tnsr::aa<DataVector, 3>*> dt_spacetime_metric,
    const size_t l_max, const double time,
    tmpl::type_<::Tags::dt<gr::Tags::SpacetimeMetric<DataVector, 3>>> /*meta*/)
    const {
  const auto& cartesian_coordinates =
      cache_or_compute<Tags::CauchyCartesianCoords>(l_max, time);

  const double dt_adjusted_x_coordinate = 4.0 * amplitude_ * frequency_ *
                                          cos(frequency_ * time) *
                                          pow<3>(sin(frequency_ * time));
  const double dt_dt_adjusted_x_coordinate =
      4.0 * amplitude_ * square(frequency_) *
      (3.0 * square(cos(frequency_ * time)) * square(sin(frequency_ * time)) -
       pow<4>(sin(frequency_ * time)));

  const DataVector adjusted_x_coordinate =
      amplitude_ * pow<4>(sin(frequency_ * time)) +
      get<0>(cartesian_coordinates);
  const DataVector r = sqrt(square(adjusted_x_coordinate) +
                            square(get<1>(cartesian_coordinates)) +
                            square(get<2>(cartesian_coordinates)));
  const DataVector inverse_r_cubed = 1.0 / pow<3>(r);

  get<0, 0>(*dt_spacetime_metric) =
      2.0 * dt_dt_adjusted_x_coordinate * dt_adjusted_x_coordinate *
          (2.0 * mass_ * square(adjusted_x_coordinate) * inverse_r_cubed +
           1.0) +
      2.0 * dt_dt_adjusted_x_coordinate * 2.0 * mass_ * adjusted_x_coordinate /
          square(r) +
      pow<3>(dt_adjusted_x_coordinate) *
          (4.0 * mass_ * adjusted_x_coordinate * inverse_r_cubed -
           6.0 * mass_ * pow<3>(adjusted_x_coordinate) / pow<5>(r)) +
      2.0 * square(dt_adjusted_x_coordinate) *
          (2.0 * mass_ / square(r) -
           4.0 * mass_ * square(adjusted_x_coordinate) / pow<4>(r)) -
      dt_adjusted_x_coordinate * 2.0 * mass_ * adjusted_x_coordinate *
          inverse_r_cubed;

  get<0, 1>(*dt_spacetime_metric) =
      dt_dt_adjusted_x_coordinate +
      2.0 * mass_ / square(r) *
          (dt_adjusted_x_coordinate -
           2.0 * square(adjusted_x_coordinate) * dt_adjusted_x_coordinate /
               square(r) +
           square(adjusted_x_coordinate) * dt_dt_adjusted_x_coordinate / r +
           2.0 * adjusted_x_coordinate * square(dt_adjusted_x_coordinate) / r -
           3.0 * pow<3>(adjusted_x_coordinate) *
               square(dt_adjusted_x_coordinate) * inverse_r_cubed);

  for (size_t i = 1; i < 3; ++i) {
    dt_spacetime_metric->get(0, i + 1) =
        2.0 * mass_ * cartesian_coordinates.get(i) * inverse_r_cubed *
        (dt_dt_adjusted_x_coordinate * adjusted_x_coordinate +
         square(dt_adjusted_x_coordinate) -
         2.0 * adjusted_x_coordinate * dt_adjusted_x_coordinate / r -
         3.0 * square(adjusted_x_coordinate) *
             square(dt_adjusted_x_coordinate) / square(r));

    dt_spacetime_metric->get(1, i + 1) =
        2.0 * mass_ * dt_adjusted_x_coordinate * cartesian_coordinates.get(i) *
        inverse_r_cubed *
        (1.0 - 3.0 * square(adjusted_x_coordinate) / square(r));

    dt_spacetime_metric->get(i + 1, i + 1) =
        -6.0 * mass_ * adjusted_x_coordinate *
        square(cartesian_coordinates.get(i)) * dt_adjusted_x_coordinate /
        pow<5>(r);
  }
  get<1, 1>(*dt_spacetime_metric) =
      2.0 * mass_ * adjusted_x_coordinate * dt_adjusted_x_coordinate *
      inverse_r_cubed * (2.0 - 3.0 * square(adjusted_x_coordinate) / square(r));

  get<2, 3>(*dt_spacetime_metric) =
      -6.0 * mass_ * adjusted_x_coordinate * get<1>(cartesian_coordinates) *
      get<2>(cartesian_coordinates) * dt_adjusted_x_coordinate / pow<5>(r);
}

void BouncingBlackHole::variables_impl(
    const gsl::not_null<tnsr::iaa<DataVector, 3>*> d_spacetime_metric,
    const size_t l_max, const double time,
    tmpl::type_<gh::Tags::Phi<DataVector, 3>> /*meta*/) const {
  const auto& cartesian_coordinates =
      cache_or_compute<Tags::CauchyCartesianCoords>(l_max, time);

  const double dt_adjusted_x_coordinate = 4.0 * amplitude_ * frequency_ *
                                          cos(frequency_ * time) *
                                          pow<3>(sin(frequency_ * time));

  const DataVector adjusted_x_coordinate =
      amplitude_ * pow<4>(sin(frequency_ * time)) +
      get<0>(cartesian_coordinates);
  const DataVector r = sqrt(square(adjusted_x_coordinate) +
                            square(get<1>(cartesian_coordinates)) +
                            square(get<2>(cartesian_coordinates)));
  const DataVector inverse_r_cubed = 1.0 / pow<3>(r);

  get<0, 0, 0>(*d_spacetime_metric) =
      2.0 * mass_ / square(r) *
      (2.0 * dt_adjusted_x_coordinate +
       (2.0 * square(dt_adjusted_x_coordinate) * adjusted_x_coordinate -
        adjusted_x_coordinate) /
           r -
       4.0 * dt_adjusted_x_coordinate * square(adjusted_x_coordinate) /
           square(r) -
       3.0 * square(dt_adjusted_x_coordinate * adjusted_x_coordinate) *
           adjusted_x_coordinate * inverse_r_cubed);

  get<0, 0, 1>(*d_spacetime_metric) =
      2.0 * mass_ / square(r) *
      (1.0 + 2.0 * dt_adjusted_x_coordinate * adjusted_x_coordinate / r -
       2.0 * square(adjusted_x_coordinate) / square(r) -
       3.0 * dt_adjusted_x_coordinate * pow<3>(adjusted_x_coordinate) *
           inverse_r_cubed);

  get<0, 1, 1>(*d_spacetime_metric) =
      2.0 * mass_ *
      (2.0 * adjusted_x_coordinate -
       3.0 * pow<3>(adjusted_x_coordinate) / square(r)) *
      inverse_r_cubed;

  for (size_t i = 1; i < 3; ++i) {
    d_spacetime_metric->get(i, 0, 0) =
        -2.0 * mass_ * cartesian_coordinates.get(i) * inverse_r_cubed *
        (1.0 + 4.0 * dt_adjusted_x_coordinate * adjusted_x_coordinate / r +
         3.0 * square(dt_adjusted_x_coordinate * adjusted_x_coordinate) /
             square(r));

    d_spacetime_metric->get(i, 0, 1) =
        -2.0 * mass_ * adjusted_x_coordinate * cartesian_coordinates.get(i) /
        pow<4>(r) *
        (2.0 + 3.0 * dt_adjusted_x_coordinate * adjusted_x_coordinate / r);

    d_spacetime_metric->get(i, 1, 1) = -6.0 * mass_ *
                                       square(adjusted_x_coordinate) *
                                       cartesian_coordinates.get(i) / pow<5>(r);

    d_spacetime_metric->get(0, 1, i + 1) =
        2.0 * mass_ * cartesian_coordinates.get(i) *
        (1.0 - 3.0 * square(adjusted_x_coordinate) / square(r)) *
        inverse_r_cubed;

    d_spacetime_metric->get(0, 0, i + 1) =
        2.0 * mass_ * cartesian_coordinates.get(i) * inverse_r_cubed *
        (dt_adjusted_x_coordinate - 2.0 * adjusted_x_coordinate / r -
         3.0 * dt_adjusted_x_coordinate * square(adjusted_x_coordinate) /
             square(r));

    for (size_t j = 1; j < 3; ++j) {
      if (i == j) {
        d_spacetime_metric->get(i, 0, j + 1) =
            2.0 * mass_ / square(r) *
            (1.0 + dt_adjusted_x_coordinate * adjusted_x_coordinate / r -
             2.0 * square(cartesian_coordinates.get(i)) / square(r) -
             3.0 * dt_adjusted_x_coordinate * adjusted_x_coordinate *
                 square(cartesian_coordinates.get(i)) * inverse_r_cubed);

        d_spacetime_metric->get(i, 1, j + 1) =
            2.0 * mass_ * adjusted_x_coordinate *
            (1.0 - 3.0 * square(cartesian_coordinates.get(i)) / square(r)) *
            inverse_r_cubed;

        d_spacetime_metric->get(0, i + 1, j + 1) =
            -6.0 * mass_ * square(cartesian_coordinates.get(i)) *
            adjusted_x_coordinate / pow<5>(r);
      } else {
        d_spacetime_metric->get(i, 0, j + 1) =
            -2.0 * mass_ * cartesian_coordinates.get(i) *
            cartesian_coordinates.get(j) / pow<4>(r) *
            (3.0 * dt_adjusted_x_coordinate * adjusted_x_coordinate / r + 2.0);

        d_spacetime_metric->get(i, 1, j + 1) =
            -6.0 * mass_ * adjusted_x_coordinate *
            cartesian_coordinates.get(i) * cartesian_coordinates.get(j) /
            pow<5>(r);

        d_spacetime_metric->get(0, i + 1, j + 1) =
            -6.0 * mass_ * cartesian_coordinates.get(i) *
            cartesian_coordinates.get(j) * adjusted_x_coordinate / pow<5>(r);
      }
      for (size_t k = 1; k < 3; ++k) {
        if (i == j and j == k) {
          d_spacetime_metric->get(k, i + 1, j + 1) =
              2.0 * mass_ *
              (2.0 * cartesian_coordinates.get(i) -
               3.0 * pow<3>(cartesian_coordinates.get(i)) / square(r)) *
              inverse_r_cubed;

        } else if (i == j) {
          d_spacetime_metric->get(i, j + 1, k + 1) =
              2.0 * mass_ *
              (cartesian_coordinates.get(k) -
               3.0 * square(cartesian_coordinates.get(i)) *
                   cartesian_coordinates.get(k) / square(r)) *
              inverse_r_cubed;
          d_spacetime_metric->get(k, i + 1, j + 1) =
              -6.0 * mass_ * square(cartesian_coordinates.get(i)) *
              cartesian_coordinates.get(k) / pow<5>(r);
        }
      }
    }
  }
}

void BouncingBlackHole::variables_impl(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -2>>*> news,
    const size_t /*output_l_max*/, const double /*time*/,
    tmpl::type_<Tags::News> /*meta*/) const {
  get(*news).data() = 0.0;
}

void BouncingBlackHole::pup(PUP::er& p) {
  WorldtubeData::pup(p);
  p | amplitude_;
  p | mass_;
  p | frequency_;
}

PUP::able::PUP_ID BouncingBlackHole::my_PUP_ID = 0;
}  // namespace Cce::Solutions
