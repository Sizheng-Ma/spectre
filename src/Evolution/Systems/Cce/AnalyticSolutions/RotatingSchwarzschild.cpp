// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/AnalyticSolutions/RotatingSchwarzschild.hpp"

#include <complex>
#include <vector>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshInterpolation.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce::detail4 {
double a0(double u) { return sin(u); }
double a0dot(double u) { return cos(u); }

double a2(double u) { return -0.5 * cos(u); }
double a2dot(double u) { return 0.5 * sin(u); }

double a3(double u) { return 0.5 * sin(u); }
double a3dot(double u) { return 0.5 * cos(u); }

double a4(double u) { return 0.75 * cos(u) - 9. / 8. * sin(u); }
double a4dot(double u) { return -0.75 * sin(u) - 9. / 8. * cos(u); }

double a5(double u) { return -77. / 20 * cos(u) - 1.5 * sin(u); }
double a5dot(double u) { return 77. / 20 * sin(u) - 1.5 * cos(u); }

double a6(double u) { return 15. / 16 * cos(u) + 51. / 4. * sin(u); }
double a6dot(double u) { return 15. / 16 * cos(u) + 51. / 4. * sin(u); }

double a7(double u) { return (1287. * cos(u)) / 28. - (1809. * sin(u)) / 80.; }
double a7dot(double u) {
  return -(1287. * sin(u)) / 28. - (1809. * cos(u)) / 80.;
}

double a8(double u) {
  return -(12579. * cos(u) / 80.) - (19857. * sin(u)) / 128.;
}
double a8dot(double u) {
  return (12579. * sin(u) / 80.) - (19857. * cos(u)) / 128.;
}

double a9(double u) {
  return -(73557. * cos(u) / 160) + (133813. * sin(u)) / 140.;
}
double a9dot(double u) {
  return (73557. * sin(u) / 160) + (133813. * cos(u)) / 140.;
}

double a10(double u) {
  return (49797063. * cos(u)) / 8960. + (1272267. * sin(u)) / 1600.;
}
double a10dot(double u) {
  return -(49797063. * sin(u)) / 8960. + (1272267. * cos(u)) / 1600.;
}

double a11(double u) {
  return -((116136241. * cos(u)) / 24640) - (57286503. * sin(u)) / 1792.;
}
double a11dot(double u) {
  return ((116136241. * sin(u)) / 24640) - (57286503. * cos(u)) / 1792.;
}

void inverse_r_dot(DataVector& deriv, const int n, const DataVector r,
                   const DataVector drdt) {
  deriv = -n / pow(r, n + 1) * drdt;
}

void bc_psi(ComplexDataVector& theta, const double u, const double r) {
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
}

void bc_theta(ComplexDataVector& theta, const double u, const double drdt,
              const double r) {
  auto dudt = 1. - drdt / (1. - 2. / r);
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
}
}  // namespace Cce::detail4

namespace Cce::Solutions {

RotatingSchwarzschild::RotatingSchwarzschild(const double extraction_radius,
                                             const double mass,
                                             const double frequency)
    : SphericalMetricData{extraction_radius},
      frequency_{frequency},
      mass_{mass} {}

std::unique_ptr<WorldtubeData> RotatingSchwarzschild::get_clone() const {
  return std::make_unique<RotatingSchwarzschild>(*this);
}

void RotatingSchwarzschild::spherical_metric(
    const gsl::not_null<
        tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>*>
        spherical_metric,
    const size_t l_max, double /*time*/) const {
  get<0, 1>(*spherical_metric) = 0.0;
  get<0, 2>(*spherical_metric) = 0.0;

  get<1, 1>(*spherical_metric) = 1.0 / (1.0 - 2.0 * mass_ / extraction_radius_);
  get<1, 2>(*spherical_metric) = 0.0;
  get<1, 3>(*spherical_metric) = 0.0;

  get<2, 2>(*spherical_metric) = square(extraction_radius_);
  get<2, 3>(*spherical_metric) = 0.0;

  // note: omit the sin factors for the phi components due to pfaffian
  // Jacobian factors
  get<3, 3>(*spherical_metric) = square(extraction_radius_);

  const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
      Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
  for (const auto collocation_point : collocation) {
    get<0, 0>(*spherical_metric)[collocation_point.offset] =
        -(1.0 - 2.0 * mass_ / extraction_radius_ -
          square(frequency_) * square(extraction_radius_) *
              square(sin(collocation_point.theta)));
    get<0, 3>(*spherical_metric)[collocation_point.offset] =
        square(extraction_radius_) * frequency_ * sin(collocation_point.theta);
  }
}

void RotatingSchwarzschild::dr_spherical_metric(
    const gsl::not_null<
        tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>*>
        dr_spherical_metric,
    const size_t l_max, double /*time*/) const {
  get<0, 1>(*dr_spherical_metric) = 0.0;
  get<0, 2>(*dr_spherical_metric) = 0.0;

  get<1, 1>(*dr_spherical_metric) =
      -2.0 * mass_ / (square(extraction_radius_ - 2.0 * mass_));
  get<1, 2>(*dr_spherical_metric) = 0.0;
  get<1, 3>(*dr_spherical_metric) = 0.0;

  get<2, 2>(*dr_spherical_metric) = 2.0 * extraction_radius_;
  get<2, 3>(*dr_spherical_metric) = 0.0;

  // note: omit the sin factors for the phi components due to pfaffian
  // Jacobian factors
  get<3, 3>(*dr_spherical_metric) = 2.0 * extraction_radius_;

  const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
      Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
  for (const auto collocation_point : collocation) {
    get<0, 0>(*dr_spherical_metric)[collocation_point.offset] =
        -(2.0 * mass_ / square(extraction_radius_) -
          2.0 * square(frequency_) * extraction_radius_ *
              square(sin(collocation_point.theta)));
    get<0, 3>(*dr_spherical_metric)[collocation_point.offset] =
        2.0 * extraction_radius_ * frequency_ * sin(collocation_point.theta);
  }
}

void RotatingSchwarzschild::dt_spherical_metric(
    const gsl::not_null<
        tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>*>
        dt_spherical_metric,
    size_t /*l_max*/, double /*time*/) const {
  for(auto& component : *dt_spherical_metric) {
    component = 0.0;
  }
}

void RotatingSchwarzschild::variables_impl(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> st_psi,
    size_t output_l_max, double time,
    tmpl::type_<Tags::BondiSTPsi> /*meta*/) const {
  auto r = extraction_radius_;
  auto rs = r + 2 * log(r / 2. - 1.);
  detail4::bc_psi(get(*st_psi).data(), time - rs, r);
}

void RotatingSchwarzschild::variables_impl(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> st_psi,
    size_t output_l_max, double time,
    tmpl::type_<Tags::BondiSTTheta> /*meta*/) const {
  auto r = extraction_radius_;
  auto rs = r + 2 * log(r / 2. - 1.);

  detail4::bc_theta(get(*st_psi).data(), time - rs, r * 0, r);
}

void RotatingSchwarzschild::variables_impl(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -2>>*> news,
    size_t /*output_l_max*/, double /*time*/,
    tmpl::type_<Tags::News> /*meta*/) const {
  get(*news).data() = 0.0;
}

void RotatingSchwarzschild::pup(PUP::er& p) {
  SphericalMetricData::pup(p);
  p | mass_;
  p | frequency_;
}

PUP::able::PUP_ID RotatingSchwarzschild::my_PUP_ID = 0;
}  // namespace Cce::Solutions
