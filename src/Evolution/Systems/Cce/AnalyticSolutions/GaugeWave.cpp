// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/AnalyticSolutions/GaugeWave.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>
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
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce::detail5 {
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
double a6dot(double u) { return -15. / 16 * sin(u) + 51. / 4. * cos(u); }

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

double a12(double u) {
  return -((16472195091. * cos(u)) / 89600) + (419653067. * sin(u)) / 5120.;
}
double a12dot(double u) {
  return ((16472195091. * sin(u)) / 89600) + (419653067. * cos(u)) / 5120.;
}

double a13(double u) {
  return ((197057851611. * cos(u)) / 232960) +
         (517853793843. * sin(u)) / 492800.;
}
double a13dot(double u) {
  return -((197057851611. * sin(u)) / 232960) +
         (517853793843. * cos(u)) / 492800.;
}

double a14(double u) {
  return ((23027722022071. * cos(u)) / 3942400) -
         (2420206444191. * sin(u)) / 313600.;
}
double a14dot(double u) {
  return -((23027722022071. * sin(u)) / 3942400) -
         (2420206444191. * cos(u)) / 313600.;
}

double a15(double u) {
  return -((166944468961581. * cos(u)) / 2464000) -
         (218435295225339. * sin(u)) / 7321600.;
}
double a15dot(double u) {
  return ((166944468961581. * sin(u)) / 2464000) -
         (218435295225339. * cos(u)) / 7321600.;
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
  theta += a12(u) / pow(r, 13);
  theta += a13(u) / pow(r, 14);
  theta += a14(u) / pow(r, 15);
  theta += a15(u) / pow(r, 16);
}

void bc_theta(ComplexDataVector& theta, const double u, const double r,
              const double dudt) {
  theta = a0dot(u) / r * dudt;

  double drdt = 0;

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
}  // namespace Cce::detail5

namespace Cce::Solutions {

GaugeWave::GaugeWave(const double extraction_radius, const double mass,
                     const double frequency, const double amplitude,
                     const double peak_time, const double duration)
    : SphericalMetricData{extraction_radius},
      mass_{mass},
      frequency_{frequency},
      amplitude_{amplitude},
      peak_time_{peak_time},
      duration_{duration} {}

std::unique_ptr<WorldtubeData> GaugeWave::get_clone() const {
  return std::make_unique<GaugeWave>(*this);
}

double GaugeWave::coordinate_wave_function(const double time) const {
  const auto retarded_time = time - extraction_radius_;
  return amplitude_ * sin(frequency_ * retarded_time) *
         exp(-square(retarded_time - peak_time_) / square(duration_));
}

double GaugeWave::du_coordinate_wave_function(const double time) const {
  const auto retarded_time = time - extraction_radius_;
  return amplitude_ *
         (-2.0 * (retarded_time - peak_time_) / square(duration_) *
              sin(frequency_ * retarded_time) +
          frequency_ * cos(frequency_ * retarded_time)) *
         exp(-square(retarded_time - peak_time_) / square(duration_));
}

double GaugeWave::du_du_coordinate_wave_function(const double time) const {
  const auto retarded_time = time - extraction_radius_;
  return amplitude_ *
         (-4.0 * square(duration_) * (retarded_time - peak_time_) * frequency_ *
              cos(frequency_ * retarded_time) +
          (-2.0 * square(duration_) + 4.0 * square(retarded_time - peak_time_) -
           pow<4>(duration_) * square(frequency_)) *
              sin(frequency_ * retarded_time)) /
         pow<4>(duration_) *
         exp(-square(retarded_time - peak_time_) / square(duration_));
}

void GaugeWave::spherical_metric(
    const gsl::not_null<
        tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>*>
        spherical_metric,
    const size_t /*l_max*/, const double time) const {
  const auto wave_f = coordinate_wave_function(time);
  const auto du_wave_f = du_coordinate_wave_function(time);
  get<0, 0>(*spherical_metric) = -(extraction_radius_ - 2.0 * mass_) *
                                 square(extraction_radius_ + du_wave_f) /
                                 pow<3>(extraction_radius_);
  get<0, 1>(*spherical_metric) =
      (extraction_radius_ + du_wave_f) *
      (2.0 * mass_ * square(extraction_radius_) +
       (extraction_radius_ - 2.0 * mass_) *
           (extraction_radius_ * du_wave_f + wave_f)) /
      pow<4>(extraction_radius_);
  get<0, 2>(*spherical_metric) = 0.0;
  get<0, 3>(*spherical_metric) = 0.0;

  get<1, 1>(*spherical_metric) =
      (square(extraction_radius_) - extraction_radius_ * du_wave_f - wave_f) *
      (pow<3>(extraction_radius_) + 2.0 * mass_ * square(extraction_radius_) +
       (extraction_radius_ - 2.0 * mass_) *
           (extraction_radius_ * du_wave_f + wave_f)) /
      pow<5>(extraction_radius_);
  get<1, 2>(*spherical_metric) = 0.0;
  get<1, 3>(*spherical_metric) = 0.0;
  get<2, 2>(*spherical_metric) = square(extraction_radius_);
  get<2, 3>(*spherical_metric) = 0.0;
  get<3, 3>(*spherical_metric) = square(extraction_radius_);
}

void GaugeWave::dr_spherical_metric(
    const gsl::not_null<
        tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>*>
        dr_spherical_metric,
    const size_t l_max, const double time) const {
  const auto wave_f = coordinate_wave_function(time);
  const auto du_wave_f = du_coordinate_wave_function(time);
  // for simpler expressions, we take advantage of the F derivatives evaluated
  // in the dt function (because the F function depends only on retarded time
  // t - r)
  dt_spherical_metric(dr_spherical_metric, l_max, time);

  get<0, 0>(*dr_spherical_metric) =
      -get<0, 0>(*dr_spherical_metric) +
      2.0 / pow<4>(extraction_radius_) * (extraction_radius_ + du_wave_f) *
          (-mass_ * extraction_radius_ +
           (extraction_radius_ - 3.0 * mass_) * du_wave_f);

  get<0, 1>(*dr_spherical_metric) =
      -get<0, 1>(*dr_spherical_metric) -
      (2.0 * mass_ * pow<3>(extraction_radius_) +
       2.0 * extraction_radius_ * wave_f * (extraction_radius_ - 3.0 * mass_) +
       du_wave_f * (pow<3>(extraction_radius_) +
                    wave_f * (3.0 * extraction_radius_ - 8.0 * mass_)) +
       2.0 * extraction_radius_ * square(du_wave_f) *
           (extraction_radius_ - 3.0 * mass_)) /
          pow<5>(extraction_radius_);

  get<0, 2>(*dr_spherical_metric) = 0.0;
  get<0, 3>(*dr_spherical_metric) = 0.0;

  get<1, 1>(*dr_spherical_metric) =
      -get<1, 1>(*dr_spherical_metric) +
      2.0 *
          (-mass_ * pow<4>(extraction_radius_) +
           square(wave_f) * (2.0 * extraction_radius_ - 5.0 * mass_) +
           du_wave_f * square(extraction_radius_) *
               (4.0 * mass_ * extraction_radius_ +
                du_wave_f * (extraction_radius_ - 3.0 * mass_)) +
           wave_f * extraction_radius_ *
               (6.0 * mass_ * extraction_radius_ +
                du_wave_f * (3.0 * extraction_radius_ - 8.0 * mass_))) /
          pow<6>(extraction_radius_);

  get<1, 2>(*dr_spherical_metric) = 0.0;
  get<1, 3>(*dr_spherical_metric) = 0.0;
  get<2, 2>(*dr_spherical_metric) = 2.0 * extraction_radius_;
  get<2, 3>(*dr_spherical_metric) = 0.0;
  get<3, 3>(*dr_spherical_metric) = 2.0 * extraction_radius_;
}

void GaugeWave::dt_spherical_metric(
    const gsl::not_null<
        tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>*>
        dt_spherical_metric,
    const size_t /*l_max*/, const double time) const {
  const auto wave_f = coordinate_wave_function(time);
  const auto du_wave_f = du_coordinate_wave_function(time);
  const auto du_du_wave_f = du_du_coordinate_wave_function(time);

  get<0, 0>(*dt_spherical_metric) =
      -2.0 * du_du_wave_f / pow<3>(extraction_radius_) *
      (extraction_radius_ - 2.0 * mass_) * (extraction_radius_ + du_wave_f);
  get<0, 1>(*dt_spherical_metric) =
      (du_du_wave_f * (2.0 * mass_ * square(extraction_radius_) +
                       (extraction_radius_ - 2.0 * mass_) *
                           (extraction_radius_ * du_wave_f + wave_f)) +
       (extraction_radius_ + du_wave_f) * (extraction_radius_ - 2.0 * mass_) *
           (extraction_radius_ * du_du_wave_f + du_wave_f)) /
      pow<4>(extraction_radius_);

  get<0, 2>(*dt_spherical_metric) = 0.0;
  get<0, 3>(*dt_spherical_metric) = 0.0;

  get<1, 1>(*dt_spherical_metric) =
      (-(extraction_radius_ * du_du_wave_f + du_wave_f) *
           (pow<3>(extraction_radius_) +
            2.0 * mass_ * square(extraction_radius_) +
            (extraction_radius_ - 2.0 * mass_) *
                (extraction_radius_ * du_wave_f + wave_f)) +
       (square(extraction_radius_) - extraction_radius_ * du_wave_f - wave_f) *
           (extraction_radius_ - 2.0 * mass_) *
           (extraction_radius_ * du_du_wave_f + du_wave_f)) /
      pow<5>(extraction_radius_);

  get<1, 2>(*dt_spherical_metric) = 0.0;
  get<1, 3>(*dt_spherical_metric) = 0.0;
  get<2, 2>(*dt_spherical_metric) = 0.0;
  get<2, 3>(*dt_spherical_metric) = 0.0;
  get<3, 3>(*dt_spherical_metric) = 0.0;
}

void GaugeWave::variables_impl(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -2>>*> news,
    const size_t /*l_max*/, const double /*time*/,
    tmpl::type_<Tags::News> /*meta*/) const {
  get(*news).data() = 0.0;
}

void GaugeWave::variables_impl(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> st_psi,
    size_t output_l_max, double time,
    tmpl::type_<Tags::BondiSTPsi> /*meta*/) const {
  const auto wave_f = coordinate_wave_function(time);
  auto r = extraction_radius_;
  auto r_fac = 4 * log(r / 2. - 1.);

  double u = time - r + wave_f / r - r_fac;
  detail5::bc_psi(get(*st_psi).data(), u, r);
}

void GaugeWave::variables_impl(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> st_psi,
    size_t output_l_max, double time,
    tmpl::type_<Tags::BondiSTTheta> /*meta*/) const {
  const auto wave_f = coordinate_wave_function(time);
  const auto du_wave_f = du_coordinate_wave_function(time);
  auto r = extraction_radius_;
  auto r_fac = 4 * log(r / 2. - 1.);

  double u = time - r + wave_f / r - r_fac;
  double du = 1 + du_wave_f / r;
  detail5::bc_theta(get(*st_psi).data(), u, r, du);
}

void GaugeWave::pup(PUP::er& p) {
  SphericalMetricData::pup(p);
  p | mass_;
  p | frequency_;
  p | amplitude_;
  p | peak_time_;
  p | duration_;
}

PUP::able::PUP_ID GaugeWave::my_PUP_ID = 0;
}  // namespace Cce::Solutions
