// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/AnalyticSolutions/KleinGordonWaveHelper.hpp"

#include <cmath>

namespace Cce {
namespace Solutions {
namespace KleinGordon {
DataVector a0(const DataVector& u) { return sin(u); }
DataVector a0dot(const DataVector& u) { return cos(u); }

DataVector a2(const DataVector& u) { return -0.5 * cos(u); }
DataVector a2dot(const DataVector& u) { return 0.5 * sin(u); }

DataVector a3(const DataVector& u) { return 0.5 * sin(u); }
DataVector a3dot(const DataVector& u) { return 0.5 * cos(u); }

DataVector a4(const DataVector& u) { return 0.75 * cos(u) - 9. / 8. * sin(u); }
DataVector a4dot(const DataVector& u) {
  return -0.75 * sin(u) - 9. / 8. * cos(u);
}

DataVector a5(const DataVector& u) { return -77. / 20 * cos(u) - 1.5 * sin(u); }
DataVector a5dot(const DataVector& u) {
  return 77. / 20 * sin(u) - 1.5 * cos(u);
}

DataVector a6(const DataVector& u) {
  return 15. / 16 * cos(u) + 51. / 4. * sin(u);
}
DataVector a6dot(const DataVector& u) {
  return -15. / 16 * sin(u) + 51. / 4. * cos(u);
}

DataVector a7(const DataVector& u) {
  return (1287. * cos(u)) / 28. - (1809. * sin(u)) / 80.;
}
DataVector a7dot(const DataVector& u) {
  return -(1287. * sin(u)) / 28. - (1809. * cos(u)) / 80.;
}

DataVector a8(const DataVector& u) {
  return -(12579. * cos(u) / 80.) - (19857. * sin(u)) / 128.;
}
DataVector a8dot(const DataVector& u) {
  return (12579. * sin(u) / 80.) - (19857. * cos(u)) / 128.;
}

DataVector a9(const DataVector& u) {
  return -(73557. * cos(u) / 160) + (133813. * sin(u)) / 140.;
}
DataVector a9dot(const DataVector& u) {
  return (73557. * sin(u) / 160) + (133813. * cos(u)) / 140.;
}

DataVector a10(const DataVector& u) {
  return (49797063. * cos(u)) / 8960. + (1272267. * sin(u)) / 1600.;
}
DataVector a10dot(const DataVector& u) {
  return -(49797063. * sin(u)) / 8960. + (1272267. * cos(u)) / 1600.;
}

DataVector a11(const DataVector& u) {
  return -((116136241. * cos(u)) / 24640) - (57286503. * sin(u)) / 1792.;
}
DataVector a11dot(const DataVector& u) {
  return ((116136241. * sin(u)) / 24640) - (57286503. * cos(u)) / 1792.;
}

DataVector a12(const DataVector& u) {
  return -((16472195091. * cos(u)) / 89600) + (419653067. * sin(u)) / 5120.;
}
DataVector a12dot(const DataVector& u) {
  return ((16472195091. * sin(u)) / 89600) + (419653067. * cos(u)) / 5120.;
}

DataVector a13(const DataVector& u) {
  return ((197057851611. * cos(u)) / 232960) +
         (517853793843. * sin(u)) / 492800.;
}
DataVector a13dot(const DataVector& u) {
  return -((197057851611. * sin(u)) / 232960) +
         (517853793843. * cos(u)) / 492800.;
}

DataVector a14(const DataVector& u) {
  return ((23027722022071. * cos(u)) / 3942400) -
         (2420206444191. * sin(u)) / 313600.;
}
DataVector a14dot(const DataVector& u) {
  return -((23027722022071. * sin(u)) / 3942400) -
         (2420206444191. * cos(u)) / 313600.;
}

DataVector a15(const DataVector& u) {
  return -((166944468961581. * cos(u)) / 2464000) -
         (218435295225339. * sin(u)) / 7321600.;
}
DataVector a15dot(const DataVector& u) {
  return ((166944468961581. * sin(u)) / 2464000) -
         (218435295225339. * cos(u)) / 7321600.;
}

void bc_psi(const gsl::not_null<ComplexDataVector*> theta, const DataVector& u,
            const DataVector& one_over_r) {
  *theta = a0(u) * one_over_r;

  *theta += a2(u) * pow(one_over_r, 3);
  *theta += a3(u) * pow(one_over_r, 4);
  *theta += a4(u) * pow(one_over_r, 5);
  *theta += a5(u) * pow(one_over_r, 6);
  *theta += a6(u) * pow(one_over_r, 7);
  *theta += a7(u) * pow(one_over_r, 8);
  *theta += a8(u) * pow(one_over_r, 9);
  *theta += a9(u) * pow(one_over_r, 10);
  *theta += a10(u) * pow(one_over_r, 11);
  *theta += a11(u) * pow(one_over_r, 12);
  *theta += a12(u) * pow(one_over_r, 13);
  *theta += a13(u) * pow(one_over_r, 14);
  *theta += a14(u) * pow(one_over_r, 15);
  *theta += a15(u) * pow(one_over_r, 16);
}

void bc_theta(const gsl::not_null<ComplexDataVector*> theta,
              const DataVector& u, const DataVector& r,
              const DataVector& dudt) {
  DataVector drdt = 0 * dudt;

  *theta = a0dot(u) * dudt / r - (0 + 1) / square(r) * a0(u) * drdt;

  int iii = 2;
  *theta += a2dot(u) * dudt / pow(r, iii + 1) -
            (iii + 1) / pow(r, iii + 2) * a2(u) * drdt;

  iii = 3;
  *theta += a3dot(u) * dudt / pow(r, iii + 1) -
            (iii + 1) / pow(r, iii + 2) * a3(u) * drdt;

  iii = 4;
  *theta += a4dot(u) * dudt / pow(r, iii + 1) -
            (iii + 1) / pow(r, iii + 2) * a4(u) * drdt;

  iii = 5;
  *theta += a5dot(u) * dudt / pow(r, iii + 1) -
            (iii + 1) / pow(r, iii + 2) * a5(u) * drdt;

  iii = 6;
  *theta += a6dot(u) * dudt / pow(r, iii + 1) -
            (iii + 1) / pow(r, iii + 2) * a6(u) * drdt;

  iii = 7;
  *theta += a7dot(u) * dudt / pow(r, iii + 1) -
            (iii + 1) / pow(r, iii + 2) * a7(u) * drdt;

  iii = 8;
  *theta += a8dot(u) * dudt / pow(r, iii + 1) -
            (iii + 1) / pow(r, iii + 2) * a8(u) * drdt;

  iii = 9;
  *theta += a9dot(u) * dudt / pow(r, iii + 1) -
            (iii + 1) / pow(r, iii + 2) * a9(u) * drdt;

  iii = 10;
  *theta += a10dot(u) * dudt / pow(r, iii + 1) -
            (iii + 1) / pow(r, iii + 2) * a10(u) * drdt;

  iii = 11;
  *theta += a11dot(u) * dudt / pow(r, iii + 1) -
            (iii + 1) / pow(r, iii + 2) * a11(u) * drdt;

  iii = 12;
  *theta += a12dot(u) * dudt / pow(r, iii + 1) -
            (iii + 1) / pow(r, iii + 2) * a12(u) * drdt;

  iii = 13;
  *theta += a13dot(u) * dudt / pow(r, iii + 1) -
            (iii + 1) / pow(r, iii + 2) * a13(u) * drdt;

  iii = 14;
  *theta += a14dot(u) * dudt / pow(r, iii + 1) -
            (iii + 1) / pow(r, iii + 2) * a14(u) * drdt;

  iii = 15;
  *theta += a15dot(u) * dudt / pow(r, iii + 1) -
            (iii + 1) / pow(r, iii + 2) * a15(u) * drdt;
}
}  // namespace KleinGordon
}  // namespace Solutions
}  // namespace Cce
