// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/ComplexDataVector.hpp"

namespace Cce {
namespace Solutions {
namespace KleinGordon {
DataVector a0(const DataVector& u);
DataVector a0dot(const DataVector& u);

DataVector a2(const DataVector& u);
DataVector a2dot(const DataVector& u);

DataVector a3(const DataVector& u);
DataVector a3dot(const DataVector& u);

DataVector a4(const DataVector& u);
DataVector a4dot(const DataVector& u);

DataVector a5(const DataVector& u);
DataVector a5dot(const DataVector& u);

DataVector a6(const DataVector& u);
DataVector a6dot(const DataVector& u);

DataVector a7(const DataVector& u);
DataVector a7dot(const DataVector& u);

DataVector a8(const DataVector& u);
DataVector a8dot(const DataVector& u);

DataVector a9(const DataVector& u);
DataVector a9dot(const DataVector& u);

DataVector a10(const DataVector& u);
DataVector a10dot(const DataVector& u);

DataVector a11(const DataVector& u);
DataVector a11dot(const DataVector& u);

DataVector a12(const DataVector& u);
DataVector a12dot(const DataVector& u);

DataVector a13(const DataVector& u);
DataVector a13dot(const DataVector& u);

DataVector a14(const DataVector& u);
DataVector a14dot(const DataVector& u);

DataVector a15(const DataVector& u);
DataVector a15dot(const DataVector& u);

void bc_psi(gsl::not_null<ComplexDataVector*> theta, const DataVector& u,
            const DataVector& one_over_r);
void bc_theta(gsl::not_null<ComplexDataVector*> theta, const DataVector& u,
              const DataVector& r, const DataVector& dudt);
}  // namespace KleinGordon
}  // namespace Solutions
}  // namespace Cce
