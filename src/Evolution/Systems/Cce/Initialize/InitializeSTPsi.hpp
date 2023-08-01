// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/Cce/Tags.hpp"

namespace Cce {

namespace Tags {
/// \cond
struct LMax;
struct NumberOfRadialPoints;
/// \endcond
}  // namespace Tags

namespace ScalarTensor {

struct InitializeSTPsi {
  using return_tags = tmpl::list<Tags::BondiSTPsi>;
  using argument_tags = tmpl::list<Tags::LMax, Tags::NumberOfRadialPoints>;

  static void apply(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> bondi_st_psi,
      const size_t l_max, const size_t number_of_radial_points);
};

}  // namespace ScalarTensor
}  // namespace Cce
