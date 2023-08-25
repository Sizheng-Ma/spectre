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
  using argument_tags = tmpl::list<Tags::LMax, Tags::NumberOfRadialPoints,
                                   Tags::BoundaryValue<Tags::BondiSTPsi>,
                                   Tags::BoundaryValue<Tags::BondiR>>;

  static void apply(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> bondi_st_psi,
      const size_t l_max, const size_t number_of_radial_points,
      const Scalar<SpinWeighted<ComplexDataVector, 0>> st_psi_boundary,
      const Scalar<SpinWeighted<ComplexDataVector, 0>> bondi_r);
};

}  // namespace ScalarTensor
}  // namespace Cce
