// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Domain/FaceNormal.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/ReceiveTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"

/// \cond
class DataVector;

namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
/// \endcond
namespace GeneralizedHarmonic {

//TODO : comment: Dim only works for 3

template <size_t Dim, typename Frame>
struct InterpolatePsi0 {
  using return_tags = tmpl::list<Tags::Psi0FromCceInterpolate>;
  using argument_tags = tmpl::list<domain::Tags::Coordinates<Dim, Frame>,
                        Cce::Tags::BoundaryValue<Cce::Tags::Psi0Match>,
                        Cce::Tags::LMax>;
  static void interpolate_psi0(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> psi0_inte,
      const typename domain::Tags::Coordinates<Dim, Frame>::type&
          inertial_coords,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& psi0,
      size_t l_max) noexcept;
};

template <size_t Dim, typename Frame>
struct AngularTetradForCCM {
  using return_tags = tmpl::list<Tags::AngularTetrad<Dim, Frame>>;
  using argument_tags = tmpl::list<domain::Tags::Coordinates<Dim, Frame>>;

  static void get_m_vector(
      gsl::not_null<tnsr::a<ComplexDataVector, Dim, Frame>*> m,
      const typename domain::Tags::Coordinates<Dim, Frame>::type&
          inertial_coords) noexcept;
};

template <size_t Dim, typename Frame>
struct IncomingWFromCCE {
  using return_tags = tmpl::list<Tags::CCMw<Dim, Frame>>;
  using argument_tags = tmpl::list<Tags::AngularTetrad<Dim, Frame>,
                                   Tags::Psi0FromCceInterpolate>;
  static void wijfromcce(
      gsl::not_null<tnsr::aa<DataVector, Dim, Frame>*> w_ccm,
      const tnsr::a<ComplexDataVector, Dim, Frame>& m,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& psi0_cce) noexcept;
};
}  // namespace GeneralizedHarmonic
