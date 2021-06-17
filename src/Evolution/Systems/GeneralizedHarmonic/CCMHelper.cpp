// Distributed under the MIT License.
// See LICENSE.txt for details.


#include "Evolution/Systems/GeneralizedHarmonic/CCMHelper.hpp"

#include "NumericalAlgorithms/Spectral/SwshInterpolation.hpp"

#include "Parallel/Printf.hpp"

namespace GeneralizedHarmonic {

template <size_t Dim, typename Frame>
void InterpolatePsi0<Dim, Frame>::apply(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> psi0_inte,
    const typename domain::Tags::Coordinates<Dim, Frame>::type& inertial_coords,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& psi0,
    size_t l_max) noexcept {
  DataVector theta(get_size(get<0>(inertial_coords)), 0.);
  DataVector phi(get_size(get<0>(inertial_coords)), 0.);

  theta = atan2(sqrt(square(inertial_coords.get(0))
                    +square(inertial_coords.get(1))),
          inertial_coords.get(2));
  phi = atan2(inertial_coords.get(1), inertial_coords.get(0));
  Spectral::Swsh::SwshInterpolator interpolator{theta, phi, l_max};
  interpolator.interpolate(make_not_null(&get(*psi0_inte)), get(psi0));
}

template <size_t Dim, typename Frame>
void AngularTetradForCCM<Dim, Frame>::apply(
    gsl::not_null<tnsr::a<ComplexDataVector, Dim, Frame>*> m,
    const typename domain::Tags::Coordinates<Dim, Frame>::type&
          inertial_coords) noexcept {
  tnsr::a<DataVector, Dim, Frame> theta_vec;
  tnsr::a<DataVector, Dim, Frame> phi_vec;
  DataVector theta(get_size(get<0>(inertial_coords)), 0.);
  DataVector phi(get_size(get<0>(inertial_coords)), 0.);

  theta = atan2(sqrt(square(inertial_coords.get(0))
                    +square(inertial_coords.get(1))),
          inertial_coords.get(2));
  phi = atan2(inertial_coords.get(1), inertial_coords.get(0));

  // FIXME better phi_vec.get(3) and (*m).get(0)
  // TODO check later
  theta_vec.get(0) = 0.*cos(phi);
  theta_vec.get(1) = cos(theta) * cos(phi);
  theta_vec.get(2) = cos(theta) * sin(phi);
  theta_vec.get(3) = -sin(theta);

  phi_vec.get(0) = 0.*cos(phi);
  phi_vec.get(1) = -sin(phi);
  phi_vec.get(2) = cos(phi);
  phi_vec.get(3) = 0.*cos(phi);

  (*m).get(0) = std::complex<double>(0.0,0.0)*cos(phi) ;
  (*m).get(2) = std::complex<double>(1.0,0.0) * cos(theta) * sin(phi)/sqrt(2.0);
  (*m).get(2) +=
std::complex<double>(0.0,1.0) * cos(phi) * sin(theta)/sqrt(2.0);
  (*m).get(1) = std::complex<double>(1.0,0.0) * cos(theta) * cos(phi)/sqrt(2.0);
  (*m).get(1) -=
std::complex<double>(0.0,1.0) * sin(phi) * sin(theta)/sqrt(2.0);
  (*m).get(3) = (std::complex<double>(-1.0,0.0) * sin(theta))/sqrt(2.0);
}

template <size_t Dim, typename Frame>
void IncomingWFromCCE<Dim, Frame>::apply(
    gsl::not_null<tnsr::aa<DataVector, Dim, Frame>*> w_ccm,
    const tnsr::a<ComplexDataVector, Dim, Frame>& m,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& psi0_cce) noexcept {
  //TODO change w_ccm to real number
  for (size_t a = 0; a <= Dim; ++a)
    for (size_t b = 0; b < a + 1; ++b)
      //(*w_ccm).get(a,b) = conj(get(psi0_cce).data()) * m.get(a) * m.get(b) +
      //                 get(psi0_cce).data() * conj(m.get(a)) * conj(m.get(b));
      (*w_ccm).get(a,b) = 2.*real(conj(get(psi0_cce).data()) *
                          m.get(a) * m.get(b));
}

template struct IncomingWFromCCE<3,Frame::Inertial>;
template struct AngularTetradForCCM<3,Frame::Inertial>;
template struct InterpolatePsi0<3,Frame::Inertial>;
}  // namespace GeneralizedHarmonic
