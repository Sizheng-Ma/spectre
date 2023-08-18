// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/Equations.hpp"

#include <complex>
#include <iostream>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "Utilities/ConstantExpressions.hpp"

// IWYU pragma: no_forward_declare Cce::Tags::BondiBeta
// IWYU pragma: no_forward_declare Cce::Tags::BondiH
// IWYU pragma: no_forward_declare Cce::Tags::BondiQ
// IWYU pragma: no_forward_declare Cce::Tags::BondiU
// IWYU pragma: no_forward_declare Cce::Tags::BondiW
// IWYU pragma: no_forward_declare Cce::Tags::Integrand
// IWYU pragma: no_forward_declare Cce::Tags::LinearFactor
// IWYU pragma: no_forward_declare Cce::Tags::LinearFactorForConjugate
// IWYU pragma: no_forward_declare Cce::Tags::PoleOfIntegrand
// IWYU pragma: no_forward_declare Cce::Tags::RegularIntegrand
// IWYU pragma: no_forward_declare SpinWeighted

namespace Cce {
// suppresses doxygen problems with these functions

void ComputeBondiIntegrand<Tags::Integrand<Tags::BondiSTduXInt>>::apply_impl(
    gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> integrand_for_duX,
    const SpinWeighted<ComplexDataVector, 0>& ethethbar_st_X,
    const SpinWeighted<ComplexDataVector, 0>& one_minus_y,
    const SpinWeighted<ComplexDataVector, 0>& beta,
    const SpinWeighted<ComplexDataVector, 0>& bondi_r,
    const SpinWeighted<ComplexDataVector, 0>& bondi_st_X) {
  *integrand_for_duX = (-bondi_st_X * one_minus_y / bondi_r * exp(-4. * beta) +
                        exp(2. * beta) * ethethbar_st_X) /
                       (2. * bondi_r);
}

namespace detail {

void BH(SpinWeighted<ComplexDataVector, 0>& result,
        const SpinWeighted<ComplexDataVector, 0>& eth_ethbar_st_psi,
        const SpinWeighted<ComplexDataVector, 0>& one_minus_y,
        const SpinWeighted<ComplexDataVector, 0>& bondi_r,
        const SpinWeighted<ComplexDataVector, 0>& dy_st_psi,
        const SpinWeighted<ComplexDataVector, 0>& dy_dy_st_psi) {
  result = 0.25 * eth_ethbar_st_psi / bondi_r;

  const auto radius_inverse = 0.5 * one_minus_y / bondi_r;
  const auto dpsidroone_minus_y = 0.5 * (one_minus_y) / bondi_r * dy_st_psi;
  SpinWeighted<ComplexDataVector, 0> ddpsidrdr =
      0.5 * square(one_minus_y) / bondi_r * dy_dy_st_psi -
      one_minus_y / bondi_r * dy_st_psi;

  auto rtimesddpsidrdroone_minus_y = ddpsidrdr;

  auto ddpsidrdroone_minus_y = ddpsidrdr * (0.5 * (one_minus_y) / bondi_r);

  result += 0.5 * (rtimesddpsidrdroone_minus_y - 2. * ddpsidrdroone_minus_y +
                   2 * (1. - radius_inverse) * dpsidroone_minus_y);
}

void flat_spacetime(
    SpinWeighted<ComplexDataVector, 0>& result,
    const SpinWeighted<ComplexDataVector, 0>& bondi_r,
    const SpinWeighted<ComplexDataVector, 1>& eth_r_divided_by_r,
    const SpinWeighted<ComplexDataVector, 0>& one_minus_y,
    const SpinWeighted<ComplexDataVector, 0>& dy_dy_st_psi,
    const SpinWeighted<ComplexDataVector, 1>& eth_dy_st_psi,
    const SpinWeighted<ComplexDataVector, 0>& ethbar_eth_r_divided_by_r,
    const SpinWeighted<ComplexDataVector, 0>& eth_ethbar_st_psi,
    const SpinWeighted<ComplexDataVector, 0>& dy_st_psi,
    const SpinWeighted<ComplexDataVector, 0>& du_r_divided_by_r) {
  SpinWeighted<ComplexDataVector, 0> from_lhs =
      du_r_divided_by_r * one_minus_y * dy_dy_st_psi;
  result =
      -0.25 * eth_r_divided_by_r / bondi_r * conj(eth_dy_st_psi) * one_minus_y;
  result -= conj(result);
  result += 0.25 * eth_r_divided_by_r * conj(eth_r_divided_by_r) / bondi_r *
            square(one_minus_y) * dy_dy_st_psi;
  result -=
      0.25 * ethbar_eth_r_divided_by_r / bondi_r * one_minus_y * dy_st_psi;
  result += eth_ethbar_st_psi / bondi_r * 0.25;
  result += 0.25 * square(one_minus_y) * dy_dy_st_psi / bondi_r + from_lhs;
}

void Npsi1(SpinWeighted<ComplexDataVector, 0>& result,
           const SpinWeighted<ComplexDataVector, 1>& eth_beta,
           const SpinWeighted<ComplexDataVector, 1>& eth_st_psi,
           const SpinWeighted<ComplexDataVector, 0>& eth_ethbar_st_psi,
           const SpinWeighted<ComplexDataVector, 0>& bondi_k) {
  result.data() = (2. * real(eth_beta.data() * conj(eth_st_psi).data()) +
                   eth_ethbar_st_psi.data()) *
                  bondi_k.data();
}

void Npsi2(SpinWeighted<ComplexDataVector, 0>& result,
           const SpinWeighted<ComplexDataVector, 2>& j,
           const SpinWeighted<ComplexDataVector, 2>& eth_eth_st_psi,
           const SpinWeighted<ComplexDataVector, 1>& eth_beta,
           const SpinWeighted<ComplexDataVector, 1>& eth_st_psi,
           const SpinWeighted<ComplexDataVector, 1>& ethbar_j) {
  result.data() = real(conj(j).data() * eth_eth_st_psi.data());
  result.data() +=
      2 * real(conj(eth_beta).data() * conj(eth_st_psi).data() * j.data());
  result.data() += real(ethbar_j.data() * conj(eth_st_psi).data());
}

void Npsi3(SpinWeighted<ComplexDataVector, 0>& result,
           const SpinWeighted<ComplexDataVector, 1>& eth_k,
           const SpinWeighted<ComplexDataVector, 1>& eth_st_psi) {
  result.data() = real(eth_k.data() * conj(eth_st_psi).data());
}

void Npsi4DividedbyOneMinuesY2(
    SpinWeighted<ComplexDataVector, 0>& result,
    const SpinWeighted<ComplexDataVector, 1>& eth_st_psi,
    const SpinWeighted<ComplexDataVector, 1>& dy_bondi_u,
    const SpinWeighted<ComplexDataVector, 0>& ethbar_u,
    const SpinWeighted<ComplexDataVector, 1>& bondi_u,
    const SpinWeighted<ComplexDataVector, 1>& eth_dy_st_psi,
    const SpinWeighted<ComplexDataVector, 0>& dy_st_psi,
    const SpinWeighted<ComplexDataVector, 0>& bondi_r,
    const SpinWeighted<ComplexDataVector, 1>& eth_r_divided_by_r) {
  auto dr_u = 0.5 / bondi_r * dy_bondi_u;
  auto dr_psi = 0.5 / bondi_r * dy_st_psi;
  auto eth_dr_psi =
      0.5 / bondi_r * (eth_dy_st_psi + eth_r_divided_by_r * dy_st_psi);

  auto res = 2 * conj(eth_st_psi) * dr_u + 2 * conj(ethbar_u) * dr_psi +
             4 * conj(bondi_u) * eth_dr_psi;
  result.data() = real(res.data());
}

void Npsi5(SpinWeighted<ComplexDataVector, 0>& result,
           const SpinWeighted<ComplexDataVector, 1>& bondi_u,
           const SpinWeighted<ComplexDataVector, 1>& eth_st_psi) {
  result.data() = 2 * real(bondi_u.data() * conj(eth_st_psi).data());
}

void Tau(SpinWeighted<ComplexDataVector, 0>& result,
         const SpinWeighted<ComplexDataVector, 0>& one_minus_y,
         const SpinWeighted<ComplexDataVector, 0>& dy_w,
         const SpinWeighted<ComplexDataVector, 0>& dy_st_psi,
         const SpinWeighted<ComplexDataVector, 0>& dy_dy_st_psi,
         const SpinWeighted<ComplexDataVector, 0>& bondi_w,
         const SpinWeighted<ComplexDataVector, 0>& bondi_r) {
  result = 0.5 * one_minus_y * dy_w * dy_st_psi +
           0.25 * square(one_minus_y) / bondi_r * dy_dy_st_psi +
           0.5 * one_minus_y * bondi_w * dy_dy_st_psi +
           0.5 * bondi_w * dy_st_psi;
}

void compute_norm(
    const SpinWeighted<ComplexDataVector, 0> to_compare,
    const SpinWeighted<ComplexDataVector, 0> regular_integrand_for_st_theta) {
  SpinWeighted<ComplexDataVector, 0> final_diff =
      to_compare - (regular_integrand_for_st_theta);

  double norm = 0;

  for (size_t iiiii = 0; iiiii < final_diff.size(); iiiii++) {
    norm += square(abs(final_diff.data()[iiiii]));
  }

  norm /= final_diff.size();

  norm = sqrt(norm);

  std::cout << norm << std::endl;
}
}  // namespace detail

void ComputeBondiIntegrand<Tags::PoleOfIntegrand<Tags::BondiSTTheta>>::
    apply_impl(gsl::not_null<SpinWeighted<ComplexDataVector, 0>*>
                   pole_of_integrand_for_st_theta,
               const SpinWeighted<ComplexDataVector, 1>& eth_st_psi,
               const SpinWeighted<ComplexDataVector, 1>& bondi_u) {
  SpinWeighted<ComplexDataVector, 0> n_psi5;
  detail::Npsi5(n_psi5, bondi_u, eth_st_psi);

  *pole_of_integrand_for_st_theta =
      0.5 * (-eth_st_psi * conj(bondi_u) - conj(eth_st_psi) * bondi_u);

  //   detail::compute_norm(*pole_of_integrand_for_st_theta, -0.5 * n_psi5);
  //   *pole_of_integrand_for_st_theta = -0.5 * n_psi5;
}

void ComputeBondiIntegrand<Tags::RegularIntegrand<Tags::BondiSTTheta>>::
    apply_impl(
        gsl::not_null<SpinWeighted<ComplexDataVector, 0>*>
            regular_integrand_for_st_theta,
        const SpinWeighted<ComplexDataVector, 0>& dy_dy_st_psi,
        const SpinWeighted<ComplexDataVector, 0>& dy_st_psi,
        const SpinWeighted<ComplexDataVector, 0>& dy_beta,
        const SpinWeighted<ComplexDataVector, 2>& dy_j,
        const SpinWeighted<ComplexDataVector, 1>& dy_bondi_u,
        const SpinWeighted<ComplexDataVector, 0>& dy_w,
        // swsh_derivative_tags
        const SpinWeighted<ComplexDataVector, 1>& eth_dy_st_psi,
        const SpinWeighted<ComplexDataVector, 2>& eth_eth_st_psi,
        const SpinWeighted<ComplexDataVector, 1>& eth_st_psi,
        const SpinWeighted<ComplexDataVector, 1>& ethbar_j,
        const SpinWeighted<ComplexDataVector, 0>& ethbar_u,
        const SpinWeighted<ComplexDataVector, 1>& eth_beta,
        const SpinWeighted<ComplexDataVector, 1>& eth_k,
        const SpinWeighted<ComplexDataVector, 0>& eth_ethbar_st_psi,
        // swsh_derivative_tags
        const SpinWeighted<ComplexDataVector, 2>& j,
        const SpinWeighted<ComplexDataVector, 0>& exp2beta,
        const SpinWeighted<ComplexDataVector, 0>& du_r_divided_by_r,
        const SpinWeighted<ComplexDataVector, 0>& one_minus_y,
        const SpinWeighted<ComplexDataVector, 1>& eth_r_divided_by_r,
        const SpinWeighted<ComplexDataVector, 2>& eth_eth_r_divided_by_r,
        const SpinWeighted<ComplexDataVector, 0>& bondi_r,
        const SpinWeighted<ComplexDataVector, 0>& bondi_k,
        const SpinWeighted<ComplexDataVector, 1>& bondi_u,
        const SpinWeighted<ComplexDataVector, 0>& ethbar_eth_r_divided_by_r,
        const SpinWeighted<ComplexDataVector, 0>& bondi_w) {
  SpinWeighted<ComplexDataVector, 0> to_compare;

  //   detail::flat_spacetime(to_compare, bondi_r, eth_r_divided_by_r,
  //   one_minus_y,
  //                          dy_dy_st_psi, eth_dy_st_psi,
  //                          ethbar_eth_r_divided_by_r, eth_ethbar_st_psi,
  //                          dy_st_psi, du_r_divided_by_r);
  detail::BH(to_compare, eth_ethbar_st_psi, one_minus_y, bondi_r, dy_st_psi,
             dy_dy_st_psi);
  SpinWeighted<ComplexDataVector, 0> from_lhs =
      du_r_divided_by_r * one_minus_y * dy_dy_st_psi;

  SpinWeighted<ComplexDataVector, 0> real6;
  SpinWeighted<ComplexDataVector, 0> real5;
  SpinWeighted<ComplexDataVector, 0> real4;
  SpinWeighted<ComplexDataVector, 0> real3;
  SpinWeighted<ComplexDataVector, 0> real2;
  SpinWeighted<ComplexDataVector, 0> real1;
  SpinWeighted<ComplexDataVector, 0> tmptt;
  SpinWeighted<ComplexDataVector, 0> real;

  real6 = 0.5 * one_minus_y * dy_dy_st_psi * bondi_w - dy_st_psi * bondi_w;
  real5 = 0.25 * square(one_minus_y) * dy_dy_st_psi / bondi_r -
          0.5 * one_minus_y * dy_st_psi / bondi_r;
  real4 = 0.5 * one_minus_y * dy_w * dy_st_psi;
  real3 = 1.5 * dy_st_psi * bondi_w;
  real2 = 0.5 * one_minus_y * dy_st_psi / bondi_r;
  tmptt = -0.25 * exp2beta * bondi_k * eth_r_divided_by_r / bondi_r *
          conj(eth_dy_st_psi) * one_minus_y;
  real1 = tmptt + conj(tmptt);
  real1 += 0.25 * exp2beta * bondi_k * eth_r_divided_by_r *
           conj(eth_r_divided_by_r) / bondi_r * square(one_minus_y) *
           dy_dy_st_psi;
  real1 -= 0.25 * exp2beta * bondi_k * ethbar_eth_r_divided_by_r / bondi_r *
           one_minus_y * dy_st_psi;
  real1 += 0.25 * exp2beta * bondi_k / bondi_r * eth_ethbar_st_psi;
  real = real1 + real2 + real3 + real4 + real5 + real6;

  SpinWeighted<ComplexDataVector, 0> complex6;
  SpinWeighted<ComplexDataVector, 0> complex5;
  SpinWeighted<ComplexDataVector, 0> complex4;
  SpinWeighted<ComplexDataVector, 0> complex3;
  SpinWeighted<ComplexDataVector, 0> complex2;
  SpinWeighted<ComplexDataVector, 0> complex1;
  SpinWeighted<ComplexDataVector, 0> complex_final;

  SpinWeighted<ComplexDataVector, 0> dy_k;

  dy_k = 0.5 * (conj(j) * dy_j + j * conj(dy_j)) / bondi_k;

  complex6 = -0.5 * ethbar_u * dy_st_psi + 0.5 * conj(eth_r_divided_by_r) *
                                               one_minus_y * dy_bondi_u *
                                               dy_st_psi;
  complex5 =
      -0.5 * eth_st_psi * conj(dy_bondi_u) +
      0.5 * eth_r_divided_by_r * one_minus_y * conj(dy_bondi_u) * dy_st_psi;
  complex4 = -eth_dy_st_psi * conj(bondi_u) +
             eth_r_divided_by_r * one_minus_y * dy_dy_st_psi * conj(bondi_u) -
             eth_r_divided_by_r * dy_st_psi * conj(bondi_u);
  complex3 = eth_r_divided_by_r * dy_st_psi * conj(bondi_u);

  complex1 = -0.25 * exp2beta * conj(j) * square(eth_r_divided_by_r) / bondi_r *
             square(one_minus_y) * dy_dy_st_psi;
  complex1 += 0.5 * exp2beta * conj(j) * eth_dy_st_psi * eth_r_divided_by_r /
              bondi_r * one_minus_y;
  complex1 += 0.25 * exp2beta * conj(j) * eth_eth_r_divided_by_r / bondi_r *
              one_minus_y * dy_st_psi;
  complex1 -= 0.25 * exp2beta * conj(j) * eth_eth_st_psi / bondi_r;

  complex2 = -0.25 * exp2beta * square(eth_r_divided_by_r) / bondi_r *
             square(one_minus_y) * conj(dy_j) * dy_st_psi;
  complex2 += 0.25 * exp2beta * eth_r_divided_by_r * conj(eth_r_divided_by_r) /
              bondi_r * square(one_minus_y) * dy_k * dy_st_psi;
  complex2 -= 0.5 * exp2beta * conj(j) * square(eth_r_divided_by_r) / bondi_r *
              square(one_minus_y) * dy_beta * dy_st_psi;
  complex2 += 0.5 * exp2beta * bondi_k * eth_r_divided_by_r *
              conj(eth_r_divided_by_r) / bondi_r * square(one_minus_y) *
              dy_beta * dy_st_psi;
  complex2 += 0.25 * exp2beta * eth_r_divided_by_r / bondi_r * eth_st_psi *
              one_minus_y * conj(dy_j);
  complex2 -= 0.25 * exp2beta * conj(eth_r_divided_by_r) / bondi_r *
              eth_st_psi * one_minus_y * dy_k;
  complex2 += 0.5 * exp2beta * conj(j) * eth_r_divided_by_r / bondi_r *
              eth_st_psi * one_minus_y * dy_beta;
  complex2 -= 0.5 * exp2beta * bondi_k * eth_st_psi * conj(eth_r_divided_by_r) /
              bondi_r * one_minus_y * dy_beta;
  complex2 += 0.25 * exp2beta * conj(ethbar_j) * eth_r_divided_by_r / bondi_r *
              one_minus_y * dy_st_psi;

  complex2 += 0.5 * exp2beta * conj(j) * eth_r_divided_by_r / bondi_r *
              eth_beta * one_minus_y * dy_st_psi;
  complex2 -= 0.25 * exp2beta * eth_r_divided_by_r / bondi_r * conj(eth_k) *
              one_minus_y * dy_st_psi;
  complex2 -= 0.5 * exp2beta * bondi_k * eth_r_divided_by_r / bondi_r *
              conj(eth_beta) * one_minus_y * dy_st_psi;

  complex2 -= 0.25 * exp2beta * conj(ethbar_j) * eth_st_psi / bondi_r;
  complex2 -= 0.5 * exp2beta * conj(j) * eth_beta * eth_st_psi / bondi_r;
  complex2 += 0.25 * exp2beta * eth_st_psi * conj(eth_k) / bondi_r;
  complex2 += 0.5 * exp2beta * bondi_k * eth_st_psi * conj(eth_beta) / bondi_r;

  complex_final =
      complex1 + complex2 + complex3 + complex4 + complex5 + complex6;

  *regular_integrand_for_st_theta =
      0.5 * (complex_final + conj(complex_final)) + real + from_lhs;

  SpinWeighted<ComplexDataVector, 0> n_psi1;
  SpinWeighted<ComplexDataVector, 0> n_psi2;
  SpinWeighted<ComplexDataVector, 0> n_psi3;
  SpinWeighted<ComplexDataVector, 0> n_psi4;
  SpinWeighted<ComplexDataVector, 0> tau;

  detail::Npsi1(n_psi1, eth_beta, eth_st_psi, eth_ethbar_st_psi, bondi_k);
  detail::Npsi2(n_psi2, j, eth_eth_st_psi, eth_beta, eth_st_psi, ethbar_j);
  detail::Npsi3(n_psi3, eth_k, eth_st_psi);
  detail::Npsi4DividedbyOneMinuesY2(n_psi4, eth_st_psi, dy_bondi_u, ethbar_u,
                                    bondi_u, eth_dy_st_psi, dy_st_psi, bondi_r,
                                    eth_r_divided_by_r);
  detail::Tau(tau, one_minus_y, dy_w, dy_st_psi, dy_dy_st_psi, bondi_w,
              bondi_r);

  auto middle_result = 0.25 * exp2beta / bondi_r * (n_psi1 - n_psi2 + n_psi3) -
                       0.5 * bondi_r * n_psi4 + tau + from_lhs;

  detail::compute_norm(middle_result, *regular_integrand_for_st_theta);
}

void ComputeBondiIntegrand<Tags::Integrand<Tags::BondiBeta>>::apply_impl(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> integrand_for_beta,
    const SpinWeighted<ComplexDataVector, 2>& dy_j,
    const SpinWeighted<ComplexDataVector, 2>& j,
    const SpinWeighted<ComplexDataVector, 0>& dy_st_psi,
    const SpinWeighted<ComplexDataVector, 0>& one_minus_y) {
  *integrand_for_beta =
      0.125 * one_minus_y *
      (dy_j * conj(dy_j) -
       0.25 * square(j * conj(dy_j) + conj(j) * dy_j) / (1.0 + j * conj(j)));

  //   *integrand_for_beta += 2 * M_PI * one_minus_y * square(dy_st_psi);
}

void ComputeBondiIntegrand<Tags::PoleOfIntegrand<Tags::BondiQ>>::apply_impl(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 1>*>
        pole_of_integrand_for_q,
    const SpinWeighted<ComplexDataVector, 1>& eth_beta) {
  *pole_of_integrand_for_q = -4.0 * eth_beta;
}

void ComputeBondiIntegrand<Tags::RegularIntegrand<Tags::BondiQ>>::apply_impl(
    gsl::not_null<SpinWeighted<ComplexDataVector, 1>*> regular_integrand_for_q,
    gsl::not_null<SpinWeighted<ComplexDataVector, 1>*> script_aq,
    const SpinWeighted<ComplexDataVector, 0>& dy_beta,
    const SpinWeighted<ComplexDataVector, 2>& dy_j,
    const SpinWeighted<ComplexDataVector, 2>& j,
    const SpinWeighted<ComplexDataVector, 0>& dy_st_psi,
    const SpinWeighted<ComplexDataVector, 1>& eth_dy_beta,
    const SpinWeighted<ComplexDataVector, 1>& eth_j_jbar,
    const SpinWeighted<ComplexDataVector, 1>& eth_jbar_dy_j,
    const SpinWeighted<ComplexDataVector, 1>& ethbar_dy_j,
    const SpinWeighted<ComplexDataVector, 1>& ethbar_j,
    const SpinWeighted<ComplexDataVector, 1>& eth_st_psi,
    const SpinWeighted<ComplexDataVector, 1>& eth_r_divided_by_r,
    const SpinWeighted<ComplexDataVector, 0>& k,
    const SpinWeighted<ComplexDataVector, 0>& one_minus_y) {
  *script_aq =
      0.25 * (j * conj(ethbar_dy_j) - eth_jbar_dy_j - conj(ethbar_j) * dy_j +
              0.5 * eth_j_jbar * (conj(j) * dy_j + j * conj(dy_j)) /
                  (1.0 + j * conj(j)) -
              (conj(j) * dy_j - j * conj(dy_j)) * eth_r_divided_by_r);

  *regular_integrand_for_q =
      -2.0 * (*script_aq + j * conj(*script_aq) / k - eth_dy_beta +
              0.5 * ethbar_dy_j / k - dy_beta * eth_r_divided_by_r +
              0.5 * dy_j * conj(eth_r_divided_by_r) / k);

  //   *regular_integrand_for_q +=
  //       16. * M_PI * eth_st_psi * dy_st_psi -
  //       16. * M_PI * eth_r_divided_by_r * one_minus_y * square(dy_st_psi);
}

void ComputeBondiIntegrand<Tags::Integrand<Tags::BondiU>>::apply_impl(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 1>*>
        regular_integrand_for_u,
    const SpinWeighted<ComplexDataVector, 0>& exp_2_beta,
    const SpinWeighted<ComplexDataVector, 2>& j,
    const SpinWeighted<ComplexDataVector, 1>& q,
    const SpinWeighted<ComplexDataVector, 0>& k,
    const SpinWeighted<ComplexDataVector, 0>& r) {
  *regular_integrand_for_u = 0.5 * exp_2_beta / r * (k * q - j * conj(q));
}

void ComputeBondiIntegrand<Tags::PoleOfIntegrand<Tags::BondiW>>::apply_impl(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*>
        pole_of_integrand_for_w,
    const SpinWeighted<ComplexDataVector, 0>& ethbar_u) {
  *pole_of_integrand_for_w = ethbar_u + conj(ethbar_u);
}

void ComputeBondiIntegrand<Tags::RegularIntegrand<Tags::BondiW>>::apply_impl(
    gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> regular_integrand_for_w,
    gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> script_av,
    const SpinWeighted<ComplexDataVector, 1>& dy_u,
    const SpinWeighted<ComplexDataVector, 0>& exp_2_beta,
    const SpinWeighted<ComplexDataVector, 2>& j,
    const SpinWeighted<ComplexDataVector, 1>& q,
    const SpinWeighted<ComplexDataVector, 0>& dy_st_psi,
    const SpinWeighted<ComplexDataVector, 1>& eth_beta,
    const SpinWeighted<ComplexDataVector, 2>& eth_eth_beta,
    const SpinWeighted<ComplexDataVector, 0>& eth_ethbar_beta,
    const SpinWeighted<ComplexDataVector, 2>& eth_ethbar_j,
    const SpinWeighted<ComplexDataVector, 0>& eth_ethbar_j_jbar,
    const SpinWeighted<ComplexDataVector, 1>& eth_j_jbar,
    const SpinWeighted<ComplexDataVector, 0>& ethbar_dy_u,
    const SpinWeighted<ComplexDataVector, 0>& ethbar_ethbar_j,
    const SpinWeighted<ComplexDataVector, 1>& ethbar_j,
    const SpinWeighted<ComplexDataVector, 1>& eth_st_psi,
    const SpinWeighted<ComplexDataVector, 1>& eth_r_divided_by_r,
    const SpinWeighted<ComplexDataVector, 0>& k,
    const SpinWeighted<ComplexDataVector, 0>& r,
    const SpinWeighted<ComplexDataVector, 0>& one_minus_y)

{
  // this computation is split over two lines because GCC-10 on release mode
  // optimizes the long expression templates in such a way to cause segfaults.
  *script_av =
      eth_beta * conj(ethbar_j) + 0.5 * ethbar_ethbar_j +
      j * square(conj(eth_beta)) + j * conj(eth_eth_beta) +
      0.125 * eth_j_jbar * conj(eth_j_jbar) / (k * (1.0 + j * conj(j))) +
      0.5 *
          (1.0 - 0.25 * eth_ethbar_j_jbar - eth_j_jbar * conj(eth_beta) -
           0.5 * conj(ethbar_j) * ethbar_j - 0.5 * conj(j) * eth_ethbar_j) /
          k;
  *script_av += k * (0.5 - eth_ethbar_beta - eth_beta * conj(eth_beta) -
                     0.25 * q * conj(q)) +
                0.25 * j * square(conj(q));

  *regular_integrand_for_w =
      0.5 * (0.5 * (ethbar_dy_u + conj(ethbar_dy_u) +
                    conj(dy_u) * eth_r_divided_by_r +
                    dy_u * conj(eth_r_divided_by_r)) -
             1.0 / r + 0.5 * exp_2_beta * (*script_av + conj(*script_av)) / r);

  SpinWeighted<ComplexDataVector, 0> complex1;
  SpinWeighted<ComplexDataVector, 0> complex2;
  SpinWeighted<ComplexDataVector, 0> complex3;
  SpinWeighted<ComplexDataVector, 0> complex_final;

  complex1 = 0.5 * j * square(one_minus_y) * square(conj(eth_r_divided_by_r)) *
             square(dy_st_psi) / r;
  complex2 = 0.5 * j * square(conj(eth_st_psi)) / r;
  complex3 = -j * one_minus_y * dy_st_psi * conj(eth_st_psi) *
             conj(eth_r_divided_by_r) / r;

  complex_final = complex1 + complex2 + complex3 + conj(complex1) +
                  conj(complex2) + conj(complex3);

  SpinWeighted<ComplexDataVector, 0> real1;
  SpinWeighted<ComplexDataVector, 0> real2;
  SpinWeighted<ComplexDataVector, 0> real3;
  SpinWeighted<ComplexDataVector, 0> real_final;

  real1 = -k * square(one_minus_y) * square(dy_st_psi) *
          conj(eth_r_divided_by_r) * eth_r_divided_by_r / r;

  real2 = -k * eth_st_psi * conj(eth_st_psi) / r;
  real3 =
      k * one_minus_y * dy_st_psi * eth_st_psi * conj(eth_r_divided_by_r) / r;

  real_final = real1 + real2 + real3 + conj(real3);

  //   *regular_integrand_for_w +=
  //       2 * M_PI * exp_2_beta * (complex_final + real_final);
}

void ComputeBondiIntegrand<Tags::PoleOfIntegrand<Tags::BondiH>>::apply_impl(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*>
        pole_of_integrand_for_h,
    const SpinWeighted<ComplexDataVector, 2>& j,
    const SpinWeighted<ComplexDataVector, 1>& u,
    const SpinWeighted<ComplexDataVector, 0>& w,
    const SpinWeighted<ComplexDataVector, 2>& eth_u,
    const SpinWeighted<ComplexDataVector, 1>& ethbar_j,
    const SpinWeighted<ComplexDataVector, -2>& ethbar_jbar_u,
    const SpinWeighted<ComplexDataVector, 0>& ethbar_u,
    const SpinWeighted<ComplexDataVector, 0>& k) {
  *pole_of_integrand_for_h = -0.5 * conj(ethbar_jbar_u) - j * conj(ethbar_u) -
                             0.5 * j * ethbar_u - k * eth_u -
                             0.5 * u * ethbar_j + 2.0 * j * w;
}

void ComputeBondiIntegrand<Tags::RegularIntegrand<Tags::BondiH>>::apply_impl(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*>
        regular_integrand_for_h,
    const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> script_aj,
    const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> script_bj,
    const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> script_cj,
    const SpinWeighted<ComplexDataVector, 2>& dy_dy_j,
    const SpinWeighted<ComplexDataVector, 2>& dy_j,
    const SpinWeighted<ComplexDataVector, 0>& dy_w,
    const SpinWeighted<ComplexDataVector, 0>& exp_2_beta,
    const SpinWeighted<ComplexDataVector, 2>& j,
    const SpinWeighted<ComplexDataVector, 1>& q,
    const SpinWeighted<ComplexDataVector, 1>& u,
    const SpinWeighted<ComplexDataVector, 0>& w,
    const SpinWeighted<ComplexDataVector, 0>& dy_st_psi,
    const SpinWeighted<ComplexDataVector, 1>& eth_beta,
    const SpinWeighted<ComplexDataVector, 2>& eth_eth_beta,
    const SpinWeighted<ComplexDataVector, 0>& eth_ethbar_beta,
    const SpinWeighted<ComplexDataVector, 2>& eth_ethbar_j,
    const SpinWeighted<ComplexDataVector, 0>& eth_ethbar_j_jbar,
    const SpinWeighted<ComplexDataVector, 1>& eth_j_jbar,
    const SpinWeighted<ComplexDataVector, 2>& eth_q,
    const SpinWeighted<ComplexDataVector, 2>& eth_u,
    const SpinWeighted<ComplexDataVector, 2>& eth_ubar_dy_j,
    const SpinWeighted<ComplexDataVector, 1>& ethbar_dy_j,
    const SpinWeighted<ComplexDataVector, 0>& ethbar_ethbar_j,
    const SpinWeighted<ComplexDataVector, 1>& ethbar_j,
    const SpinWeighted<ComplexDataVector, -1>& ethbar_jbar_dy_j,
    const SpinWeighted<ComplexDataVector, -2>& ethbar_jbar_q_minus_2_eth_beta,
    const SpinWeighted<ComplexDataVector, 0>& ethbar_q,
    const SpinWeighted<ComplexDataVector, 0>& ethbar_u,
    const SpinWeighted<ComplexDataVector, 1>& eth_st_psi,
    const SpinWeighted<ComplexDataVector, 0>& du_r_divided_by_r,
    const SpinWeighted<ComplexDataVector, 1>& eth_r_divided_by_r,
    const SpinWeighted<ComplexDataVector, 0>& k,
    const SpinWeighted<ComplexDataVector, 0>& one_minus_y,
    const SpinWeighted<ComplexDataVector, 0>& r) {
  *script_aj =
      0.25 *
      (conj(ethbar_ethbar_j) -
       0.25 * (4.0 + eth_ethbar_j_jbar - j * conj(eth_ethbar_j)) /
           (k * (1.0 + j * conj(j))) +
       (3.0 - eth_ethbar_beta -
        conj(j) * eth_ethbar_j * (1.0 - 0.25 / (1.0 + j * conj(j)))) /
           k +
       conj(ethbar_j) * (2.0 * eth_beta +
                         0.5 *
                             (j * conj(eth_j_jbar) -
                              ethbar_j * (2.0 * (1.0 + j * conj(j)) - 1.0)) /
                             (k * (1.0 + j * conj(j))) -
                         q));

  // this computation is split over multiple lines because GCC-10 on release
  // mode optimizes the long expression templates in such a way to cause
  // segfaults.
  *script_bj =
      0.25 *
      (2.0 * dy_w - conj(j) * eth_u * (conj(j) * dy_j + j * conj(dy_j)) / k +
       1.0 / r + u * ethbar_j * conj(dy_j) -
       0.5 * u * conj(eth_j_jbar) * (conj(j) * dy_j + j * conj(dy_j)) /
           (1.0 + j * conj(j)) +
       conj(u) * (conj(ethbar_jbar_dy_j) - j * conj(ethbar_dy_j)));
  *script_bj +=
      square(one_minus_y) * 0.125 *
      (0.25 * square(conj(j) * dy_j + j * conj(dy_j)) / (1.0 + j * conj(j)) -
       dy_j * conj(dy_j)) /
      r;
  *script_bj +=
      one_minus_y * 0.25 *
      (du_r_divided_by_r * dy_j *
           (conj(j) * (conj(j) * dy_j + j * conj(dy_j)) / (1.0 + j * conj(j)) -
            2.0 * conj(dy_j)) -
       w * (dy_j * conj(dy_j) - 0.25 *
                                    square((conj(j) * dy_j + j * conj(dy_j))) /
                                    (1.0 + j * conj(j))));

  *script_cj = 0.5 * ethbar_j * k * (eth_beta - 0.5 * q);

  *regular_integrand_for_h =
      j * (*script_bj + conj(*script_bj)) -
      0.5 * (eth_ubar_dy_j + u * ethbar_dy_j +
             u * dy_j * conj(eth_r_divided_by_r)) +
      0.5 * exp_2_beta / r *
          (*script_cj + square(j) / (1.0 + j * conj(j)) * conj(*script_cj) -
           j * (*script_aj + conj(*script_aj)) + eth_eth_beta - 0.5 * eth_q +
           0.25 * (conj(ethbar_jbar_q_minus_2_eth_beta) - j * conj(ethbar_q)) /
               k +
           square(eth_beta - 0.5 * q)) -
      dy_j * 0.5 *
          (conj(j) * eth_u / k - j * k * conj(eth_u) +
           0.5 * (1.0 + j * conj(j)) * (conj(ethbar_u) - ethbar_u) +
           conj(u) * eth_r_divided_by_r - w) +
      conj(dy_j) * (0.5 * j * eth_u * (j * conj(j) / k) -
                    0.25 * square(j) * (ethbar_u - conj(ethbar_u))) +
      one_minus_y *
          (0.5 * (dy_dy_j * (w + 2.0 * du_r_divided_by_r) - dy_j / r) +
           0.5 * dy_j * (dy_w + 1.0 / r)) +
      square(one_minus_y) * 0.25 * dy_dy_j / r;

  //   SpinWeighted<ComplexDataVector, 1> from_st;

  //   from_st = eth_st_psi - eth_r_divided_by_r * one_minus_y * dy_st_psi;
  //   *regular_integrand_for_h += 2 * M_PI * exp_2_beta / r * square(from_st);
}

void ComputeBondiIntegrand<Tags::LinearFactor<Tags::BondiH>>::apply_impl(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*>
        linear_factor_for_h,
    const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> script_djbar,
    const SpinWeighted<ComplexDataVector, 2>& dy_j,
    const SpinWeighted<ComplexDataVector, 2>& j,
    const SpinWeighted<ComplexDataVector, 0>& one_minus_y) {
  *script_djbar = 0.25 * one_minus_y *
                  (-2.0 * dy_j +
                   j * (conj(j) * dy_j + j * conj(dy_j)) / (1.0 + j * conj(j)));
  *linear_factor_for_h = 1.0 + j * conj(*script_djbar);
}

void ComputeBondiIntegrand<Tags::LinearFactorForConjugate<Tags::BondiH>>::
    apply_impl(
        const gsl::not_null<SpinWeighted<ComplexDataVector, 4>*>
            linear_factor_for_conjugate_h,
        const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> script_djbar,
        const SpinWeighted<ComplexDataVector, 2>& dy_j,
        const SpinWeighted<ComplexDataVector, 2>& j,
        const SpinWeighted<ComplexDataVector, 0>& one_minus_y) {
  *script_djbar = 0.25 * one_minus_y *
                  (-2.0 * dy_j +
                   j * (conj(j) * dy_j + j * conj(dy_j)) / (1.0 + j * conj(j)));
  *linear_factor_for_conjugate_h = j * (*script_djbar);
}
}  // namespace Cce
