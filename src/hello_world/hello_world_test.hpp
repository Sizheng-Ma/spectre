// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <complex>
#include <vector>

#include "Evolution/Systems/Cce/Tags.hpp"

void ccm_functions11(
    std::vector<double>& re_h, std::vector<double>& im_h, const size_t l_max,
    const size_t number_of_radial_points, const double radius,
    const std::vector<double>& re_j, const std::vector<double>& im_j,
    const std::vector<std::vector<double>>& cauchy_cart,
    const std::vector<std::vector<double>>& inertial_cart,
    const Cce::Tags::BoundaryValue<Cce::Tags::BondiBeta>::type& cce_bondi_beta,
    const Cce::Tags::BoundaryValue<Cce::Tags::Dr<Cce::Tags::BondiJ>>::type&
        cce_bondi_dr_j,
    const Cce::Tags::BoundaryValue<Cce::Tags::Du<Cce::Tags::BondiR>>::type&
        cce_du_R,
    const Cce::Tags::BoundaryValue<Cce::Tags::BondiH>::type& cce_bondi_h,
    const Cce::Tags::BoundaryValue<Cce::Tags::BondiJ>::type& cce_bondi_j,
    const Cce::Tags::BoundaryValue<Cce::Tags::BondiQ>::type& cce_bondi_q,
    const Cce::Tags::BoundaryValue<Cce::Tags::BondiR>::type& cce_bondi_R,
    const Cce::Tags::BoundaryValue<Cce::Tags::BondiU>::type& cce_bondi_u,
    const Cce::Tags::BoundaryValue<Cce::Tags::BondiW>::type& cce_bondi_w,
    const Cce::Tags::BoundaryValue<Cce::Tags::Dr<Cce::Tags::BondiU>>::type&
        cce_dr_u,
    const Cce::Tags::BoundaryValue<Cce::Tags::Du<Cce::Tags::BondiJ>>::type&
        cce_du_j,
    const Cce::Tags::BoundaryValue<Cce::Tags::Du<Cce::Tags::BondiR>>::type&
        cce_du_r);
