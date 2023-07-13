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
        cce_bondi_dr_j);
