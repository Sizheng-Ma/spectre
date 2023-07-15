#pragma once

#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "hello_world/hello_world_test.hpp"

namespace Cce {
namespace Actions {

template <typename Boundary>
struct MyCCMAction {
  using const_global_cache_tags =
      tmpl::list<Tags::LMax, Tags::NumberOfRadialPoints>;
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    auto l_max = db::get<Tags::LMax>(box);
    auto number_of_radial_points = db::get<Tags::NumberOfRadialPoints>(box);

    double radius;
    if constexpr (tt::is_a_v<AnalyticWorldtubeBoundary, Boundary>) {
      radius = db::get<Tags::AnalyticBoundaryDataManager>(box)
                   .get_extraction_radius();
    } else if (tt::is_a_v<H5WorldtubeBoundary, Boundary>) {
      // TODO This is hardcoded.
      radius = 267.;
    }

    auto bondi_j = db::get<Tags::BondiJ>(box);
    auto cauchy_cart = db::get<Tags::CauchyCartesianCoords>(box);

    const size_t boundary_size =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    std::vector<double> test(boundary_size, 1.);
    std::vector<double> mtest(boundary_size, -1.);
    std::vector<double> zero(boundary_size, 0);

    std::vector<std::vector<double>> pi{zero, zero, zero, zero, zero,
                                        zero, zero, zero, zero, zero};
    std::vector<std::vector<double>> spacetime_metric{
        mtest, zero, zero, zero, test, zero, zero, test, zero, test};
    std::vector<std::vector<std::vector<double>>> phi{pi, pi, pi};

    std::vector<double> re_j;
    std::vector<double> im_j;

    for (size_t iii = 0; iii < get(bondi_j).data().size(); iii++) {
      re_j.push_back(real(get(bondi_j).data())[iii]);
    }
    for (size_t iii = 0; iii < get(bondi_j).data().size(); iii++) {
      im_j.push_back(imag(get(bondi_j).data())[iii]);
    }

    std::vector<double> cauchy_cartx;
    std::vector<double> cauchy_carty;
    std::vector<double> cauchy_cartz;

    for (size_t iii = 0; iii < boundary_size; iii++) {
      cauchy_cartx.push_back(get<0>(cauchy_cart).data()[iii]);
    }
    for (size_t iii = 0; iii < boundary_size; iii++) {
      cauchy_carty.push_back(get<1>(cauchy_cart).data()[iii]);
    }
    for (size_t iii = 0; iii < boundary_size; iii++) {
      cauchy_cartz.push_back(get<2>(cauchy_cart).data()[iii]);
    }

    std::vector<std::vector<double>> cauchy_cart_std{cauchy_cartx, cauchy_carty,
                                                     cauchy_cartz};
    std::vector<std::vector<double>> inertial_cart_std{
        cauchy_cartx, cauchy_carty, cauchy_cartz};

    std::vector<double> re_h;
    std::vector<double> im_h;

    auto& bondi_beta_bdry =
        db::get<Cce::Tags::BoundaryValue<Cce::Tags::BondiBeta>>(box);

    auto& dr_j_bdry =
        db::get<Cce::Tags::BoundaryValue<Cce::Tags::Dr<Cce::Tags::BondiJ>>>(
            box);

    auto& du_r_bdry =
        db::get<Cce::Tags::BoundaryValue<Cce::Tags::Du<Cce::Tags::BondiR>>>(
            box);

    auto& bondi_h_bdry =
        db::get<Cce::Tags::BoundaryValue<Cce::Tags::BondiH>>(box);

    auto& bondi_j_bdry =
        db::get<Cce::Tags::BoundaryValue<Cce::Tags::BondiJ>>(box);

    auto& bondi_q_bdry =
        db::get<Cce::Tags::BoundaryValue<Cce::Tags::BondiQ>>(box);

    auto& bondi_r_bdry =
        db::get<Cce::Tags::BoundaryValue<Cce::Tags::BondiR>>(box);

    auto& bondi_u_bdry =
        db::get<Cce::Tags::BoundaryValue<Cce::Tags::BondiU>>(box);

    auto& bondi_w_bdry =
        db::get<Cce::Tags::BoundaryValue<Cce::Tags::BondiW>>(box);

    auto& bondi_dr_u_bdry =
        db::get<Cce::Tags::BoundaryValue<Cce::Tags::Dr<Cce::Tags::BondiU>>>(
            box);

    auto& bondi_du_j_bdry =
        db::get<Cce::Tags::BoundaryValue<Cce::Tags::Du<Cce::Tags::BondiJ>>>(
            box);

    auto& bondi_du_r_bdry_DuRDividedByR =
        db::get<Tags::BoundaryValue<Tags::DuRDividedByR>>(box);

    // db::get<::Tags::Variables<typename
    // Metavariables::cce_boundary_communication_tags>>(box);

    std::vector<double> dt_cauchy_x, dt_cauchy_y, dt_cauchy_z, dt_u_scri;

    ccm_functions11(
        re_h, im_h, dt_cauchy_x, dt_cauchy_y, dt_cauchy_z, dt_u_scri, l_max,
        number_of_radial_points, radius, re_j, im_j, cauchy_cart_std,
        inertial_cart_std, bondi_beta_bdry, dr_j_bdry, du_r_bdry, bondi_h_bdry,
        bondi_j_bdry, bondi_q_bdry, bondi_r_bdry, bondi_u_bdry, bondi_w_bdry,
        bondi_dr_u_bdry, bondi_du_j_bdry, bondi_du_r_bdry_DuRDividedByR);

    auto bondi_h = db::get<Tags::BondiH>(box);

    double resres = 0;
    for (size_t iii = 0; iii < re_h.size(); iii++) {
      resres += pow(im_h.at(iii) - imag(get(bondi_h).data())[iii], 2);
      resres += pow(re_h.at(iii) - real(get(bondi_h).data())[iii], 2);
    }

    resres /= re_h.size();
    std::cout << std::setprecision(30) << sqrt(resres) << " ";

    auto& dt_cauchy_cart =
        db::get<::Tags::dt<Cce::Tags::CauchyCartesianCoords>>(box);

    double resres_cauchy = 0;
    for (size_t iii = 0; iii < get<0>(dt_cauchy_cart).size(); iii++) {
      resres_cauchy +=
          pow(dt_cauchy_x.at(iii) - get<0>(dt_cauchy_cart)[iii], 2);
      resres_cauchy +=
          pow(dt_cauchy_y.at(iii) - get<1>(dt_cauchy_cart)[iii], 2);
      resres_cauchy +=
          pow(dt_cauchy_z.at(iii) - get<2>(dt_cauchy_cart)[iii], 2);
    }
    resres_cauchy /= get<0>(dt_cauchy_cart).size();
    std::cout << std::setprecision(30) << sqrt(resres_cauchy) << " ";

    auto& du_t = get<::Tags::dt<Cce::Tags::InertialRetardedTime>>(box);
    double resres_du_t = 0;
    for (size_t iii = 0; iii < dt_u_scri.size(); iii++) {
      resres_du_t += pow(dt_u_scri.at(iii) - du_t.get()[iii], 2);
    }

    resres_du_t /= dt_u_scri.size();
    std::cout << std::setprecision(30) << sqrt(resres_du_t) << std::endl;
    std::cout << std::endl;
    // BondiBeta
    // std::cout << std::setprecision(30) << im_h.at(41) << " true "
    //           << imag(get(bondi_h).data())[41] << std::endl;
    // auto& dt_cauchy_cart = db::get<Cce::Tags::DuRDividedByR>(box);
    // std::cout << "my beta " << std::setprecision(30)
    //           << get(dt_cauchy_cart).data()[0] << std::endl;
    return {Parallel ::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
}  // namespace Cce
