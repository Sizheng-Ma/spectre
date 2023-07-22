#pragma once

#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "hello_world/hello_world.hpp"

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

    double radius = 20.;
    if constexpr (tt::is_a_v<AnalyticWorldtubeBoundary, Boundary>) {
      radius = db::get<Tags::AnalyticBoundaryDataManager>(box)
                   .get_extraction_radius();
    } else if (tt::is_a_v<H5WorldtubeBoundary, Boundary>) {
      // TODO This is hardcoded.
      radius = 267.;
    }

    auto bondi_j = db::get<Tags::BondiJ>(box);
    auto cauchy_cart = db::get<Tags::CauchyCartesianCoords>(box);

    auto spectre_inertial_retarded_time =
        db::get<Tags::InertialRetardedTime>(box);

    const size_t boundary_size =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    // std::vector<double> test(boundary_size, 1.);
    // std::vector<double> mtest(boundary_size, -1.);
    // std::vector<double> zero(boundary_size, 0);

    std::vector<std::vector<double>> pi;
    std::vector<std::vector<double>> spacetime_metric;
    std::vector<std::vector<std::vector<double>>> phi;

    std::vector<double> re_j;
    std::vector<double> spectre_inertial_retarded_time_std;

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
    for (size_t iii = 0; iii < boundary_size; iii++) {
      spectre_inertial_retarded_time_std.push_back(
          get(spectre_inertial_retarded_time)[iii]);
    }

    std::vector<std::vector<double>> cauchy_cart_std{cauchy_cartx, cauchy_carty,
                                                     cauchy_cartz};
    std::vector<std::vector<double>> inertial_cart_std{
        cauchy_cartx, cauchy_carty, cauchy_cartz};

    std::vector<double> re_h;
    std::vector<double> im_h;

    std::vector<std::complex<double>> eth_inertial_retarded_time;
    std::vector<std::complex<double>> news;
    std::vector<std::complex<double>> strain;
    std::vector<std::complex<double>> psi0;
    std::vector<std::complex<double>> psi1;
    std::vector<std::complex<double>> psi2;
    std::vector<std::complex<double>> psi3;
    std::vector<std::complex<double>> psi4;

    auto& my_space_time = db::get<Cce::Tags::TestSpaceTimeMetric>(box);
    auto& my_phi = db::get<Cce::Tags::TestPhi>(box);
    auto& my_pi = db::get<Cce::Tags::TestPi>(box);

    ThisThisDataVector_to_std_vector(my_space_time, spacetime_metric);
    ThisThisDataVector_to_std_vector(my_pi, pi);
    ThisThisDataVector_to_tri_std_vector(my_phi, phi);

    // auto& bondi_beta_bdry =
    //     db::get<Cce::Tags::BoundaryValue<Cce::Tags::BondiBeta>>(box);

    // auto& dr_j_bdry =
    //     db::get<Cce::Tags::BoundaryValue<Cce::Tags::Dr<Cce::Tags::BondiJ>>>(
    //         box);

    // auto& du_r_bdry =
    //     db::get<Cce::Tags::BoundaryValue<Cce::Tags::Du<Cce::Tags::BondiR>>>(
    //         box);

    // auto& bondi_h_bdry =
    //     db::get<Cce::Tags::BoundaryValue<Cce::Tags::BondiH>>(box);

    // auto& bondi_j_bdry =
    //     db::get<Cce::Tags::BoundaryValue<Cce::Tags::BondiJ>>(box);

    // auto& bondi_q_bdry =
    //     db::get<Cce::Tags::BoundaryValue<Cce::Tags::BondiQ>>(box);

    // auto& bondi_r_bdry =
    //     db::get<Cce::Tags::BoundaryValue<Cce::Tags::BondiR>>(box);

    // auto& bondi_u_bdry =
    //     db::get<Cce::Tags::BoundaryValue<Cce::Tags::BondiU>>(box);

    // auto& bondi_w_bdry =
    //     db::get<Cce::Tags::BoundaryValue<Cce::Tags::BondiW>>(box);

    // auto& bondi_dr_u_bdry =
    //     db::get<Cce::Tags::BoundaryValue<Cce::Tags::Dr<Cce::Tags::BondiU>>>(
    //         box);

    // auto& bondi_du_j_bdry =
    //     db::get<Cce::Tags::BoundaryValue<Cce::Tags::Du<Cce::Tags::BondiJ>>>(
    //         box);

    // auto& bondi_du_r_bdry_DuRDividedByR =
    //     db::get<Tags::BoundaryValue<Tags::DuRDividedByR>>(box);

    // db::get<::Tags::Variables<typename
    // Metavariables::cce_boundary_communication_tags>>(box);

    std::vector<double> dt_cauchy_x, dt_cauchy_y, dt_cauchy_z, dt_u_scri;
    std::vector<double> dt_inertial_x, dt_inertial_y, dt_inertial_z;

    // ccm_functions11(
    //     re_h, im_h, dt_cauchy_x, dt_cauchy_y, dt_cauchy_z, dt_u_scri,
    //     l_max, number_of_radial_points, radius, re_j, im_j,
    //     cauchy_cart_std, inertial_cart_std, bondi_beta_bdry, dr_j_bdry,
    //     du_r_bdry, bondi_h_bdry, bondi_j_bdry, bondi_q_bdry, bondi_r_bdry,
    //     bondi_u_bdry, bondi_w_bdry, bondi_dr_u_bdry, bondi_du_j_bdry,
    //     bondi_du_r_bdry_DuRDividedByR);

    ccm_functions(
        re_h, im_h, dt_cauchy_x, dt_cauchy_y, dt_cauchy_z, dt_inertial_x,
        dt_inertial_y, dt_inertial_z, eth_inertial_retarded_time, news, strain,
        psi0, psi1, psi2, psi3, psi4, dt_u_scri, l_max, number_of_radial_points,
        spacetime_metric, pi, phi, radius, re_j, im_j, cauchy_cart_std,
        inertial_cart_std, spectre_inertial_retarded_time_std);

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
  static void ThisThisDataVector_to_std_vector(
      const tnsr::aa<DataVector, 3>& pi,
      std::vector<std::vector<double>>& data) {
    const auto size = get<0, 0>(pi).size();
    std::vector<double> datattt(size);
    std::vector<double> datattx(size);
    std::vector<double> datatty(size);
    std::vector<double> datattz(size);
    std::vector<double> datatxx(size);
    std::vector<double> datatxy(size);
    std::vector<double> datatxz(size);
    std::vector<double> datatyy(size);
    std::vector<double> datatyz(size);
    std::vector<double> datatzz(size);
    for (unsigned int i = 0; i < size; i++) {
      datattt[i] = get<0, 0>(pi)[i];
      datattx[i] = get<0, 1>(pi)[i];
      datatty[i] = get<0, 2>(pi)[i];
      datattz[i] = get<0, 3>(pi)[i];
      datatxx[i] = get<1, 1>(pi)[i];
      datatxy[i] = get<1, 2>(pi)[i];
      datatxz[i] = get<1, 3>(pi)[i];
      datatyy[i] = get<2, 2>(pi)[i];
      datatyz[i] = get<2, 3>(pi)[i];
      datatzz[i] = get<3, 3>(pi)[i];
    }
    data.push_back(datattt);
    data.push_back(datattx);
    data.push_back(datatty);
    data.push_back(datattz);
    data.push_back(datatxx);
    data.push_back(datatxy);
    data.push_back(datatxz);
    data.push_back(datatyy);
    data.push_back(datatyz);
    data.push_back(datatzz);
  }

  static void ThisThisDataVector_to_tri_std_vector(
      const tnsr::iaa<DataVector, 3>& pi,
      std::vector<std::vector<std::vector<double>>>& data) {
    const auto size = pi.get(0, 0, 0).size();
    std::vector<double> datattt(size);
    std::vector<double> datattx(size);
    std::vector<double> datatty(size);
    std::vector<double> datattz(size);
    std::vector<double> datatxx(size);
    std::vector<double> datatxy(size);
    std::vector<double> datatxz(size);
    std::vector<double> datatyy(size);
    std::vector<double> datatyz(size);
    std::vector<double> datatzz(size);
    for (size_t ijj = 0; ijj < 3; ++ijj) {
      for (size_t i = 0; i < size; i++) {
        datattt[i] = pi.get(ijj, 0, 0)[i];
        datattx[i] = pi.get(ijj, 0, 1)[i];
        datatty[i] = pi.get(ijj, 0, 2)[i];
        datattz[i] = pi.get(ijj, 0, 3)[i];
        datatxx[i] = pi.get(ijj, 1, 1)[i];
        datatxy[i] = pi.get(ijj, 1, 2)[i];
        datatxz[i] = pi.get(ijj, 1, 3)[i];
        datatyy[i] = pi.get(ijj, 2, 2)[i];
        datatyz[i] = pi.get(ijj, 2, 3)[i];
        datatzz[i] = pi.get(ijj, 3, 3)[i];
      }
      std::vector<std::vector<double>> datafinal{
          datattt, datattx, datatty, datattz, datatxx,
          datatxy, datatxz, datatyy, datatyz, datatzz};
      data.push_back(datafinal);
    }
  }
};
}  // namespace Actions
}  // namespace Cce
