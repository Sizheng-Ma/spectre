#pragma once

#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "hello_world/hello_world_test.hpp"

namespace Cce {
namespace Actions {

struct MyCCMAction {
  using const_global_cache_tags =
      tmpl::list<Tags::LMax, Tags::NumberOfRadialPoints,
                 InitializationTags::ExtractionRadius>;
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    auto l_max = db::get<Tags::LMax>(box);
    auto number_of_radial_points = db::get<Tags::NumberOfRadialPoints>(box);
    auto radius = db::get<InitializationTags::ExtractionRadius>(box);

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

    ccm_functions11(re_h, im_h, l_max, number_of_radial_points, radius, re_j,
                    im_j, cauchy_cart_std, inertial_cart_std);

    auto bondi_h = db::get<Tags::BondiH>(box);
    std::cout << re_h.at(0) << "true " << get(bondi_h).data()[0] << std::endl;
    return {Parallel ::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
}  // namespace Cce
