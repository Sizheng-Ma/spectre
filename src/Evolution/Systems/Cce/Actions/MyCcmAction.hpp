#pragma once

#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "hello_world/hello_world.hpp"

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
    // ccm_functions(std::vector<double>& re_h, std::vector<double>& im_h,
    //            std::vector<double>& dt_cauchy_x,
    //            std::vector<double>& dt_cauchy_y,
    //            std::vector<double>& dt_cauchy_z,
    //            std::vector<double>& dt_inertial_x,
    //            std::vector<double>& dt_inertial_y,
    //            std::vector<double>& dt_inertial_z,
    //            std::vector<double>& re_psi3, std::vector<double>& im_psi3,
    //            std::vector<double>& dt_u_scri, const size_t l_max,
    //            const size_t number_of_radial_points,
    //            const std::vector<std::vector<double>>& spacetime_metric,
    //            const std::vector<std::vector<double>>& pi,
    //            const std::vector<std::vector<std::vector<double>>>& phi,
    //            const double radius, const std::vector<double>& re_j,
    //            const std::vector<double>& im_j,
    //            const std::vector<std::vector<double>>& cauchy_cart,
    //            const std::vector<std::vector<double>>& inertial_cart);
    return {Parallel ::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
}  // namespace Cce
