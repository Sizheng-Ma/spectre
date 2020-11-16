// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/Cce/ReceiveTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/TypeTraits/CreateHasTypeAlias.hpp"
#include "Time/TimeStepId.hpp"

namespace Cce {
namespace Actions {

namespace detail {
  CREATE_HAS_TYPE_ALIAS(gh_dg_element_array)
  CREATE_HAS_TYPE_ALIAS_V(gh_dg_element_array)
}

template <typename Metavariables, typename CceComponent>
struct SendPsi0 {
  // TODO Require...
  template <
      typename ParallelComponent, typename... DbTags, typename ArrayIndex,
      Requires<tmpl2::flat_any_v<std::is_same_v<
          ::Tags::Variables<typename Metavariables::ccm_psi0>, DbTags>...>> =
          nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const TimeStepId& time) noexcept {
    if constexpr (detail::has_gh_dg_element_array_v<Metavariables>) {
      Parallel::receive_data<
          Cce::ReceiveTags::BoundaryData<typename Metavariables::ccm_psi0>>(
          Parallel::get_parallel_component<
              typename Metavariables::gh_dg_element_array>(cache),
          time,
          db::get<::Tags::Variables<typename Metavariables::ccm_psi0>>(box),
          true);
    }
  }
};
}  // namespace Actions
}  // namespace Cce
