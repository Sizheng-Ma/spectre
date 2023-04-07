// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/Cce/ReceiveTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/TypeTraits/CreateHasTypeAlias.hpp"

namespace Cce {
namespace Actions {

namespace detail {
CREATE_HAS_TYPE_ALIAS(gh_dg_element_array)
CREATE_HAS_TYPE_ALIAS_V(gh_dg_element_array)
}  // namespace detail

template <typename Metavariables, typename CceComponent>
struct SendPsi0 {
  template <typename ParallelComponent, typename... DbTags, typename ArrayIndex>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/, const TimeStepId& time) {
    if constexpr (detail::has_gh_dg_element_array_v<Metavariables>) {
      Parallel::receive_data<
          Cce::ReceiveTags::BoundaryData<typename Metavariables::ccm_psi0>>(
          Parallel::get_parallel_component<
              typename Metavariables::gh_dg_element_array>(cache),
          time,
          db::get<::Tags::Variables<typename Metavariables::ccm_psi0>>(box),
          false);
    }
  }
};
}  // namespace Actions
}  // namespace Cce
