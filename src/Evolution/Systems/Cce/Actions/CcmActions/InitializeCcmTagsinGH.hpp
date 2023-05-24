// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/ReceiveTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"

namespace Cce {
namespace Actions {

/*!
 * \ingroup ActionsGroup
 * \brief Initializes a
 *
 * \details Databox changes:
 * - Adds:
 *   - `Tags::Variables<typename Metavariables::ccm_tags>`
 * - Removes: nothing
 * - Modifies: nothing
 */
template <typename Metavariables>
struct InitializeCcmTags {
  using ccm_tag = ::Tags::Variables<typename Metavariables::ccm_psi0>;
  using simple_tags = tmpl::list<ccm_tag>;
  using const_global_cache_tags = tmpl::list<Cce::Tags::LMax>;
  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const size_t l_max = db::get<Spectral::Swsh::Tags::LMaxBase>(box);
    const size_t number_of_angular_grid_points =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    Initialization::mutate_assign<simple_tags>(
        make_not_null(&box),
        typename ccm_tag::type{number_of_angular_grid_points, 0.0});
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
}  // namespace Cce
