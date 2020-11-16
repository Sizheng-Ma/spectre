// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Systems/Cce/Actions/SendPsi0ToEvolution.hpp"
#include "Evolution/Systems/Cce/NewmanPenrose.hpp"
#include "Evolution/Systems/Cce/PreSwshDerivatives.hpp"

namespace GeneralizedHarmonic {
namespace Actions {

template <typename Metavariables>
struct ReceiveCCEData {
  using inbox_tags = tmpl::list<
      Cce::ReceiveTags::BoundaryData<typename Metavariables::ccm_psi0>>;
  template <
      typename DbTags, typename... InboxTags, typename ArrayIndex,
      typename ActionList, typename ParallelComponent,
      Requires<tmpl::list_contains_v<tmpl::list<InboxTags...>,
                                     Cce::ReceiveTags::BoundaryData<
                                         typename Metavariables::ccm_psi0>> and
               tmpl::list_contains_v<DbTags, ::Tags::TimeStepId>> = nullptr>
      static std::tuple<db::DataBox<DbTags>&&,
                        Parallel::AlgorithmExecution, size_t>
      // Is this BoundaryData correct?
      apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& inboxes,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    auto& inbox = tuples::get<
        Cce::ReceiveTags::BoundaryData<typename Metavariables::ccm_psi0>>(
        inboxes);
    if (inbox.count(db::get<::Tags::TimeStepId>(box)) != 1) {
      return {std::move(box), Parallel::AlgorithmExecution::Pause,
              tmpl::index_of<ActionList, ReceiveCCEData>::value};
    }
    tmpl::for_each<typename Metavariables::ccm_psi0>(
        [&inbox, &box](auto tag_v) noexcept {
          using tag = typename decltype(tag_v)::type;
          db::mutate<tag>(
              make_not_null(&box),
              [&inbox](const gsl::not_null<typename tag::type*> destination,
                       const TimeStepId& time) noexcept {
                *destination = get<tag>(inbox[time]);
              },
              db::get<::Tags::TimeStepId>(box));
        });
    inbox.erase(db::get<::Tags::TimeStepId>(box));
    return {std::move(box), Parallel::AlgorithmExecution::Continue,
            tmpl::index_of<ActionList, ReceiveCCEData>::value + 1};

      }
  template <
      typename DbTags, typename... InboxTags, typename ArrayIndex,
      typename ActionList, typename ParallelComponent,
      Requires<
          not tmpl::list_contains_v<tmpl::list<InboxTags...>,
                                    Cce::ReceiveTags::BoundaryData<
                                        typename Metavariables::ccm_psi0>> or
          not tmpl::list_contains_v<DbTags, ::Tags::TimeStepId>> = nullptr>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    ERROR(
        "Required tags not present in the inbox or databox to transfer the "
        "Psi0 data");
    return std::forward_as_tuple(std::move(box));
  }
};

}  // namespace Actions
}  // namespace GeneralizedHarmonic
