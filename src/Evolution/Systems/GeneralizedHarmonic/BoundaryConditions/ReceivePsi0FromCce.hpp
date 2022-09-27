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
  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    auto& inbox = tuples::get<
        Cce::ReceiveTags::BoundaryData<typename Metavariables::ccm_psi0>>(
        inboxes);
    const auto& element = db::get<domain::Tags::Element<3_st>>(box);
    if (element.external_boundaries().size() == 0_st) {
      inbox.clear();
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }
    if (inbox.count(db::get<::Tags::TimeStepId>(box)) != 1) {
      return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    }
    tmpl::for_each<typename Metavariables::ccm_psi0>(
        [&inbox, &box](auto tag_v) {
          using tag = typename decltype(tag_v)::type;
          db::mutate<tag>(
              make_not_null(&box),
              [&inbox](const gsl::not_null<typename tag::type*> destination,
                       const TimeStepId& time) {
                *destination = get<tag>(inbox[time]);
              },
              db::get<::Tags::TimeStepId>(box));
        });
    inbox.erase(db::get<::Tags::TimeStepId>(box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

}  // namespace Actions
}  // namespace GeneralizedHarmonic
