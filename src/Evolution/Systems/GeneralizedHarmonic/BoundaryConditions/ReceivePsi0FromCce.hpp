
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
//TODO add comment later
//TODO change later
template <typename Metavariables>
struct ReceiveCCEData {
  // TODO typename or not?
  using inbox_tags = tmpl::list<
      Cce::ReceiveTags::BoundaryData<typename Metavariables::ccm_psi0>>;

  template <
      typename DbTags, typename... InboxTags, typename ArrayIndex,
      typename ActionList, typename ParallelComponent,
      Requires<tmpl::list_contains_v<tmpl::list<InboxTags...>,
                                     Cce::ReceiveTags::BoundaryData<
                                         typename Metavariables::ccm_psi0>> and
               tmpl::list_contains_v<DbTags, ::Tags::TimeStepId>> = nullptr>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& inboxes,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    auto& inbox = tuples::get<
        Cce::ReceiveTags::BoundaryData<typename Metavariables::ccm_psi0>>(
        inboxes);
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
    return std::forward_as_tuple(std::move(box));
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

  template <
      typename DbTags, typename... InboxTags, typename ArrayIndex,
      Requires<tmpl::list_contains_v<tmpl::list<InboxTags...>,
                                     Cce::ReceiveTags::BoundaryData<
                                         typename Metavariables::ccm_psi0>> and
               tmpl::list_contains_v<DbTags, ::Tags::TimeStepId>> = nullptr>
  static bool is_ready(const db::DataBox<DbTags>& box,
                       const tuples::TaggedTuple<InboxTags...>& inboxes,
                       const Parallel::GlobalCache<Metavariables>& /*cache*/,
                       const ArrayIndex& /*array_index*/) noexcept {
    return tuples::get<Cce::ReceiveTags::BoundaryData<
               typename Metavariables::ccm_psi0>>(
               inboxes)
               .count(db::get<::Tags::TimeStepId>(box)) == 1;
  }

  template <
      typename DbTags, typename... InboxTags, typename ArrayIndex,
      Requires<
          not tmpl::list_contains_v<tmpl::list<InboxTags...>,
                                    Cce::ReceiveTags::BoundaryData<
                                        typename Metavariables::ccm_psi0>> or
          not tmpl::list_contains_v<DbTags, ::Tags::TimeStepId>> = nullptr>
  static bool is_ready(const db::DataBox<DbTags>& /*box*/,
                       const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                       const Parallel::GlobalCache<Metavariables>& /*cache*/,
                       const ArrayIndex& /*array_index*/) noexcept {
    return false;
  }
};

}  // namespace Actions
}  // namespace GeneralizedHarmonic
