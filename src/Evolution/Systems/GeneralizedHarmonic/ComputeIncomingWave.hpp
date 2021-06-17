// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/CCMHelper.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"

namespace GeneralizedHarmonic {
namespace Actions {

template <size_t Dim, typename Frame>
struct CalculateWij {
  using const_global_cache_tags = tmpl::list<Cce::Tags::LMax>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    db::mutate_apply<AngularTetradForCCM<Dim, Frame>>(make_not_null(&box));
    db::mutate_apply<InterpolatePsi0<Dim, Frame>>(make_not_null(&box));
    db::mutate_apply<IncomingWFromCCE<Dim, Frame>>(make_not_null(&box));
    return std::forward_as_tuple(std::move(box));
  }
};

}  // namespace Actions
}  // namespace GeneralizedHarmonic
