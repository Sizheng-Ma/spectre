// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Rational.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Cce {
namespace Actions {
/// \brief Sets the CCE slab size according to the CCE-specific option following
/// the self-start procedure.
///
/// \details This is a simple version of a one-time slab choice, and primarily
/// required in the case of a coupled GH system that cannot perform local time
/// stepping. In that case, to keep CCE steps reasonably large, we need to first
/// match steps during self-start, then reset the CCE slab to be larger to
/// accomodate larger steps.
/// This feature could also be implemented using slab choosers, but the
/// reduction process needed by that control-flow is more difficult to apply to
/// CCE singletons.
///
/// \ref DataBoxGroup changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - `Tags::TimeStepId`
///   - `Tags::Next<Tags::TimeStepId>`
///   - `Tags::TimeStep`
///   - `Tags::Next<Tags::TimeStep>`
///   - `::Tags::Time`
struct SetEvolutionSlabSize {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const auto current_time_id = db::get<::Tags::TimeStepId>(box);
    if (LIKELY(current_time_id.slab_number() != 1 or
               current_time_id.substep() != 0)) {
      return std::make_tuple(std::move(box));
    }
    // do not reset the slab size unless we are forced into a small slab size
    // because the DG system is using global time stepping.
    if constexpr (not Metavariables::local_time_stepping) {
      db::mutate<::Tags::TimeStepId, ::Tags::Next<::Tags::TimeStepId>,
                 ::Tags::TimeStep, ::Tags::Next<::Tags::TimeStep>,
                 ::Tags::Time>(
          make_not_null(&box),
          [](const gsl::not_null<TimeStepId*> time_step_id,
             const gsl::not_null<TimeStepId*> next_time_step_id,
             const gsl::not_null<TimeDelta*> time_step,
             const gsl::not_null<TimeDelta*> next_time_step,
             const gsl::not_null<double*> time,
             const double& cce_initial_slab_size, const auto& time_stepper,
             const auto& step_controller) {
            const Slab step_slab{
                time_step_id->substep_time().value(),
                time_step_id->substep_time().value() + cce_initial_slab_size};
            const Time initial_time = Time{step_slab, {0, 1}};
            *time_step_id =
                TimeStepId{true, time_step_id->slab_number(), initial_time};
            *time_step =
                step_controller.choose_step(initial_time, time_step->value());
            *next_time_step = *time_step;
            *next_time_step_id =
                time_stepper.next_time_id(*time_step_id, *time_step);
            *time = time_step_id->substep_time().value();
          },

          db::get<Tags::InitialSlabSize>(box),
          db::get<::Tags::TimeStepper<>>(box),
          db::get<Tags::CceEvolutionPrefix<::Tags::StepController>>(box));
    }
    return std::make_tuple(std::move(box));
  }
};

}  // namespace Actions
}  // namespace Cce
