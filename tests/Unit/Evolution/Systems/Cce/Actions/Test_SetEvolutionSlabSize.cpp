// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "Evolution/Systems/Cce/Actions/SetEvolutionSlabSize.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Time/StepControllers/BinaryFraction.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tags = tmpl::list<Tags::TimeStepper<LtsTimeStepper>>;
  using simple_tags =
      tmpl::list<::Tags::TimeStepId, ::Tags::Next<Tags::TimeStepId>,
                 ::Tags::TimeStep, ::Tags::Next<Tags::TimeStep>, ::Tags::Time,
                 Cce::Tags::InitialSlabSize,
                 Cce::Tags::CceEvolutionPrefix<::Tags::StepController>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing,
                             tmpl::list<Cce::Actions::SetEvolutionSlabSize>>>;
};

struct Metavariables {
  static constexpr bool local_time_stepping = false;
  using component_list = tmpl::list<Component<Metavariables>>;
  enum class Phase { Initialization, Testing, Exit };
};

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.Actions.SetEvolutionSlabSize",
                  "[Unit][Cce]") {
  Parallel::register_derived_classes_with_charm<TimeStepper>();
  Parallel::register_classes_with_charm<StepControllers::BinaryFraction>();
  using component = Component<Metavariables>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  MockRuntimeSystem runner{
      {std::make_unique<TimeSteppers::AdamsBashforthN>(3)}};

  const Slab slab(-5.0, -3.0);
  ActionTesting::emplace_component_and_initialize<component>(
      &runner, 0,
      {TimeStepId{true, 1, slab.start()},
       TimeStepId{true, 2, Time{{-3.0, -1.0}, {0, 1}}}, slab.duration(),
       slab.duration(), -5.0, 100.0,
       std::make_unique<StepControllers::BinaryFraction>()});
  ActionTesting::set_phase(make_not_null(&runner),
                           Metavariables::Phase::Testing);
  runner.template next_action<component>(0);
  const auto& box =
      ActionTesting::get_databox<component, typename component::simple_tags>(
          runner, 0);
  CHECK(db::get<Tags::TimeStepId>(box) ==
        TimeStepId{true, 1, Time{{-5.0, 95.0}, {0, 1}}});
  CHECK(db::get<Tags::Next<Tags::TimeStepId>>(box) ==
        TimeStepId{true, 1, Time{{-5.0, 95.0}, {1, 64}}});
  CHECK(db::get<Tags::TimeStep>(box) == TimeDelta{{-5.0, 95.0}, {1, 64}});
  CHECK(db::get<Tags::Next<Tags::TimeStep>>(box) ==
        TimeDelta{{-5.0, 95.0}, {1, 64}});
  CHECK(approx(db::get<Tags::Time>(box)) == -5.0);
}
}  // namespace
