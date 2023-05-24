// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Evolution/Systems/Cce/Actions/CcmActions/InitializeCcmTagsinGH.hpp"
#include "Evolution/Systems/Cce/Actions/CcmActions/Psi0Matching.hpp"
#include "Evolution/Systems/Cce/Actions/CcmActions/ReceivePsi0FromCce.hpp"
#include "Evolution/Systems/Cce/Components/CharacteristicEvolution.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Time/Slab.hpp"
#include "Time/StepChoosers/Constant.hpp"
#include "Time/StepChoosers/ErrorControl.hpp"
#include "Time/StepChoosers/Increase.hpp"
#include "Time/TimeSequence.hpp"
#include "Time/TimeSteppers/Factory.hpp"

namespace {
template <typename Metavariables>
struct mutate_psi0 {
  template <typename ParallelComponent, typename... DbTags, typename ArrayIndex>
  static void apply(
      db::DataBox<tmpl::list<DbTags...>>& box,
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& psi0_cce) {
    db::mutate<Cce::Tags::BoundaryValue<Cce::Tags::Psi0Match>>(
        make_not_null(&box),
        [&psi0_cce](
            gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> psi0) {
          *psi0 = psi0_cce;
        });
  }
};

template <typename Metavariables>
struct mock_gh_evolution {
  using component_being_mocked = typename Metavariables::gh_dg_element_array;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;

  using simple_tags = tmpl::list<::Tags::TimeStepId, domain::Tags::Element<3>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<
              ActionTesting::InitializeDataBox<simple_tags, tmpl::list<>>,
              Cce::Actions::InitializeCcmTags<Metavariables>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Evolve,
          tmpl::list<Cce::Actions::ReceiveCCEData<Metavariables>>>>;
};  // GH component

template <typename Metavariables>
struct mock_characteristic_evolution {
  using component_being_mocked = Cce::CharacteristicEvolution<Metavariables>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;

  using simple_tags = tmpl::list<::Tags::TimeStepId>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<
              ActionTesting::InitializeDataBox<simple_tags, tmpl::list<>>,
              Cce::Actions::InitializeCcmTags<Metavariables>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Evolve,
          tmpl::list<Cce::Actions::TransferPsi0<
              Cce::CharacteristicEvolution<Metavariables>>>>>;
  using const_global_cache_tags =
      Parallel::get_const_global_cache_tags_from_actions<
          phase_dependent_action_list>;
};  // characteristic component

struct test_metavariables {
  static constexpr bool evolve_ccm = true;
  static constexpr bool local_time_stepping = true;

  using gh_dg_element_array =
      DgElementArray<test_metavariables, tmpl::flatten<tmpl::list<>>>;
  using evolved_swsh_tag = Cce::Tags::BondiJ;
  using evolved_swsh_dt_tag = Cce::Tags::BondiH;
  using evolved_coordinates_variables_tag =
      ::Tags::Variables<tmpl::list<Cce::Tags::CauchyCartesianCoords,
                                   Cce::Tags::InertialRetardedTime>>;
  using cce_boundary_communication_tags =
      Cce::Tags::characteristic_worldtube_boundary_tags<
          Cce::Tags::BoundaryValue>;
  using cce_boundary_component = Cce::GhWorldtubeBoundary<test_metavariables>;
  using cce_gauge_boundary_tags = tmpl::flatten<tmpl::list<
      tmpl::transform<
          tmpl::list<Cce::Tags::BondiR, Cce::Tags::DuRDividedByR,
                     Cce::Tags::BondiJ, Cce::Tags::Dr<Cce::Tags::BondiJ>,
                     Cce::Tags::BondiBeta, Cce::Tags::BondiQ, Cce::Tags::BondiU,
                     Cce::Tags::BondiW, Cce::Tags::BondiH>,
          tmpl::bind<Cce::Tags::EvolutionGaugeBoundaryValue, tmpl::_1>>,
      Cce::Tags::BondiUAtScri, Cce::Tags::PartiallyFlatGaugeC,
      Cce::Tags::PartiallyFlatGaugeD, Cce::Tags::PartiallyFlatGaugeOmega,
      Cce::Tags::Du<Cce::Tags::PartiallyFlatGaugeOmega>,
      Spectral::Swsh::Tags::Derivative<Cce::Tags::PartiallyFlatGaugeOmega,
                                       Spectral::Swsh::Tags::Eth>,
      Cce::all_boundary_pre_swsh_derivative_tags_for_scri,
      Cce::all_boundary_swsh_derivative_tags_for_scri>>;

  using scri_values_to_observe =
      tmpl::list<Cce::Tags::News, Cce::Tags::ScriPlus<Cce::Tags::Strain>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi3>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi2>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi1>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi0>,
                 Cce::Tags::Du<Cce::Tags::TimeIntegral<
                     Cce::Tags::ScriPlus<Cce::Tags::Psi4>>>,
                 Cce::Tags::EthInertialRetardedTime>;
  using cce_integrand_tags = tmpl::flatten<tmpl::transform<
      Cce::bondi_hypersurface_step_tags,
      tmpl::bind<Cce::integrand_terms_to_compute_for_bondi_variable,
                 tmpl::_1>>>;
  using ccm_matching_tags =
      tmpl::list<Cce::Tags::BondiJCauchyView, Cce::Tags::Psi0Match,
                 Cce::Tags::Dy<Cce::Tags::Psi0Match>, Cce::Tags::Psi0,
                 Cce::Tags::Dy<Cce::Tags::BondiJCauchyView>,
                 Cce::Tags::Dy<Cce::Tags::Dy<Cce::Tags::BondiJCauchyView>>>;

  using cce_integration_independent_tags = tmpl::conditional_t<
      evolve_ccm,
      tmpl::append<Cce::pre_computation_tags, ccm_matching_tags,
                   tmpl::list<Cce::Tags::DuRDividedByR>>,
      tmpl::push_back<Cce::pre_computation_tags, Cce::Tags::DuRDividedByR>>;
  using cce_temporary_equations_tags = tmpl::remove_duplicates<tmpl::flatten<
      tmpl::transform<cce_integrand_tags,
                      tmpl::bind<Cce::integrand_temporary_tags, tmpl::_1>>>>;
  using cce_pre_swsh_derivatives_tags = Cce::all_pre_swsh_derivative_tags;
  using cce_transform_buffer_tags = Cce::all_transform_buffer_tags;
  using cce_swsh_derivative_tags = Cce::all_swsh_derivative_tags;
  using cce_angular_coordinate_tags =
      tmpl::list<Cce::Tags::CauchyAngularCoords>;
  using cce_scri_tags =
      tmpl::list<Cce::Tags::News, Cce::Tags::ScriPlus<Cce::Tags::Strain>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi3>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi2>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi1>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi0>,
                 Cce::Tags::TimeIntegral<Cce::Tags::ScriPlus<Cce::Tags::Psi4>>,
                 Cce::Tags::EthInertialRetardedTime>;

  using ccm_psi0 = tmpl::list<Cce::Tags::BoundaryValue<Cce::Tags::Psi0Match>>;

  using component_list =
      tmpl::list<mock_gh_evolution<test_metavariables>,
                 mock_characteristic_evolution<test_metavariables>>;

  struct swsh_vars_selector {
    static std::string name() { return "SwshVars"; }
  };

  struct coord_vars_selector {
    static std::string name() { return "CoordVars"; }
  };

  using cce_step_choosers = tmpl::list<
      StepChoosers::Constant<StepChooserUse::LtsStep>,
      StepChoosers::Increase<StepChooserUse::LtsStep>,
      StepChoosers::ErrorControl<StepChooserUse::LtsStep,
                                 Tags::Variables<tmpl::list<evolved_swsh_tag>>,
                                 swsh_vars_selector>,
      StepChoosers::ErrorControl<StepChooserUse::LtsStep,
                                 evolved_coordinates_variables_tag,
                                 coord_vars_selector>>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<LtsTimeStepper, TimeSteppers::lts_time_steppers>,
        tmpl::pair<StepChooser<StepChooserUse::LtsStep>, cce_step_choosers>,
        tmpl::pair<TimeSequence<double>,
                   TimeSequences::all_time_sequences<double>>,
        tmpl::pair<TimeSequence<std::uint64_t>,
                   TimeSequences::all_time_sequences<std::uint64_t>>>;
  };
};  // metavariables

// This unit test valids the one-way communication flow from characteristic to
// generalized harmonic (GH) by calling the action TransferPsi0 and
// ReceiveCCEData separately in them. These two actions are used by CCM. The
// test involves
// (a) Calls TransferPsi0 in characteristic when its data (psi0)
//     is not ready, and ensures GH doesn't receive it.
// (b) Assigns a random value to characteristic's psi0 by mutating its tag
//     Cce::Tags::BoundaryValue<Cce::Tags::Psi0Match> with a simple action
//     mutate_psi0.
// (c) Calls TransferPsi0 again, and checks whether GH receives it.
// (d) Compares GH's received psi0 with the orginal generated one.
SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.Actions.CcmActions.SendPsi0ToGh",
                  "[Unit][Cce]") {
  using characteristic_component =
      mock_characteristic_evolution<test_metavariables>;
  using gh_component = mock_gh_evolution<test_metavariables>;

  MAKE_GENERATOR(gen);
  UniformCustomDistribution<size_t> resolution_distribution{7, 10};
  const size_t l_max = resolution_distribution(gen);

  // CCM uses TimeStepId as communication ID (key).
  const Slab single_step_slab{0.0, 0.0 + 0.1};
  const Time initial_time = single_step_slab.start();
  TimeStepId initial_time_id{true, 0, initial_time};

  // Builds and initializes a parallel runtime system that consists of a mock
  // characteristic component and a mock GH component.
  ActionTesting::MockRuntimeSystem<test_metavariables> runner{
      tuples::tagged_tuple_from_typelist<
          Parallel::get_const_global_cache_tags<test_metavariables>>{l_max}};

  runner.set_phase(Parallel::Phase::Initialization);

  ActionTesting::emplace_component_and_initialize<characteristic_component>(
      &runner, 0, initial_time_id);
  ActionTesting::next_action<characteristic_component>(make_not_null(&runner),
                                                       0);

  const ElementId<3> element_id{0};
  const Element<3> element{element_id, {}};
  ActionTesting::emplace_component_and_initialize<gh_component>(
      &runner, 0, {initial_time_id, element});
  ActionTesting::next_action<gh_component>(make_not_null(&runner), 0);

  // (a) Calls TransferPsi0 in characteristic when its data (psi0) is not ready,
  // and ensures GH doesn't receive it.
  runner.set_phase(Parallel::Phase::Evolve);
  ActionTesting::next_action<characteristic_component>(make_not_null(&runner),
                                                       0);
  REQUIRE_FALSE(ActionTesting::next_action_if_ready<gh_component>(
      make_not_null(&runner), 0));

  // (b) Assigns a random value to characteristic's psi0 by mutating its tag
  // with a simple action mutate_psi0
  UniformCustomDistribution<double> value_distribution{0.1, 1.0};
  const size_t number_of_angular_grid_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  Scalar<SpinWeighted<ComplexDataVector, 2>> generated_psi0{
      number_of_angular_grid_points};
  for (size_t i = 0; i < get(generated_psi0).data().size(); ++i)
    get(generated_psi0).data()[i] = value_distribution(gen);

  ActionTesting::simple_action<characteristic_component,
                               mutate_psi0<test_metavariables>>(
      make_not_null(&runner), 0, generated_psi0);

  // A sanity check: Characteristic's psi0 is indeed what we generated.
  const auto& cce_psi0 = ActionTesting::get_databox_tag<
      characteristic_component, Cce::Tags::BoundaryValue<Cce::Tags::Psi0Match>>(
      runner, 0);

  Approx angular_derivative_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon())
          .scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(cce_psi0, generated_psi0,
                               angular_derivative_approx);

  // (c) Calls TransferPsi0 again, and checks whether GH receives it.
  ActionTesting::invoke_queued_simple_action<characteristic_component>(
      make_not_null(&runner), 0);
  CHECK(ActionTesting::next_action_if_ready<gh_component>(
      make_not_null(&runner), 0));

  // (d) Compares GH's received data with the orginal generated one.
  const auto& psi0_received_by_gh = ActionTesting::get_databox_tag<
      gh_component, Cce::Tags::BoundaryValue<Cce::Tags::Psi0Match>>(runner, 0);

  CHECK_ITERABLE_CUSTOM_APPROX(psi0_received_by_gh, generated_psi0,
                               angular_derivative_approx);
}
}  // namespace
