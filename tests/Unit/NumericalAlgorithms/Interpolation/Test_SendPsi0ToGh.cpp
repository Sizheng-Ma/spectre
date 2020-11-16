// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>

#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Systems/Cce/Actions/BoundaryComputeAndSendToEvolution.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeCharacteristicEvolutionScri.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeCharacteristicEvolutionTime.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeCharacteristicEvolutionVariables.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeWorldtubeBoundary.hpp"
#include "Evolution/Systems/Cce/Actions/ReceiveGhWorldtubeData.hpp"
#include "Evolution/Systems/Cce/Actions/ReceiveWorldtubeData.hpp"
#include "Evolution/Systems/Cce/Actions/RequestBoundaryData.hpp"
#include "Evolution/Systems/Cce/Actions/Psi0Matching.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/Components/CharacteristicEvolution.hpp"
#include "Evolution/Systems/Cce/Components/WorldtubeBoundary.hpp"
#include "Evolution/Systems/Cce/IntegrandInputSteps.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhLockstep.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/Systems/Cce/BoundaryTestHelpers.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
#include "NumericalAlgorithms/Spectral/SwshCoefficients.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/RungeKutta3.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/ReceivePsi0FromCce.hpp"
namespace Cce {
namespace {
template <typename Metavariables>
struct test_action {
  template <typename ParallelComponent, typename... DbTags, typename ArrayIndex>
  static void apply(
      db::DataBox<tmpl::list<DbTags...>>& box,
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& psi0_cce) noexcept {
    if constexpr (tmpl::list_contains_v<
                      tmpl::list<DbTags...>,
                      Cce::Tags::BoundaryValue<Cce::Tags::Psi0Match>>) {
      db::mutate<Cce::Tags::BoundaryValue<Cce::Tags::Psi0Match>>(
          make_not_null(&box),
          [&psi0_cce](gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
                          psi0) noexcept { *psi0 = psi0_cce; });
    } else {
      ERROR("Required tag: BoundaryValue<Psi0Match> not present");
    }
  }
};

template <typename Metavariables>
struct mock_gh_evolution {
  using component_being_mocked = typename Metavariables::gh_dg_element_array;
  using replace_these_simple_actions = tmpl::list<>;
  using with_these_simple_actions = tmpl::list<>;

  using initialize_action_list = tmpl::list<
      ::Actions::SetupDataBox,
      Initialization::Actions::InitializeCcmTags<Metavariables>
      >;
  using initialization_tags =
      Parallel::get_initialization_tags<initialize_action_list>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;

  using simple_tags = tmpl::list<
        ::Tags::TimeStepId
        >;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<
              ActionTesting::InitializeDataBox<simple_tags, tmpl::list<>>,
              ::Actions::SetupDataBox,
              Initialization::Actions::InitializeCcmTags<Metavariables>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Evolve,
          tmpl::list<
              GeneralizedHarmonic::Actions::ReceiveCCEData<Metavariables>>>>;
  using const_global_cache_tags =
      Parallel::get_const_global_cache_tags_from_actions<
          phase_dependent_action_list>;
};

template <typename Metavariables>
struct mock_characteristic_evolution {
  using component_being_mocked = CharacteristicEvolution<Metavariables>;
  using replace_these_simple_actions = tmpl::list<>;
  using with_these_simple_actions = tmpl::list<>;

  using initialize_action_list = tmpl::list<
             ::Actions::SetupDataBox,
             Initialization::Actions::InitializeCcmTags<Metavariables>
      >;
  using initialization_tags =
      Parallel::get_initialization_tags<initialize_action_list>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;

//FIXME db::AddSimpleTags ?
  using simple_tags = tmpl::list<::Tags::TimeStepId>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Initialization,
      tmpl::list<
             ActionTesting::InitializeDataBox<simple_tags, tmpl::list<>>,
             ::Actions::SetupDataBox,
             Initialization::Actions::InitializeCcmTags<Metavariables>
                >>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Evolve,
          tmpl::list<
                     Actions::TransferPsi0<CharacteristicEvolution
                                           <Metavariables>>
                >>>;
  using const_global_cache_tags =
      Parallel::get_const_global_cache_tags_from_actions<
          phase_dependent_action_list>;
};

struct test_metavariables {
  using gh_dg_element_array = DgElementArray<test_metavariables,tmpl::flatten<
                                tmpl::list<>>>;
  using evolved_swsh_tag = Tags::BondiJ;
  using evolved_swsh_dt_tag = Tags::BondiH;
  using evolved_coordinates_variables_tag = ::Tags::Variables<
      tmpl::list<Tags::CauchyCartesianCoords, Tags::InertialRetardedTime>>;
  using cce_boundary_communication_tags =
      Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>;
  using cce_boundary_component = GhWorldtubeBoundary<test_metavariables>;
  using cce_gauge_boundary_tags = tmpl::flatten<tmpl::list<
      tmpl::transform<
          tmpl::list<Tags::BondiR, Tags::DuRDividedByR, Tags::BondiJ,
                     Tags::Dr<Tags::BondiJ>, Tags::BondiBeta, Tags::BondiQ,
                     Tags::BondiU, Tags::BondiW, Tags::BondiH>,
          tmpl::bind<Tags::EvolutionGaugeBoundaryValue, tmpl::_1>>,
      Tags::BondiUAtScri, Tags::GaugeC, Tags::GaugeD, Tags::GaugeOmega,
      Tags::Du<Tags::GaugeOmega>,
      Spectral::Swsh::Tags::Derivative<Tags::GaugeOmega,
                                       Spectral::Swsh::Tags::Eth>>>;

  using scri_values_to_observe = tmpl::list<>;
  using cce_integrand_tags = tmpl::flatten<tmpl::transform<
      bondi_hypersurface_step_tags,
      tmpl::bind<integrand_terms_to_compute_for_bondi_variable, tmpl::_1>>>;
  using cce_integration_independent_tags = pre_computation_tags;
  using cce_temporary_equations_tags =
      tmpl::remove_duplicates<tmpl::flatten<tmpl::transform<
          cce_integrand_tags, tmpl::bind<integrand_temporary_tags, tmpl::_1>>>>;
  using cce_pre_swsh_derivatives_tags = all_pre_swsh_derivative_tags;
  using cce_transform_buffer_tags = all_transform_buffer_tags;
  using cce_swsh_derivative_tags = all_swsh_derivative_tags;
  using cce_angular_coordinate_tags = tmpl::list<Tags::CauchyAngularCoords>;
  using cce_scri_tags =
      tmpl::list<Cce::Tags::News, Cce::Tags::ScriPlus<Cce::Tags::Strain>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi0>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi1>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi2>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi3>,
                 Cce::Tags::TimeIntegral<Cce::Tags::ScriPlus<Cce::Tags::Psi4>>,
                 Cce::Tags::ScriPlusFactor<Cce::Tags::Psi4>>;

  using ccm_psi0 = tmpl::list<Cce::Tags::BoundaryValue<Cce::Tags::Psi0Match>>;

  using component_list =
      tmpl::list<mock_gh_evolution<test_metavariables>,
                 mock_characteristic_evolution<test_metavariables>>;
  using temporal_id = ::Tags::TimeStepId;
  enum class Phase { Initialization, Evolve, Exit };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.Actions.GhBoundaryCommunication",
                  "[Unit][Cce]") {
  using cce_component = mock_characteristic_evolution<test_metavariables>;
  using gh_component = mock_gh_evolution<test_metavariables>;

  MAKE_GENERATOR(gen);
  UniformCustomDistribution<size_t> resolution_distribution{7, 10};
  const size_t l_max = resolution_distribution(gen);

  const Slab single_step_slab{0.0,0.0 + 0.1};
  const Time initial_time = single_step_slab.start();
  TimeStepId initial_time_id{true, 0, initial_time};

  ActionTesting::MockRuntimeSystem<test_metavariables> runner{
      tuples::tagged_tuple_from_typelist<
          Parallel::get_const_global_cache_tags<test_metavariables>>{
          l_max}};

  runner.set_phase(test_metavariables::Phase::Initialization);
  ActionTesting::emplace_component_and_initialize<cce_component>(
      &runner, 0, initial_time_id);
  ActionTesting::emplace_component_and_initialize<
      gh_component>(&runner, 0, initial_time_id);

  ActionTesting::next_action<gh_component>(make_not_null(&runner), 0);
  ActionTesting::next_action<gh_component>(make_not_null(&runner), 0);

  ActionTesting::next_action<cce_component>(make_not_null(&runner), 0);
  ActionTesting::next_action<cce_component>(make_not_null(&runner), 0);

  runner.set_phase(test_metavariables::Phase::Evolve);
  ActionTesting::next_action<cce_component>(make_not_null(&runner), 0);
  CHECK_FALSE(ActionTesting::is_ready<gh_component>(runner, 0));

  // TODO change later
  UniformCustomDistribution<double> value_distribution{0.1, 1.0};
  const size_t number_of_angular_grid_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  Scalar<SpinWeighted<ComplexDataVector, 2>> generated_psi0{
      number_of_angular_grid_points};
  for (size_t i = 0; i < get(generated_psi0).data().size(); ++i)
    get(generated_psi0).data()[i] = value_distribution(gen);


  ActionTesting::simple_action<cce_component,
                               test_action<test_metavariables>>(
      make_not_null(&runner), 0, generated_psi0);

  const auto& test_lhs_cce = ActionTesting::get_databox_tag<
      cce_component, Cce::Tags::BoundaryValue<Cce::Tags::Psi0Match>>(
      runner, 0);

  Approx angular_derivative_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e4)
          .scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(test_lhs_cce, generated_psi0,
                               angular_derivative_approx);

  ActionTesting::invoke_queued_simple_action<cce_component>(
      make_not_null(&runner), 0);
  CHECK(ActionTesting::is_ready<gh_component>(runner, 0));
  ActionTesting::next_action<gh_component>(make_not_null(&runner), 0);

  const auto& test_lhs_gh = ActionTesting::get_databox_tag<
      gh_component, Cce::Tags::BoundaryValue<Cce::Tags::Psi0Match>>(
      runner, 0);

  CHECK_ITERABLE_CUSTOM_APPROX(test_lhs_gh, generated_psi0,
                               angular_derivative_approx);
}
}  // namespace Cce
