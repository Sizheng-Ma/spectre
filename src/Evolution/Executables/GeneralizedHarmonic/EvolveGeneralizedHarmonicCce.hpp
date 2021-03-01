// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <vector>

#include "ApparentHorizons/Tags.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tags.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/ComputeTags.hpp"
#include "Evolution/Executables/GeneralizedHarmonic/GeneralizedHarmonicBase.hpp"
#include "Evolution/Systems/Cce/Actions/SendNextTimeToCce.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/Components/CharacteristicEvolution.hpp"
#include "Evolution/Systems/Cce/Components/WorldtubeBoundary.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Evolution/Systems/Cce/Initialize/InverseCubic.hpp"
#include "Evolution/Systems/Cce/Initialize/NoIncomingRadiation.hpp"
#include "Evolution/Systems/Cce/Initialize/ZeroNonSmooth.hpp"
#include "Evolution/Systems/Cce/IntegrandInputSteps.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhInterfaceManager.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhLocalTimeStepping.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhLockstep.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/System.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Evolution/Systems/Cce/WorldtubeBufferUpdater.hpp"
#include "Evolution/Systems/Cce/WorldtubeDataManager.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/RegisterDerivedWithCharm.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Tags.hpp"
#include "Informer/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ComputeNonconservativeBoundaryFluxes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ImposeBoundaryConditions.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/Actions/ElementInitInterpPoints.hpp"
#include "NumericalAlgorithms/Interpolation/Actions/InterpolateToTarget.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/Callbacks/SendGhWorldtubeData.hpp"
#include "NumericalAlgorithms/Interpolation/CubicSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetKerrHorizon.hpp"
#include "NumericalAlgorithms/Interpolation/LinearSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/CollectDataForFluxes.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/FluxCommunication.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"
#include "ParallelAlgorithms/Initialization/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Actions/AdvanceTime.hpp"
#include "Time/Actions/ChangeSlabSize.hpp"
#include "Time/Actions/ChangeStepSize.hpp"
#include "Time/Actions/RecordTimeStepperData.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/Actions/UpdateU.hpp"
#include "Time/StepControllers/StepController.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/TMPL.hpp"

#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/ReceivePsi0FromCce.hpp"
/// \cond
namespace Frame {
// IWYU pragma: no_forward_declare MathFunction
struct Inertial;
}  // namespace Frame
namespace Parallel {
template <typename Metavariables>
class CProxy_ConstGlobalCache;
}  // namespace Parallel
/// \endcond

template <typename InitialData, typename BoundaryConditions,
          bool BjorhusExternalBoundary = false>
struct EvolutionMetavars
    : public GeneralizedHarmonicTemplateBase<EvolutionMetavars<
          InitialData, BoundaryConditions, BjorhusExternalBoundary>>,
      public virtual GeneralizedHarmonicDefaults {
  using evolved_swsh_tag = Cce::Tags::BondiJ;
  using evolved_swsh_dt_tag = Cce::Tags::BondiH;
  using evolved_coordinates_variables_tag =
      Tags::Variables<tmpl::list<Cce::Tags::CauchyCartesianCoords,
                               Cce::Tags::InertialCartesianCoords,
                                 Cce::Tags::InertialRetardedTime>>;
  using cce_boundary_communication_tags =
      Cce::Tags::characteristic_worldtube_boundary_tags<
          Cce::Tags::BoundaryValue>;

  using cce_gauge_boundary_tags = tmpl::flatten<tmpl::list<
      tmpl::transform<
          tmpl::list<Cce::Tags::BondiR, Cce::Tags::DuRDividedByR,
                     Cce::Tags::BondiJ, Cce::Tags::Dr<Cce::Tags::BondiJ>,
                     Cce::Tags::BondiBeta, Cce::Tags::BondiQ, Cce::Tags::BondiU,
                     Cce::Tags::BondiW, Cce::Tags::BondiH>,
          tmpl::bind<Cce::Tags::EvolutionGaugeBoundaryValue, tmpl::_1>>,
      Cce::Tags::BondiUAtScri, Cce::Tags::GaugeC, Cce::Tags::GaugeD,
      Cce::Tags::GaugeOmega, Cce::Tags::Du<Cce::Tags::GaugeOmega>,
      Cce::Tags::GaugeCnohat, Cce::Tags::GaugeDnohat,
      Cce::Tags::GaugeOmeganohat, Cce::Tags::InertialDamping,
      Spectral::Swsh::Tags::Derivative<Cce::Tags::GaugeOmega,
                                       Spectral::Swsh::Tags::Eth>,
      Spectral::Swsh::Tags::Derivative<Cce::Tags::GaugeOmeganohat,
                                       Spectral::Swsh::Tags::Eth>,
      Cce::all_boundary_pre_swsh_derivative_tags_for_scri,
      Cce::all_boundary_swsh_derivative_tags_for_scri>>;

  using scri_values_to_observe =
      tmpl::list<Cce::Tags::News, Cce::Tags::ScriPlus<Cce::Tags::Strain>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi0>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi1>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi2>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi3>,
                 Cce::Tags::Du<Cce::Tags::TimeIntegral<
                     Cce::Tags::ScriPlus<Cce::Tags::Psi4>>>,
                 Cce::Tags::EthInertialRetardedTime>;

  using cce_scri_tags =
      tmpl::list<Cce::Tags::News, Cce::Tags::ScriPlus<Cce::Tags::Strain>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi0>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi1>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi2>,
                 Cce::Tags::ScriPlus<Cce::Tags::Psi3>,
                 Cce::Tags::TimeIntegral<Cce::Tags::ScriPlus<Cce::Tags::Psi4>>,
                 Cce::Tags::EthInertialRetardedTime>;
  using cce_integrand_tags = tmpl::flatten<tmpl::transform<
      Cce::bondi_hypersurface_step_tags,
      tmpl::bind<Cce::integrand_terms_to_compute_for_bondi_variable,
                 tmpl::_1>>>;
  using cce_integration_independent_tags =
      tmpl::append<Cce::pre_computation_tags,
                   tmpl::list<Cce::Tags::BondiJ_Cauchyview,Cce::Tags::Psi0Match,
                 Cce::Tags::Dy<Cce::Tags::Psi0Match>,
                 Cce::Tags::Psi0,
                 Cce::Tags::Dy<Cce::Tags::BondiJ_Cauchyview>,
                 Cce::Tags::Dy<Cce::Tags::Dy<Cce::Tags::BondiJ_Cauchyview>>,
                 Cce::Tags::DuRDividedByR>>;
  using cce_temporary_equations_tags = tmpl::remove_duplicates<tmpl::flatten<
      tmpl::transform<cce_integrand_tags,
                      tmpl::bind<Cce::integrand_temporary_tags, tmpl::_1>>>>;
  using cce_pre_swsh_derivatives_tags = Cce::all_pre_swsh_derivative_tags;
  using cce_transform_buffer_tags = Cce::all_transform_buffer_tags;
  using cce_swsh_derivative_tags = Cce::all_swsh_derivative_tags;
  using cce_angular_coordinate_tags =
      tmpl::list<Cce::Tags::CauchyAngularCoords,
                 Cce::Tags::InertialAngularCoords>;

  using cce_boundary_component = Cce::GhWorldtubeBoundary<EvolutionMetavars>;
  using ccm_psi0 = tmpl::list<Cce::Tags::BoundaryValue<Cce::Tags::Psi0Match>>;
using ccm_dpsi0 = tmpl::list<                 Cce::Tags::BoundaryValue<
                     Cce::Tags::Dlambda<Cce::Tags::Psi0Match>>>;
  using tags_for_matching1 =
        tmpl::list<GeneralizedHarmonic::Tags::AngularTetrad<volume_dim, frame>>;
  using tags_for_matching2 =
        tmpl::list<GeneralizedHarmonic::Tags::Psi0FromCceInterpolate>;
  using tags_for_matching3 =
        tmpl::list<GeneralizedHarmonic::Tags::CCMw<volume_dim, frame>>;

  struct CceWorldtubeTarget;
  struct Horizon;

  using interpolator_source_vars =
      tmpl::list<gr::Tags::SpacetimeMetric<volume_dim, frame>,
                 GeneralizedHarmonic::Tags::Pi<volume_dim, frame>,
                 GeneralizedHarmonic::Tags::Phi<volume_dim, frame>,
                 ::Tags::dt<gr::Tags::SpacetimeMetric<volume_dim, frame>>,
                 ::Tags::dt<GeneralizedHarmonic::Tags::Pi<volume_dim, frame>>,
                 ::Tags::dt<GeneralizedHarmonic::Tags::Phi<volume_dim, frame>>>;

  using observation_events = typename GeneralizedHarmonicTemplateBase<
      EvolutionMetavars<InitialData, BoundaryConditions,
                        BjorhusExternalBoundary>>::observation_events;
  using events = tmpl::push_back<observation_events,
                                 intrp::Events::Registrars::Interpolate<
                                     3, Horizon, interpolator_source_vars>>;

  using analytic_solution_tag = typename GeneralizedHarmonicTemplateBase<
      EvolutionMetavars<InitialData, BoundaryConditions,
                        BjorhusExternalBoundary>>::analytic_solution_tag;
  using const_global_cache_tags =
      tmpl::list<analytic_solution_tag, normal_dot_numerical_flux,
                 time_stepper_tag, Tags::EventsAndTriggers<events, triggers>>;

  using initial_data = typename GeneralizedHarmonicTemplateBase<
      EvolutionMetavars<InitialData, BoundaryConditions,
                        BjorhusExternalBoundary>>::initial_data;

  // use the default same step actions except for sending the next time and
  // interpolating to the CCE target. Assumes that the last action is `UpdateU`,
  // so that the insert places the new actions before that.
  template <bool send_to_cce>
  using step_actions = tmpl::list<
      evolution::dg::Actions::ComputeTimeDerivative<EvolutionMetavars>,
      dg::Actions::ComputeNonconservativeBoundaryFluxes<
          domain::Tags::BoundaryDirectionsInterior<volume_dim>>,
      tmpl::conditional_t<
          BjorhusExternalBoundary, tmpl::list<>,
          tmpl::list<
              dg::Actions::ImposeDirichletBoundaryConditions<EvolutionMetavars>,
              dg::Actions::CollectDataForFluxes<
                  boundary_scheme,
                  domain::Tags::BoundaryDirectionsInterior<volume_dim>>>>,
      dg::Actions::ReceiveDataForFluxes<boundary_scheme>,
      tmpl::conditional_t<
          send_to_cce,
          tmpl::list<Cce::Actions::SendNextTimeToCce<CceWorldtubeTarget>,
                     intrp::Actions::InterpolateToTarget<CceWorldtubeTarget>>,
          tmpl::list<>>,
      std::conditional_t<
          local_time_stepping,
          tmpl::list<tmpl::conditional_t<
                         BjorhusExternalBoundary,
                         tmpl::list<
tmpl::conditional_t<send_to_cce,tmpl::list<GeneralizedHarmonic::Actions::
ReceiveCCEData<EvolutionMetavars>>,tmpl::list<>>,
                                    GeneralizedHarmonic::Actions::
                                        ImposeBjorhusBoundaryConditions<
                                            EvolutionMetavars>>,
                         tmpl::list<>>,
                     Actions::RecordTimeStepperData<>,
                     Actions::MutateApply<boundary_scheme>>,
          tmpl::list<Actions::MutateApply<boundary_scheme>,
                     tmpl::conditional_t<
                         BjorhusExternalBoundary,
                         tmpl::list<
tmpl::conditional_t<send_to_cce,tmpl::list<GeneralizedHarmonic::Actions::
ReceiveCCEData<EvolutionMetavars>>,tmpl::list<>>,
                                    GeneralizedHarmonic::Actions::
                                        ImposeBjorhusBoundaryConditions<
                                            EvolutionMetavars>>,
                         tmpl::list<>>,
                     Actions::RecordTimeStepperData<>>>,
      Actions::UpdateU<>>;

  // initialization actions are the same as the default, with the single
  // addition of initializing the interpolation points. Assumes that the last
  // action is `RemoveOptionsAndTerminatePhase` so that the new initialization
  // occurs before termination.
  using initialization_actions = tmpl::list<
      ::Actions::SetupDataBox,
      Initialization::Actions::TimeAndTimeStep<EvolutionMetavars>,
      evolution::dg::Initialization::Domain<volume_dim,
                                            override_cubic_functions_of_time>,
      Initialization::Actions::NonconservativeSystem<system>,
      std::conditional_t<
          evolution::is_numeric_initial_data_v<initial_data>, tmpl::list<>,
          evolution::Initialization::Actions::SetVariables<
              domain::Tags::Coordinates<volume_dim, Frame::Logical>>>,
      Initialization::Actions::TimeStepperHistory<EvolutionMetavars>,
      Initialization::Actions::InitializeCcmTags<EvolutionMetavars>,
      Initialization::Actions::InitializeCcmOtherTags<EvolutionMetavars>,
      GeneralizedHarmonic::Actions::InitializeGhAnd3Plus1Variables<volume_dim>,
      dg::Actions::InitializeInterfaces<
          system,
          dg::Initialization::slice_tags_to_face<
              typename system::variables_tag,
              gr::Tags::SpatialMetric<volume_dim, frame, DataVector>,
              typename gr::Tags::DetAndInverseSpatialMetricCompute<
                  volume_dim, frame, DataVector>::base,
              gr::Tags::Shift<volume_dim, frame, DataVector>,
              gr::Tags::Lapse<DataVector>,
              GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma0,
              GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma1,
              GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma2>,
          tmpl::conditional_t<
              BjorhusExternalBoundary,
              dg::Initialization::slice_tags_to_exterior<
                  typename system::variables_tag,
                  gr::Tags::SpatialMetric<volume_dim, frame, DataVector>,
                  typename gr::Tags::DetAndInverseSpatialMetricCompute<
                      volume_dim, frame, DataVector>::base,
                  gr::Tags::Shift<volume_dim, frame, DataVector>,
                  gr::Tags::Lapse<DataVector>,
                  GeneralizedHarmonic::ConstraintDamping::Tags::
                      ConstraintGamma0,
                  GeneralizedHarmonic::ConstraintDamping::Tags::
                      ConstraintGamma1,
                  GeneralizedHarmonic::ConstraintDamping::Tags::
                      ConstraintGamma2>,
              dg::Initialization::slice_tags_to_exterior<
                  gr::Tags::SpatialMetric<volume_dim, frame, DataVector>,
                  typename gr::Tags::DetAndInverseSpatialMetricCompute<
                      volume_dim, frame, DataVector>::base,
                  gr::Tags::Shift<volume_dim, frame, DataVector>,
                  gr::Tags::Lapse<DataVector>,
                  GeneralizedHarmonic::ConstraintDamping::Tags::
                      ConstraintGamma0,
                  GeneralizedHarmonic::ConstraintDamping::Tags::
                      ConstraintGamma1,
                  GeneralizedHarmonic::ConstraintDamping::Tags::
                      ConstraintGamma2>>,

          dg::Initialization::face_compute_tags<
              domain::Tags::BoundaryCoordinates<volume_dim, true>,
              GeneralizedHarmonic::CharacteristicFieldsCompute<volume_dim,
                                                               frame>>,
          dg::Initialization::exterior_compute_tags<
              GeneralizedHarmonic::CharacteristicFieldsCompute<volume_dim,
                                                               frame>>,
          !BjorhusExternalBoundary, true>,
      Initialization::Actions::AddComputeTags<
          tmpl::list<evolution::Tags::AnalyticCompute<
              volume_dim, analytic_solution_tag, analytic_solution_fields>>>,
      dg::Actions::InitializeMortars<boundary_scheme, !BjorhusExternalBoundary>,
      Initialization::Actions::DiscontinuousGalerkin<EvolutionMetavars>,
      intrp::Actions::ElementInitInterpPoints<
          intrp::Tags::InterpPointInfo<EvolutionMetavars>>,
      Initialization::Actions::RemoveOptionsAndTerminatePhase>;

  using gh_dg_element_array = DgElementArray<
      EvolutionMetavars,
      tmpl::flatten<tmpl::list<
          Parallel::PhaseActions<Phase, Phase::Initialization,
                                 initialization_actions>,
          tmpl::conditional_t<
              evolution::is_numeric_initial_data_v<initial_data>,
              tmpl::list<
                  Parallel::PhaseActions<
                      Phase, Phase::RegisterWithElementDataReader,
                      tmpl::list<
                          importers::Actions::RegisterWithElementDataReader,
                          Parallel::Actions::TerminatePhase>>,
                  Parallel::PhaseActions<
                      Phase, Phase::ImportInitialData,
                      tmpl::list<importers::Actions::ReadVolumeData<
                                     evolution::OptionTags::NumericInitialData,
                                     typename system::variables_tag::tags_list>,
                                 importers::Actions::ReceiveVolumeData<
                                     evolution::OptionTags::NumericInitialData,
                                     typename system::variables_tag::tags_list>,
                                 Parallel::Actions::TerminatePhase>>>,
              tmpl::list<>>,
          Parallel::PhaseActions<
              Phase, Phase::InitializeInitialDataDependentQuantities,
              initialize_initial_data_dependent_quantities_actions>,
          Parallel::PhaseActions<
              Phase, Phase::InitializeTimeStepperHistory,
              SelfStart::self_start_procedure<step_actions<false>>>,
          Parallel::PhaseActions<
              Phase, Phase::Register,
              tmpl::list<intrp::Actions::RegisterElementWithInterpolator,
                         observers::Actions::RegisterEventsWithObservers,
                         Parallel::Actions::TerminatePhase>>,
          Parallel::PhaseActions<
              Phase, Phase::Evolve,
              tmpl::list<Actions::RunEventsAndTriggers, Actions::ChangeSlabSize,
                         step_actions<true>, Actions::AdvanceTime>>>>>;

  struct Horizon {
    using tags_to_observe =
        tmpl::list<StrahlkorperGr::Tags::AreaCompute<frame>>;
    using compute_items_on_source = tmpl::list<
        gr::Tags::SpatialMetricCompute<volume_dim, frame, DataVector>,
        ah::Tags::InverseSpatialMetricCompute<volume_dim, frame>,
        ah::Tags::ExtrinsicCurvatureCompute<volume_dim, frame>,
        ah::Tags::SpatialChristoffelSecondKindCompute<volume_dim, frame>>;
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::SpatialMetric<volume_dim, frame, DataVector>,
                   gr::Tags::InverseSpatialMetric<volume_dim, frame>,
                   gr::Tags::ExtrinsicCurvature<volume_dim, frame>,
                   gr::Tags::SpatialChristoffelSecondKind<volume_dim, frame>>;
    using compute_items_on_target = tmpl::append<
        tmpl::list<StrahlkorperGr::Tags::AreaElementCompute<frame>>,
        tags_to_observe>;
    using compute_target_points =
        intrp::TargetPoints::ApparentHorizon<Horizon, ::Frame::Inertial>;
    using post_interpolation_callback =
        intrp::callbacks::FindApparentHorizon<Horizon, ::Frame::Inertial>;
    using post_horizon_find_callback =
        intrp::callbacks::ObserveTimeSeriesOnSurface<tags_to_observe, Horizon,
                                                     Horizon>;
  };

  struct CceWorldtubeTarget {
    using compute_items_on_source = tmpl::list<>;
    using compute_items_on_target = tmpl::list<>;
    using compute_target_points =
        intrp::TargetPoints::KerrHorizon<CceWorldtubeTarget, ::Frame::Inertial>;
    using post_interpolation_callback = intrp::callbacks::SendGhWorldtubeData<
        Cce::CharacteristicEvolution<EvolutionMetavars>>;
    using vars_to_interpolate_to_target = tmpl::list<
        gr::Tags::SpacetimeMetric<volume_dim, frame>,
        GeneralizedHarmonic::Tags::Pi<volume_dim, frame>,
        GeneralizedHarmonic::Tags::Phi<volume_dim, frame>,
        ::Tags::dt<gr::Tags::SpacetimeMetric<volume_dim, frame>>,
        ::Tags::dt<GeneralizedHarmonic::Tags::Pi<volume_dim, frame>>,
        ::Tags::dt<GeneralizedHarmonic::Tags::Phi<volume_dim, frame>>>;
    template <typename DbTagList>
    static bool should_interpolate(const db::DataBox<DbTagList>& box) noexcept {
      return Cce::InterfaceManagers::should_interpolate_for_strategy(
          box, db::get<Cce::Tags::InterfaceManagerInterpolationStrategy>(box));
    }
    using interpolating_component = gh_dg_element_array;
  };

  using interpolation_target_tags = tmpl::list<Horizon, CceWorldtubeTarget>;
  using observed_reduction_data_tags = observers::collect_reduction_data_tags<
      tmpl::push_back<typename Event<observation_events>::creatable_classes,
                      typename Horizon::post_horizon_find_callback>>;

  using component_list = tmpl::flatten<tmpl::list<
      observers::Observer<EvolutionMetavars>,
      observers::ObserverWriter<EvolutionMetavars>,
      std::conditional_t<evolution::is_numeric_initial_data_v<initial_data>,
                         importers::ElementDataReader<EvolutionMetavars>,
                         tmpl::list<>>,
      intrp::Interpolator<EvolutionMetavars>,
      intrp::InterpolationTarget<EvolutionMetavars, Horizon>,
      intrp::InterpolationTarget<EvolutionMetavars, CceWorldtubeTarget>,
      cce_boundary_component, Cce::CharacteristicEvolution<EvolutionMetavars>,
      gh_dg_element_array>>;

  static constexpr Options::String help{
      "Evolve a generalized harmonic system\n"
      "with a coupled CCE evolution for asymptotic wave data output"};
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling,
    &disable_openblas_multithreading,
    &domain::creators::time_dependence::register_derived_with_charm,
    &domain::FunctionsOfTime::register_derived_with_charm,
    &domain::creators::register_derived_with_charm,
    &GeneralizedHarmonic::ConstraintDamping::register_derived_with_charm,
    &Parallel::register_derived_classes_with_charm<
        Cce::InitializeJ::InitializeJ>,
    &Parallel::register_derived_classes_with_charm<TimeSequence<double>>,
    &Parallel::register_derived_classes_with_charm<TimeSequence<std::uint64_t>>,
    &Parallel::register_derived_classes_with_charm<
        Cce::InterfaceManagers::GhInterfaceManager>,
    &Parallel::register_derived_classes_with_charm<Cce::WorldtubeDataManager>,
    &Parallel::register_derived_classes_with_charm<intrp::SpanInterpolator>,
    &Parallel::register_derived_classes_with_charm<
        Event<metavariables::events>>,
    &Parallel::register_derived_classes_with_charm<
        StepChooser<metavariables::slab_choosers>>,
    &Parallel::register_derived_classes_with_charm<
        StepChooser<metavariables::step_choosers>>,
    &Parallel::register_derived_classes_with_charm<StepController>,
    &Parallel::register_derived_classes_with_charm<TimeStepper>,
    &Parallel::register_derived_classes_with_charm<
        Trigger<metavariables::triggers>>};

static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
