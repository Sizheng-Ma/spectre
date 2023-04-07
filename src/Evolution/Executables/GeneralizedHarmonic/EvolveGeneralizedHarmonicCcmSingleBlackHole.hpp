// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <vector>

#include "ApparentHorizons/ComputeExcisionBoundaryVolumeQuantities.hpp"
#include "ApparentHorizons/ComputeExcisionBoundaryVolumeQuantities.tpp"
#include "ApparentHorizons/ComputeHorizonVolumeQuantities.hpp"
#include "ApparentHorizons/ComputeHorizonVolumeQuantities.tpp"
#include "ApparentHorizons/ComputeItems.hpp"
#include "ApparentHorizons/HorizonAliases.hpp"
#include "ApparentHorizons/Tags.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Evolution/Executables/Cce/CharacteristicExtractBase.hpp"
#include "Evolution/Executables/GeneralizedHarmonic/GeneralizedHarmonicBase.hpp"
#include "Evolution/Systems/Cce/Actions/SendGhVarsToCce.hpp"
#include "Evolution/Systems/Cce/Callbacks/SendGhWorldtubeData.hpp"
#include "Evolution/Systems/Cce/Components/CharacteristicEvolution.hpp"
#include "Evolution/Systems/Cce/Components/WorldtubeBoundary.hpp"
#include "Evolution/Systems/Cce/Initialize/ConformalFactor.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Evolution/Systems/Cce/Initialize/InverseCubic.hpp"
#include "Evolution/Systems/Cce/Initialize/NoIncomingRadiation.hpp"
#include "Evolution/Systems/Cce/Initialize/RegisterInitializeJWithCharm.hpp"
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
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/RegisterDerived.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/RegisterDerivedWithCharm.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/CubicSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/LinearSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "Options/FactoryHelpers.hpp"
#include "Options/Options.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/PhaseControl/PhaseControlTags.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/Actions/InitializeItems.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/CleanUpInterpolator.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/ElementInitInterpPoints.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InitializeInterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolationTargetReceiveVars.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolatorReceivePoints.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolatorReceiveVolumeData.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolatorRegisterElement.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/TryToInterpolate.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ErrorOnFailedApparentHorizon.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/FindApparentHorizon.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/IgnoreFailedApparentHorizon.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ObserveSurfaceData.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ObserveTimeSeriesOnSurface.hpp"
#include "ParallelAlgorithms/Interpolation/Events/Interpolate.hpp"
#include "ParallelAlgorithms/Interpolation/Events/InterpolateWithoutInterpComponent.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Interpolator.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/ApparentHorizon.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/KerrHorizon.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/Sphere.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/GaugeWave.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"

#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/ReceivePsi0FromCce.hpp"

/// \cond
namespace Frame {
// IWYU pragma: no_forward_declare MathFunction
struct Inertial;
}  // namespace Frame
namespace Parallel {
template <typename Metavariables>
class CProxy_GlobalCache;
}  // namespace Parallel
/// \endcond

template <size_t VolumeDim, bool UseNumericalInitialData, bool EvolveCcm>
struct EvolutionMetavars
    : public GeneralizedHarmonicTemplateBase<
          EvolutionMetavars<VolumeDim, UseNumericalInitialData, EvolveCcm>>,
      public CharacteristicExtractDefaults<EvolveCcm> {
  using gh_base = GeneralizedHarmonicTemplateBase<
      EvolutionMetavars<VolumeDim, UseNumericalInitialData, EvolveCcm>>;
  using cce_base = CharacteristicExtractDefaults<EvolveCcm>;
  using cce_base::uses_partially_flat_cartesian_coordinates;
  using typename gh_base::initialize_initial_data_dependent_quantities_actions;
  using cce_boundary_component = Cce::GhWorldtubeBoundary<EvolutionMetavars>;

  static constexpr bool local_time_stepping = gh_base::local_time_stepping;

  template <bool DuringSelfStart>
  struct CceWorldtubeTarget;

  struct AhA : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::Time;
    using tags_to_observe = ::ah::tags_for_observing<Frame::Inertial>;
    using surface_tags_to_observe = ::ah::surface_tags_for_observing;
    using compute_vars_to_interpolate = ah::ComputeHorizonVolumeQuantities;
    using vars_to_interpolate_to_target =
        ::ah::vars_to_interpolate_to_target<VolumeDim, ::Frame::Inertial>;
    using compute_items_on_target =
        ::ah::compute_items_on_target<VolumeDim, Frame::Inertial>;
    using compute_target_points =
        intrp::TargetPoints::ApparentHorizon<AhA, ::Frame::Inertial>;
    using post_interpolation_callback =
        intrp::callbacks::FindApparentHorizon<AhA, ::Frame::Inertial>;
    using horizon_find_failure_callback =
        intrp::callbacks::IgnoreFailedApparentHorizon;
    using post_horizon_find_callbacks = tmpl::list<
        intrp::callbacks::ObserveTimeSeriesOnSurface<tags_to_observe, AhA>,
        intrp::callbacks::ObserveSurfaceData<surface_tags_to_observe, AhA,
                                             ::Frame::Inertial>>;
  };

  struct ExcisionBoundaryA
      : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::Time;
    using tags_to_observe = tmpl::list<
        gr::Tags::LapseCompute<VolumeDim, ::Frame::Inertial, DataVector>>;
    using compute_vars_to_interpolate =
        ah::ComputeExcisionBoundaryVolumeQuantities;
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::SpacetimeMetric<VolumeDim, Frame::Inertial>>;
    using compute_items_on_source = tmpl::list<>;
    using compute_items_on_target = tmpl::append<tmpl::list<
        gr::Tags::SpatialMetricCompute<VolumeDim, Frame::Inertial, DataVector>,
        gr::Tags::DetAndInverseSpatialMetricCompute<VolumeDim, Frame::Inertial,
                                                    DataVector>,
        gr::Tags::ShiftCompute<VolumeDim, Frame::Inertial, DataVector>,
        gr::Tags::LapseCompute<VolumeDim, Frame::Inertial, DataVector>,
        GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma1>>;

    using compute_target_points =
        intrp::TargetPoints::Sphere<ExcisionBoundaryA, ::Frame::Grid>;
    using post_interpolation_callback =
        intrp::callbacks::ObserveSurfaceData<tags_to_observe, ExcisionBoundaryA,
                                             ::Frame::Grid>;
    // run_callbacks
    template <typename metavariables>
    using interpolating_component = typename metavariables::gh_dg_element_array;
  };

  using interpolator_source_vars_excision_boundary = tmpl::list<
      gr::Tags::SpacetimeMetric<VolumeDim, Frame::Inertial>,
      GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma1>;
  using interpolator_source_vars = tmpl::remove_duplicates<
      tmpl::append<interpolator_source_vars_excision_boundary,
                   ::ah::source_vars<VolumeDim>>>;

  using dg_registration_list =
      tmpl::push_back<typename gh_base::dg_registration_list,
                      intrp::Actions::RegisterElementWithInterpolator>;

  using typename gh_base::system;

  template <bool DuringSelfStart>
  using step_actions = tmpl::list<
      tmpl::conditional_t<
          DuringSelfStart,
          Cce::Actions::SendGhVarsToCce<CceWorldtubeTarget<true>>,
          Cce::Actions::SendGhVarsToCce<CceWorldtubeTarget<false>>>,
      tmpl::conditional_t<
         uses_partially_flat_cartesian_coordinates,
         GeneralizedHarmonic::Actions::ReceiveCCEData<EvolutionMetavars>,
         tmpl::list<>>,
      evolution::dg::Actions::ComputeTimeDerivative<
          VolumeDim, system, AllStepChoosers, local_time_stepping>,
      tmpl::conditional_t<
          local_time_stepping,
          tmpl::list<evolution::Actions::RunEventsAndDenseTriggers<
                         tmpl::list<evolution::dg::ApplyBoundaryCorrections<
                             local_time_stepping, system, VolumeDim, true>>>,
                     evolution::dg::Actions::ApplyLtsBoundaryCorrections<
                         system, VolumeDim, false>>,
          tmpl::list<
              evolution::dg::Actions::ApplyBoundaryCorrectionsToTimeDerivative<
                  system, VolumeDim, false>,
              Actions::RecordTimeStepperData<system>,
              evolution::Actions::RunEventsAndDenseTriggers<tmpl::list<>>,
              Actions::UpdateU<system>,
              dg::Actions::Filter<
                  Filters::Exponential<0>,
                  tmpl::list<
                      gr::Tags::SpacetimeMetric<VolumeDim, Frame::Inertial,
                                                DataVector>,
                      GeneralizedHarmonic::Tags::Pi<VolumeDim, Frame::Inertial>,
                      GeneralizedHarmonic::Tags::Phi<VolumeDim,
                                                     Frame::Inertial>>>>>>;

  using typename cce_base::cce_step_choosers;

  static constexpr bool override_functions_of_time = false;

  // initialization actions are the same as the default, with the single
  // addition of initializing the interpolation points (second-to-last action).
  using initialization_actions = tmpl::list<
      Initialization::Actions::InitializeItems<
          Initialization::TimeStepping<EvolutionMetavars, local_time_stepping>,
          evolution::dg::Initialization::Domain<VolumeDim>,
          Initialization::TimeStepperHistory<EvolutionMetavars>>,
      Initialization::Actions::NonconservativeSystem<system>,
      std::conditional_t<
          UseNumericalInitialData, tmpl::list<>,
          evolution::Initialization::Actions::SetVariables<
              domain::Tags::Coordinates<VolumeDim, Frame::ElementLogical>>>,
      Initialization::Actions::AddComputeTags<::Tags::DerivCompute<
          typename system::variables_tag,
          domain::Tags::InverseJacobian<VolumeDim, Frame::ElementLogical,
                                        Frame::Inertial>,
          typename system::gradient_variables>>,
      Initialization::Actions::InitializeCcmTags<EvolutionMetavars>,
      GeneralizedHarmonic::Actions::InitializeGhAnd3Plus1Variables<VolumeDim>,
      Initialization::Actions::AddComputeTags<
          tmpl::push_back<StepChoosers::step_chooser_compute_tags<
              EvolutionMetavars, local_time_stepping>>>,
      ::evolution::dg::Initialization::Mortars<VolumeDim, system>,
      evolution::Actions::InitializeRunEventsAndDenseTriggers,
      intrp::Actions::ElementInitInterpPoints<
          intrp::Tags::InterpPointInfo<EvolutionMetavars>>,
      Parallel::Actions::TerminatePhase>;

  using gh_dg_element_array = DgElementArray<
      EvolutionMetavars,
      tmpl::flatten<tmpl::list<
          Parallel::PhaseActions<Parallel::Phase::Initialization,
                                 initialization_actions>,
          tmpl::conditional_t<
              UseNumericalInitialData,
              tmpl::list<
                  Parallel::PhaseActions<
                      Parallel::Phase::RegisterWithElementDataReader,
                      tmpl::list<
                          importers::Actions::RegisterWithElementDataReader,
                          Parallel::Actions::TerminatePhase>>,
                  Parallel::PhaseActions<
                      Parallel::Phase::ImportInitialData,
                      tmpl::list<
                          GeneralizedHarmonic::Actions::ReadNumericInitialData<
                              evolution::OptionTags::NumericInitialData>,
                          GeneralizedHarmonic::Actions::SetNumericInitialData<
                              evolution::OptionTags::NumericInitialData>,
                          Parallel::Actions::TerminatePhase>>>,
              tmpl::list<>>,
          Parallel::PhaseActions<
              Parallel::Phase::InitializeInitialDataDependentQuantities,
              initialize_initial_data_dependent_quantities_actions>,
          Parallel::PhaseActions<
              Parallel::Phase::InitializeTimeStepperHistory,
              SelfStart::self_start_procedure<step_actions<true>, system>>,
          Parallel::PhaseActions<Parallel::Phase::Register,
                                 tmpl::list<dg_registration_list,
                                            Parallel::Actions::TerminatePhase>>,
          Parallel::PhaseActions<
              Parallel::Phase::Evolve,
              tmpl::list<Actions::RunEventsAndTriggers, Actions::ChangeSlabSize,
                         step_actions<false>, Actions::AdvanceTime,
                         PhaseControl::Actions::ExecutePhaseChange>>>>>;

  template <bool DuringSelfStart>
  struct CceWorldtubeTarget
      : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::TimeStepId;

    static std::string name() {
      return DuringSelfStart ? "SelfStartCceWorldtubeTarget"
                             : "CceWorldtubeTarget";
    }
    using compute_items_on_source = tmpl::list<>;
    using compute_items_on_target = tmpl::list<>;
    using compute_target_points =
        intrp::TargetPoints::Sphere<CceWorldtubeTarget, ::Frame::Inertial>;
    using post_interpolation_callback = intrp::callbacks::SendGhWorldtubeData<
        Cce::CharacteristicEvolution<EvolutionMetavars>, CceWorldtubeTarget,
        DuringSelfStart, local_time_stepping>;
    using vars_to_interpolate_to_target =
        tmpl::list<::gr::Tags::SpacetimeMetric<VolumeDim, Frame::Inertial>,
                   ::GeneralizedHarmonic::Tags::Phi<VolumeDim, Frame::Inertial>,
                   ::GeneralizedHarmonic::Tags::Pi<VolumeDim, Frame::Inertial>>;
    template <typename Metavariables>
    using interpolating_component = gh_dg_element_array;
  };

  using interpolation_target_tags =
      tmpl::list<CceWorldtubeTarget<false>, CceWorldtubeTarget<true>, AhA,
                 ExcisionBoundaryA>;

  template <typename ParallelComponent>
  struct registration_list {
    using type = std::conditional_t<
        std::is_same_v<ParallelComponent, gh_dg_element_array>,
        dg_registration_list, tmpl::list<>>;
  };
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = Options::add_factory_classes<
        typename gh_base::factory_creation::factory_classes,
        tmpl::pair<
            Event,
            tmpl::list<
                intrp::Events::Interpolate<3, AhA, interpolator_source_vars>,
                intrp::Events::InterpolateWithoutInterpComponent<
                    3, ExcisionBoundaryA, EvolutionMetavars,
                    interpolator_source_vars_excision_boundary>>>,
        tmpl::pair<StepChooser<StepChooserUse::LtsStep>, cce_step_choosers>>;
  };

  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<tmpl::push_back<
          tmpl::at<typename factory_creation::factory_classes, Event>,
          typename AhA::post_horizon_find_callbacks>>;

  using component_list = tmpl::flatten<tmpl::list<
      observers::Observer<EvolutionMetavars>,
      observers::ObserverWriter<EvolutionMetavars>,
      std::conditional_t<UseNumericalInitialData,
                         importers::ElementDataReader<EvolutionMetavars>,
                         tmpl::list<>>,
      gh_dg_element_array, intrp::Interpolator<EvolutionMetavars>,
      tmpl::transform<interpolation_target_tags,
                      tmpl::bind<intrp::InterpolationTarget,
                                 tmpl::pin<EvolutionMetavars>, tmpl::_1>>,
      cce_boundary_component, Cce::CharacteristicEvolution<EvolutionMetavars>>>;

  static constexpr Options::String help{
      "Evolve the Einstein field equations using the Generalized Harmonic "
      "formulation\n"
      "with a coupled CCE evolution for asymptotic wave data output.\n"
      "The system shouldn't have black holes."};
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &setup_memory_allocation_failure_reporting,
    &disable_openblas_multithreading,
    &domain::creators::time_dependence::register_derived_with_charm,
    &domain::FunctionsOfTime::register_derived_with_charm,
    &domain::creators::register_derived_with_charm,
    &GeneralizedHarmonic::BoundaryCorrections::register_derived_with_charm,
    &GeneralizedHarmonic::ConstraintDamping::register_derived_with_charm,
    &Cce::register_initialize_j_with_charm<
        metavariables::uses_partially_flat_cartesian_coordinates,
        metavariables::cce_boundary_component>,
    &Parallel::register_derived_classes_with_charm<Cce::WorldtubeDataManager>,
    &Parallel::register_derived_classes_with_charm<intrp::SpanInterpolator>,
    &Parallel::register_factory_classes_with_charm<metavariables>};

static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
