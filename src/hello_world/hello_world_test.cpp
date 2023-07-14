// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "hello_world/hello_world_test.hpp"

#include <boost/preprocessor.hpp>
#include <complex>
#include <iostream>
#include <string>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Transpose.hpp"
#include "Evolution/Executables/Cce/CharacteristicExtractBase.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeCharacteristicEvolutionScri.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeCharacteristicEvolutionVariables.hpp"
#include "Evolution/Systems/Cce/Actions/Psi0Matching.hpp"
#include "Evolution/Systems/Cce/Actions/UpdateGauge.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/Components/CharacteristicEvolution.hpp"
#include "Evolution/Systems/Cce/Equations.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/PreSwshDerivatives.hpp"
#include "Evolution/Systems/Cce/PrecomputeCceDependencies.hpp"
#include "Evolution/Systems/Cce/SwshDerivatives.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Evolution/Systems/Cce/WorldtubeBufferUpdater.hpp"
#include "Evolution/Systems/Cce/WorldtubeDataManager.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/CubicSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/LinearSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
#include "Options/Options.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Printf.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Time/StepChoosers/Factory.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/Factory.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Utilities/MakeString.hpp"

struct MyEvolutionMetavars : CharacteristicExtractDefaults<true> {
  using cce_boundary_component = tmpl::list<>;
  static constexpr bool local_time_stepping = false;
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
};

void ccm_functions11(
    std::vector<double>& re_h, std::vector<double>& im_h, const size_t l_max,
    const size_t number_of_radial_points, const double radius,
    const std::vector<double>& re_j, const std::vector<double>& im_j,
    const std::vector<std::vector<double>>& cauchy_cart,
    const std::vector<std::vector<double>>& inertial_cart,
    const Cce::Tags::BoundaryValue<Cce::Tags::BondiBeta>::type& cce_bondi_beta,
    const Cce::Tags::BoundaryValue<Cce::Tags::Dr<Cce::Tags::BondiJ>>::type&
        cce_bondi_dr_j,
    const Cce::Tags::BoundaryValue<Cce::Tags::Du<Cce::Tags::BondiR>>::type&
        cce_du_R,
    const Cce::Tags::BoundaryValue<Cce::Tags::BondiH>::type& cce_bondi_h,
    const Cce::Tags::BoundaryValue<Cce::Tags::BondiJ>::type& cce_bondi_j,
    const Cce::Tags::BoundaryValue<Cce::Tags::BondiQ>::type& cce_bondi_q,
    const Cce::Tags::BoundaryValue<Cce::Tags::BondiR>::type& cce_bondi_R,
    const Cce::Tags::BoundaryValue<Cce::Tags::BondiU>::type& cce_bondi_u,
    const Cce::Tags::BoundaryValue<Cce::Tags::BondiW>::type& cce_bondi_w,
    const Cce::Tags::BoundaryValue<Cce::Tags::Dr<Cce::Tags::BondiU>>::type&
        cce_dr_u,
    const Cce::Tags::BoundaryValue<Cce::Tags::Du<Cce::Tags::BondiJ>>::type&
        cce_du_j,
    const Cce::Tags::BoundaryValue<Cce::Tags::DuRDividedByR>::type&
        bondi_du_r_bdry_DuRDividedByR) {
  // const DataVector gh_read{const_cast<double*>(gh.data()), gh.size()};

  const size_t filter_l_max = l_max - 2;
  const size_t scri_interpolation_order = 5;
  const double radial_filter_alpha = 35.0;
  const size_t radial_filter_half_power = 24;

  using Metavariables = MyEvolutionMetavars;

  using initialize_action =
      Cce::Actions::InitializeCharacteristicEvolutionVariables<Metavariables>;
  using initialize_scri = Cce::Actions::InitializeCharacteristicEvolutionScri<
      Metavariables::scri_values_to_observe,
      Metavariables::cce_boundary_component>;
  using simple_tags_for_evolution =
      initialize_action::simple_tags_for_evolution;
  using simple_tags_for_scri = initialize_scri::simple_tags;
  using from_cache =
      tmpl::list<Cce::InitializationTags::ScriInterpolationOrder,
                 Cce::Tags::LMax, Cce::Tags::NumberOfRadialPoints,
                 Cce::Tags::FilterLMax, Cce::Tags::RadialFilterAlpha,
                 Cce::Tags::RadialFilterHalfPower>;
  using simple_tags =
      tmpl::append<from_cache, simple_tags_for_evolution, simple_tags_for_scri>;

  auto spectre_box = db::create<db::AddSimpleTags<simple_tags>>();

  /****************************Initialization*************************************/
  Initialization::mutate_assign<from_cache>(
      make_not_null(&spectre_box),
      Cce::InitializationTags::ScriInterpolationOrder::type{
          scri_interpolation_order},
      Cce::Tags::LMax::type{l_max},
      Cce::OptionTags::NumberOfRadialPoints::type{number_of_radial_points},
      Cce::Tags::FilterLMax::type{filter_l_max},
      Cce::Tags::RadialFilterAlpha::type{radial_filter_alpha},
      Cce::Tags::RadialFilterHalfPower::type{radial_filter_half_power});
  initialize_action::initialize_impl(make_not_null(&spectre_box));

  initialize_scri::initialize_impl(
      make_not_null(&spectre_box),
      typename Metavariables::scri_values_to_observe{});

  /****************************Get_Boundary_Data*************************************/
  db::mutate<Cce::Tags::BoundaryValue<Cce::Tags::BondiBeta>,
             Cce::Tags::BoundaryValue<Cce::Tags::Dr<Cce::Tags::BondiJ>>,
             Cce::Tags::BoundaryValue<Cce::Tags::Du<Cce::Tags::BondiR>>,
             Cce::Tags::BoundaryValue<Cce::Tags::BondiH>,
             Cce::Tags::BoundaryValue<Cce::Tags::BondiJ>,
             Cce::Tags::BoundaryValue<Cce::Tags::BondiQ>,
             Cce::Tags::BoundaryValue<Cce::Tags::BondiR>,
             Cce::Tags::BoundaryValue<Cce::Tags::BondiU>,
             Cce::Tags::BoundaryValue<Cce::Tags::BondiW>,
             Cce::Tags::BoundaryValue<Cce::Tags::Dr<Cce::Tags::BondiU>>,
             Cce::Tags::BoundaryValue<Cce::Tags::Du<Cce::Tags::BondiJ>>,
             Cce::Tags::BoundaryValue<Cce::Tags::Du<Cce::Tags::BondiR>>>(
      [&l_max, &radius, &cce_bondi_beta, &cce_bondi_dr_j, &cce_du_R,
       &cce_bondi_h, &cce_bondi_j, &cce_bondi_q, &cce_bondi_R, &cce_bondi_u,
       &cce_bondi_w, &cce_dr_u, &cce_du_j, &bondi_du_r_bdry_DuRDividedByR](
          const gsl::not_null<
              Cce::Tags::BoundaryValue<Cce::Tags::BondiBeta>::type*>
              bondi_beta,
          const gsl::not_null<
              Cce::Tags::BoundaryValue<Cce::Tags::Dr<Cce::Tags::BondiJ>>::type*>
              bondi_dr_j,
          const gsl::not_null<
              Cce::Tags::BoundaryValue<Cce::Tags::Du<Cce::Tags::BondiR>>::type*>
              bondi_du_r,
          const gsl::not_null<
              Cce::Tags::BoundaryValue<Cce::Tags::BondiH>::type*>
              bondi_h,
          const gsl::not_null<
              Cce::Tags::BoundaryValue<Cce::Tags::BondiJ>::type*>
              bondi_j,
          const gsl::not_null<
              Cce::Tags::BoundaryValue<Cce::Tags::BondiQ>::type*>
              bondi_q,
          const gsl::not_null<
              Cce::Tags::BoundaryValue<Cce::Tags::BondiR>::type*>
              bondi_r,
          const gsl::not_null<
              Cce::Tags::BoundaryValue<Cce::Tags::BondiU>::type*>
              bondi_u,
          const gsl::not_null<
              Cce::Tags::BoundaryValue<Cce::Tags::BondiW>::type*>
              bondi_w,
          const gsl::not_null<
              Cce::Tags::BoundaryValue<Cce::Tags::Dr<Cce::Tags::BondiU>>::type*>
              dr_u,
          const gsl::not_null<
              Cce::Tags::BoundaryValue<Cce::Tags::Du<Cce::Tags::BondiJ>>::type*>
              du_j,
          const gsl::not_null<
              Cce::Tags::BoundaryValue<Cce::Tags::DuRDividedByR>::type*>
              du_r_r) {
        *bondi_beta = cce_bondi_beta;
        *bondi_dr_j = cce_bondi_dr_j;
        *bondi_du_r = cce_du_R;
        *bondi_h = cce_bondi_h;
        *bondi_j = cce_bondi_j;
        *bondi_q = cce_bondi_q;
        *bondi_r = cce_bondi_R;
        *bondi_u = cce_bondi_u;
        *bondi_w = cce_bondi_w;
        *dr_u = cce_dr_u;
        *du_j = cce_du_j;
        *du_r_r = bondi_du_r_bdry_DuRDividedByR;
      },
      make_not_null(&spectre_box));

  /****************************Construct_Bondi_J*************************************/
  db::mutate<Cce::Tags::BondiJ, Cce::Tags::CauchyCartesianCoords,
             Cce::Tags::PartiallyFlatCartesianCoords>(
      [&cauchy_cart, &inertial_cart, &re_j, &im_j](
          const gsl::not_null<Cce::Tags::BondiJ::type*> bondi_j,
          const gsl::not_null<Cce::Tags::CauchyCartesianCoords::type*>
              spectre_cauchy_cart,
          const gsl::not_null<Cce::Tags::PartiallyFlatCartesianCoords::type*>
              spectre_inertial_cart) {
        for (int jij = 0; jij < cauchy_cart[0].size(); jij++) {
          get<0>(*spectre_cauchy_cart)[jij] = cauchy_cart[0][jij];
          get<0>(*spectre_inertial_cart)[jij] = inertial_cart[0][jij];
          get<1>(*spectre_cauchy_cart)[jij] = cauchy_cart[1][jij];
          get<1>(*spectre_inertial_cart)[jij] = inertial_cart[1][jij];
          get<2>(*spectre_cauchy_cart)[jij] = cauchy_cart[2][jij];
          get<2>(*spectre_inertial_cart)[jij] = inertial_cart[2][jij];
        }
        for (int jij = 0; jij < re_j.size(); jij++) {
          get(*bondi_j).data()[jij] =
              re_j[jij] * std::complex<double>(1.0, 0.0) +
              im_j[jij] * std::complex<double>(0.0, 1.0);
        }
      },
      make_not_null(&spectre_box));

  /****************************UpdateGauge*************************************/
  tmpl::for_each<Cce::Actions::UpdateGauge<true>::cce_mutators>(
      [&spectre_box](auto mutator_v) {
        using mutator = typename decltype(mutator_v)::type;
        db::mutate_apply<mutator>(make_not_null(&spectre_box));
      });

  tmpl::for_each<Cce::Actions::UpdateGauge<true>::ccm_mutators>(
      [&spectre_box](auto mutator_v) {
        using mutator = typename decltype(mutator_v)::type;
        db::mutate_apply<mutator>(make_not_null(&spectre_box));
      });

  /****************************PrecomputeGlobalCceDependencies*************************************/
  tmpl::for_each<Cce::gauge_adjustments_setup_tags>([&spectre_box](auto tag_v) {
    using tag = typename decltype(tag_v)::type;
    db::mutate_apply<Cce::GaugeAdjustedBoundaryValue<tag>>(
        make_not_null(&spectre_box));
  });

  Cce::mutate_all_precompute_cce_dependencies<
      Cce::Tags::EvolutionGaugeBoundaryValue>(make_not_null(&spectre_box));

  // Sizheng stops here.
    auto& result = db::get<Cce::Tags::EthRDividedByR>(
        spectre_box);
    std::cout << std::setprecision(30) << get(result).data()[0] << " ";

  /****************************CalculatePsi0AndDerivAtInnerBoundary*************************************/
  tmpl::for_each<Cce::Actions::CalculatePsi0AndDerivAtInnerBoundary::mutators>(
      [&spectre_box](auto mutator_v) {
        using mutator = typename decltype(mutator_v)::type;
        db::mutate_apply<mutator>(make_not_null(&spectre_box));
      });

  /****************************hypersurface_computation*************************************/
  ;
  tmpl::for_each<Cce::bondi_hypersurface_step_tags>([&spectre_box](
                                                        auto tag_v1) {
    using BondiTag = typename decltype(tag_v1)::type;
    db::mutate_apply<Cce::GaugeAdjustedBoundaryValue<BondiTag>>(
        make_not_null(&spectre_box));
    Cce::mutate_all_pre_swsh_derivatives_for_tag<BondiTag>(
        make_not_null(&spectre_box));
    Cce::mutate_all_swsh_derivatives_for_tag<BondiTag>(
        make_not_null(&spectre_box));

    tmpl::for_each<
        Cce::integrand_terms_to_compute_for_bondi_variable<BondiTag>>(
        [&spectre_box](auto tag_v) {
          using tag = typename decltype(tag_v)::type;
          db::mutate_apply<Cce::ComputeBondiIntegrand<tag>>(
              make_not_null(&spectre_box));
        });
    db::mutate_apply<Cce::RadialIntegrateBondi<
        Cce::Tags::EvolutionGaugeBoundaryValue, BondiTag>>(
        make_not_null(&spectre_box));
    if constexpr (std::is_same_v<BondiTag, Cce::Tags::BondiU>) {
      db::mutate_apply<Cce::GaugeUpdateTimeDerivatives>(
          make_not_null(&spectre_box));
      db::mutate_apply<Cce::GaugeUpdateInertialTimeDerivatives>(
          make_not_null(&spectre_box));
      db::mutate_apply<
          Cce::GaugeAdjustedBoundaryValue<Cce::Tags::DuRDividedByR>>(
          make_not_null(&spectre_box));
      db::mutate_apply<Cce::PrecomputeCceDependencies<
          Cce::Tags::EvolutionGaugeBoundaryValue, Cce::Tags::DuRDividedByR>>(
          make_not_null(&spectre_box));
    };
  });

  /****************************FilterSwshVolumeQuantity*************************************/
  Cce::Actions::FilterSwshVolumeQuantity<Cce::Tags::BondiH>::apply(spectre_box);

  /*************************compute_scri_quantities_and_observe*****************************/
  db::mutate_apply<
      Cce::CalculateScriPlusValue<::Tags::dt<Cce::Tags::InertialRetardedTime>>>(
      make_not_null(&spectre_box));

  Cce::Actions::CalculateScriInputs::apply(spectre_box);
  tmpl::for_each<Metavariables::cce_scri_tags>([&spectre_box](auto tag_v) {
    using tag = typename decltype(tag_v)::type;
    db::mutate_apply<Cce::CalculateScriPlusValue<tag>>(
        make_not_null(&spectre_box));
  });

  /*************************after_cce*****************************/
  // I don't have InsertInterpolationScriData and ScriObserveInterpolated
  //   std::cout << "final: BondiH size: "
  //             << get(get<Cce::Tags::BondiH>(spectre_box)).size() <<
  //             std::endl;
  auto& final_h = get(get<Cce::Tags::BondiH>(spectre_box));
  for (unsigned int i = 0; i < final_h.size(); i++) {
    re_h.push_back(real(final_h.data())[i]);
    im_h.push_back(real(final_h.data())[i]);
  }
  auto& dt_cauchy_cart =
      db::get<::Tags::dt<Cce::Tags::CauchyCartesianCoords>>(spectre_box);
  auto& dt_inertial_cart =
      db::get<::Tags::dt<Cce::Tags::PartiallyFlatCartesianCoords>>(spectre_box);
}