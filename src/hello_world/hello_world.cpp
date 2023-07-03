// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "hello_world/hello_world.hpp"

#include <boost/preprocessor.hpp>
#include <complex>
#include <iostream>
#include <string>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
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

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

std::string link_date() { return std::string(__TIMESTAMP__); }

std::string executable_name() {
  return std::string(BOOST_PP_STRINGIZE(EXECUTABLE_NAME));
}

std::string git_description() {
  return std::string(BOOST_PP_STRINGIZE(GIT_DESCRIPTION));
}

std::string git_branch() { return std::string(BOOST_PP_STRINGIZE(GIT_BRANCH)); }

namespace formaline {
std::vector<char> get_archive() {
  return {'N', 'o', 't', ' ', 's', 'u', 'p', 'p', 'o', 'r', 't', 'e', 'd'};
}

std::string get_environment_variables() { return "Not supported in Python"; }

std::string get_build_info() { return "Not supported in Python"; }

std::string get_paths() { return "Not supported in Python."; }
}  // namespace formaline

void myprint() {
  std::cout << "hello"
            << "\n"
            << std::endl;
}

int mynewfunction(int a) {
  std::cout << a << "\n" << std::endl;
  return a + 2;
}

void print_data_vector() {
  DataVector a{1.0, 2.3, 8.9};
  Parallel::printf("%s\n", a);
}

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

size_t get_vector_size(const size_t l_max) {
  return Spectral::Swsh::number_of_swsh_collocation_points(l_max);
};

std::vector<double> transpose_wt_data(const std::vector<double>& data,
                                      const size_t l_max) {
  const size_t theta_extent = l_max + 1;
  const size_t phi_extent = 2 * l_max + 1;
  auto data_transposed = data;
  // TODO: check this
  transpose(make_not_null(&data_transposed), data, phi_extent, theta_extent);
  return data_transposed;
}

void initialize_j(std::vector<double>& re_j, std::vector<double>& im_j,
                  const std::vector<std::complex<double>>& bondi_beta_spec,
                  const std::vector<std::complex<double>>& bondi_dr_j_spec,
                  const std::vector<std::complex<double>>& bondi_du_r_spec,
                  const std::vector<std::complex<double>>& bondi_h_spec,
                  const std::vector<std::complex<double>>& bondi_j_spec,
                  const std::vector<std::complex<double>>& bondi_q_spec,
                  const std::vector<std::complex<double>>& bondi_r_spec,
                  const std::vector<std::complex<double>>& bondi_u_spec,
                  const std::vector<std::complex<double>>& bondi_w_spec,
                  const size_t l_max, const size_t number_of_radial_points) {
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
             Cce::Tags::BoundaryValue<Cce::Tags::BondiW>>(
      [&bondi_beta_spec, &bondi_dr_j_spec, &bondi_du_r_spec, bondi_h_spec,
       bondi_j_spec, bondi_q_spec, bondi_r_spec, bondi_u_spec, bondi_w_spec](
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
              bondi_w) {
        for (unsigned int i = 0; i < bondi_beta_spec.size(); i++) {
          get(*bondi_beta).data()[i] =
              bondi_beta_spec.at(i) * std::complex<double>(1.0, 0.0);
          get(*bondi_dr_j).data()[i] =
              bondi_dr_j_spec.at(i) * std::complex<double>(1.0, 0.0);
          get(*bondi_du_r).data()[i] =
              bondi_du_r_spec.at(i) * std::complex<double>(1.0, 0.0);
          get(*bondi_h).data()[i] =
              bondi_h_spec.at(i) * std::complex<double>(1.0, 0.0);
          get(*bondi_j).data()[i] =
              bondi_j_spec.at(i) * std::complex<double>(1.0, 0.0);
          get(*bondi_q).data()[i] =
              bondi_q_spec.at(i) * std::complex<double>(1.0, 0.0);
          get(*bondi_r).data()[i] =
              bondi_r_spec.at(i) * std::complex<double>(1.0, 0.0);
          get(*bondi_u).data()[i] =
              bondi_u_spec.at(i) * std::complex<double>(1.0, 0.0);
          get(*bondi_w).data()[i] =
              bondi_w_spec.at(i) * std::complex<double>(1.0, 0.0);
        }
      },
      make_not_null(&spectre_box));

  db::mutate<initialize_action::boundary_value_variables_tag>(
      [](const gsl::not_null<
          initialize_action::boundary_value_variables_tag::type*>
             boundary_variables) {
        Cce::BondiWorldtubeDataManager q;
        auto blah = (*boundary_variables)
                        .reference_subset<
                            Metavariables::cce_boundary_communication_tags>();
        q.populate_hypersurface_boundary_data_spec(make_not_null(&blah));
      },
      make_not_null(&spectre_box));

  /****************************Construct_Bondi_J*************************************/

  db::mutate_apply<typename Cce::InitializeJ::InitializeJ<true>::mutate_tags,
                   typename Cce::InitializeJ::InitializeJ<true>::argument_tags>(
      Cce::InitializeJ::InverseCubic<true>(), make_not_null(&spectre_box));
  std::cout << "final: BondiJ size: "
            << get(get<Cce::Tags::BondiJ>(spectre_box)).size() << std::endl;
  auto& j_initial_data = get(get<Cce::Tags::BondiJ>(spectre_box));
  for (unsigned int i = 0; i < j_initial_data.size(); i++) {
    re_j.push_back(real(j_initial_data.data())[i]);
    im_j.push_back(real(j_initial_data.data())[i]);
  }
}

void ccm_functions(std::vector<double>& re_h, std::vector<double>& im_h,
                   const std::vector<std::complex<double>>& bondi_beta_spec,
                   const std::vector<std::complex<double>>& bondi_dr_j_spec,
                   const std::vector<std::complex<double>>& bondi_du_r_spec,
                   const std::vector<std::complex<double>>& bondi_h_spec,
                   const std::vector<std::complex<double>>& bondi_j_spec,
                   const std::vector<std::complex<double>>& bondi_q_spec,
                   const std::vector<std::complex<double>>& bondi_r_spec,
                   const std::vector<std::complex<double>>& bondi_u_spec,
                   const std::vector<std::complex<double>>& bondi_w_spec,
                   const size_t l_max, const size_t number_of_radial_points) {
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
             Cce::Tags::BoundaryValue<Cce::Tags::BondiW>>(
      [&bondi_beta_spec, &bondi_dr_j_spec, &bondi_du_r_spec, bondi_h_spec,
       bondi_j_spec, bondi_q_spec, bondi_r_spec, bondi_u_spec, bondi_w_spec](
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
              bondi_w) {
        for (unsigned int i = 0; i < bondi_beta_spec.size(); i++) {
          get(*bondi_beta).data()[i] =
              bondi_beta_spec.at(i) * std::complex<double>(1.0, 0.0);
          get(*bondi_dr_j).data()[i] =
              bondi_dr_j_spec.at(i) * std::complex<double>(1.0, 0.0);
          get(*bondi_du_r).data()[i] =
              bondi_du_r_spec.at(i) * std::complex<double>(1.0, 0.0);
          get(*bondi_h).data()[i] =
              bondi_h_spec.at(i) * std::complex<double>(1.0, 0.0);
          get(*bondi_j).data()[i] =
              bondi_j_spec.at(i) * std::complex<double>(1.0, 0.0);
          get(*bondi_q).data()[i] =
              bondi_q_spec.at(i) * std::complex<double>(1.0, 0.0);
          get(*bondi_r).data()[i] =
              bondi_r_spec.at(i) * std::complex<double>(1.0, 0.0);
          get(*bondi_u).data()[i] =
              bondi_u_spec.at(i) * std::complex<double>(1.0, 0.0);
          get(*bondi_w).data()[i] =
              bondi_w_spec.at(i) * std::complex<double>(1.0, 0.0);
        }
      },
      make_not_null(&spectre_box));

  //   std::cout <<
  //   get(db::get<Cce::Tags::BoundaryValue<Cce::Tags::DuRDividedByR>>(
  //                        spectre_box))
  //                    .data()
  //             << std::endl;

  db::mutate<initialize_action::boundary_value_variables_tag>(
      [](const gsl::not_null<
          initialize_action::boundary_value_variables_tag::type*>
             boundary_variables) {
        Cce::BondiWorldtubeDataManager q;
        auto blah = (*boundary_variables)
                        .reference_subset<
                            Metavariables::cce_boundary_communication_tags>();
        q.populate_hypersurface_boundary_data_spec(make_not_null(&blah));
      },
      make_not_null(&spectre_box));

  //   std::cout << get(get<Cce::Tags::BoundaryValue<Cce::Tags::DuRDividedByR>>(
  //                        spectre_box))
  //                    .data()
  //             << std::endl;

  /****************************Construct_Bondi_J*************************************/
  //   std::cout << get(db::get<Cce::Tags::BondiJ>(spectre_box)).data() <<
  //   std::endl;

  db::mutate_apply<typename Cce::InitializeJ::InitializeJ<true>::mutate_tags,
                   typename Cce::InitializeJ::InitializeJ<true>::argument_tags>(
      Cce::InitializeJ::InverseCubic<true>(), make_not_null(&spectre_box));

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

  // I don't have InsertInterpolationScriData and ScriObserveInterpolated
  std::cout << "final: BondiH size: "
            << get(get<Cce::Tags::BondiH>(spectre_box)).size() << std::endl;
  auto& final_h = get(get<Cce::Tags::BondiH>(spectre_box));
  for (unsigned int i = 0; i < final_h.size(); i++) {
    re_h.push_back(real(final_h.data())[i]);
    im_h.push_back(real(final_h.data())[i]);
  }
  // DataVector dv_psi0 = gh_read * 2.;

  // for (unsigned int i = 0; i < dv_psi0.size(); i++) {
  //   psi0.push_back(dv_psi0.at(i));
  // }
}

void std_vector_to_DataVector(
    tnsr::aa<DataVector, 3>& pi, const std::vector<double>& pitt,
    const std::vector<double>& pitx, const std::vector<double>& pity,
    const std::vector<double>& pitz, const std::vector<double>& pixx,
    const std::vector<double>& pixy, const std::vector<double>& pixz,
    const std::vector<double>& piyy, const std::vector<double>& piyz,
    const std::vector<double>& pizz) {
  const auto size = pitt.size();
  DataVector my_pitt{size};
  DataVector my_pitx{size};
  DataVector my_pity{size};
  DataVector my_pitz{size};
  DataVector my_pixx{size};
  DataVector my_pixy{size};
  DataVector my_pixz{size};
  DataVector my_piyy{size};
  DataVector my_piyz{size};
  DataVector my_pizz{size};
  for (unsigned int i = 0; i < pitt.size(); i++) {
    my_pitt[i] = pitt.at(i);
    my_pitx[i] = pitx.at(i);
    my_pity[i] = pity.at(i);
    my_pitz[i] = pitz.at(i);
    my_pixx[i] = pixx.at(i);
    my_pixy[i] = pixy.at(i);
    my_pixz[i] = pixz.at(i);
    my_piyy[i] = piyy.at(i);
    my_piyz[i] = piyz.at(i);
    my_pizz[i] = pizz.at(i);
  }
  get<0, 0>(pi) = my_pitt;
  get<0, 1>(pi) = my_pitx;
  get<0, 2>(pi) = my_pity;
  get<0, 3>(pi) = my_pitz;
  get<1, 1>(pi) = my_pixx;
  get<1, 2>(pi) = my_pixy;
  get<1, 3>(pi) = my_pixz;
  get<2, 2>(pi) = my_piyy;
  get<2, 3>(pi) = my_piyz;
  get<3, 3>(pi) = my_pizz;
}

void gh_to_bondi(
    const std::vector<double>& pitt, const std::vector<double>& pitx,
    const std::vector<double>& pity, const std::vector<double>& pitz,
    const std::vector<double>& pixx, const std::vector<double>& pixy,
    const std::vector<double>& pixz, const std::vector<double>& piyy,
    const std::vector<double>& piyz, const std::vector<double>& pizz) {
  // create_bondi_boundary_data
  tnsr::aa<DataVector, 3> pi;
  tnsr::aa<DataVector, 3> spacetime_metric;
  tnsr::iaa<DataVector, 3> phi;
  std_vector_to_DataVector(pi, pitt, pitx, pity, pitz, pixx, pixy, pixz, piyy,
                           piyz, pizz);
  std_vector_to_DataVector(spacetime_metric, pitt, pitx, pity, pitz, pixx, pixy,
                           pixz, piyy, piyz, pizz);
  std::cout << "pi " << get<0, 0>(pi) << std::endl;
}
