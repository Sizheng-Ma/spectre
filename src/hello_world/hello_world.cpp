// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "hello_world/hello_world.hpp"

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
                  const size_t l_max, const size_t number_of_radial_points,
                  const std::vector<std::vector<double>>& spacetime_metric,
                  const std::vector<std::vector<double>>& pi,
                  const std::vector<std::vector<std::vector<double>>>& phi) {
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
      [&spacetime_metric, &phi, &pi, &l_max](
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
        gh_to_bondi(*bondi_beta, *bondi_dr_j, *bondi_du_r, *bondi_h, *bondi_j,
                    *bondi_q, *bondi_r, *bondi_u, *bondi_w, spacetime_metric,
                    pi, phi, l_max);
        // for (unsigned int i = 0; i < bondi_beta_spec.size(); i++) {
        //   get(*bondi_beta).data()[i] =
        //       bondi_beta_spec.at(i) * std::complex<double>(1.0, 0.0);
        //   get(*bondi_dr_j).data()[i] =
        //       bondi_dr_j_spec.at(i) * std::complex<double>(1.0, 0.0);
        //   get(*bondi_du_r).data()[i] =
        //       bondi_du_r_spec.at(i) * std::complex<double>(1.0, 0.0);
        //   get(*bondi_h).data()[i] =
        //       bondi_h_spec.at(i) * std::complex<double>(1.0, 0.0);
        //   get(*bondi_j).data()[i] =
        //       bondi_j_spec.at(i) * std::complex<double>(1.0, 0.0);
        //   get(*bondi_q).data()[i] =
        //       bondi_q_spec.at(i) * std::complex<double>(1.0, 0.0);
        //   get(*bondi_r).data()[i] =
        //       bondi_r_spec.at(i) * std::complex<double>(1.0, 0.0);
        //   get(*bondi_u).data()[i] =
        //       bondi_u_spec.at(i) * std::complex<double>(1.0, 0.0);
        //   get(*bondi_w).data()[i] =
        //       bondi_w_spec.at(i) * std::complex<double>(1.0, 0.0);
        // }
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
                   const size_t l_max, const size_t number_of_radial_points,
                   const std::vector<std::vector<double>>& spacetime_metric,
                   const std::vector<std::vector<double>>& pi,
                   const std::vector<std::vector<std::vector<double>>>& phi) {
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
      [&spacetime_metric, &phi, &pi, &l_max](
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
        gh_to_bondi(*bondi_beta, *bondi_dr_j, *bondi_du_r, *bondi_h, *bondi_j,
                    *bondi_q, *bondi_r, *bondi_u, *bondi_w, spacetime_metric,
                    pi, phi, l_max);
        // for (unsigned int i = 0; i < bondi_beta_spec.size(); i++) {
        //   //   get(*bondi_beta).data()[i] =
        //   //       bondi_beta_spec.at(i) * std::complex<double>(1.0, 0.0);
        //   //   get(*bondi_dr_j).data()[i] =
        //   //       bondi_dr_j_spec.at(i) * std::complex<double>(1.0, 0.0);
        //   //   get(*bondi_du_r).data()[i] =
        //   //       bondi_du_r_spec.at(i) * std::complex<double>(1.0, 0.0);
        //   //   get(*bondi_h).data()[i] =
        //   //       bondi_h_spec.at(i) * std::complex<double>(1.0, 0.0);
        //   //   get(*bondi_j).data()[i] =
        //   //       bondi_j_spec.at(i) * std::complex<double>(1.0, 0.0);
        //   //   get(*bondi_q).data()[i] =
        //   //       bondi_q_spec.at(i) * std::complex<double>(1.0, 0.0);
        //   //   get(*bondi_r).data()[i] =
        //   //       bondi_r_spec.at(i) * std::complex<double>(1.0, 0.0);
        //   //   get(*bondi_u).data()[i] =
        //   //       bondi_u_spec.at(i) * std::complex<double>(1.0, 0.0);
        //   //   get(*bondi_w).data()[i] =
        //   //       bondi_w_spec.at(i) * std::complex<double>(1.0, 0.0);
        // }
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

void std_vector_to_DataVector(tnsr::aa<DataVector, 3>& pi,
                              const std::vector<std::vector<double>>& data) {
  const auto size = data.at(0).size();
  for (unsigned int i = 0; i < size; i++) {
    get<0, 0>(pi)[i] = data.at(0)[i];
    get<0, 1>(pi)[i] = data.at(1)[i];
    get<0, 2>(pi)[i] = data.at(2)[i];
    get<0, 3>(pi)[i] = data.at(3)[i];
    get<1, 1>(pi)[i] = data.at(4)[i];
    get<1, 2>(pi)[i] = data.at(5)[i];
    get<1, 3>(pi)[i] = data.at(6)[i];
    get<2, 2>(pi)[i] = data.at(7)[i];
    get<2, 3>(pi)[i] = data.at(8)[i];
    get<3, 3>(pi)[i] = data.at(9)[i];
  }
}

void tri_std_vector_to_DataVector(
    tnsr::iaa<DataVector, 3>& pi,
    const std::vector<std::vector<std::vector<double>>>& data) {
  const auto size = data.at(0).at(0).size();
  for (size_t ijj = 0; ijj < 3; ++ijj) {
    for (size_t i = 0; i < size; i++) {
      pi.get(ijj, 0, 0)[i] = data.at(ijj).at(0)[i];
      pi.get(ijj, 0, 1)[i] = data.at(ijj).at(1)[i];
      pi.get(ijj, 0, 2)[i] = data.at(ijj).at(2)[i];
      pi.get(ijj, 0, 3)[i] = data.at(ijj).at(3)[i];
      pi.get(ijj, 1, 1)[i] = data.at(ijj).at(4)[i];
      pi.get(ijj, 1, 2)[i] = data.at(ijj).at(5)[i];
      pi.get(ijj, 1, 3)[i] = data.at(ijj).at(6)[i];
      pi.get(ijj, 2, 2)[i] = data.at(ijj).at(7)[i];
      pi.get(ijj, 2, 3)[i] = data.at(ijj).at(8)[i];
      pi.get(ijj, 3, 3)[i] = data.at(ijj).at(9)[i];
    }
  }
}

void gh_to_bondi(Scalar<SpinWeighted<ComplexDataVector, 0>>& beta,
                 Scalar<SpinWeighted<ComplexDataVector, 2>>& dr_j,
                 Scalar<SpinWeighted<ComplexDataVector, 0>>& du_r,
                 Scalar<SpinWeighted<ComplexDataVector, 2>>& bondih,
                 Scalar<SpinWeighted<ComplexDataVector, 2>>& bondij,
                 Scalar<SpinWeighted<ComplexDataVector, 1>>& bondiq,
                 Scalar<SpinWeighted<ComplexDataVector, 0>>& bondir,
                 Scalar<SpinWeighted<ComplexDataVector, 1>>& bondiu,
                 Scalar<SpinWeighted<ComplexDataVector, 0>>& bondiw,
                 const std::vector<std::vector<double>>& spacetime_metric,
                 const std::vector<std::vector<double>>& pi,
                 const std::vector<std::vector<std::vector<double>>>& phi,
                 const size_t l_max) {
  // create_bondi_boundary_data
  const auto size = pi.at(0).size();
  tnsr::aa<DataVector, 3> pi_datavector{size};
  tnsr::aa<DataVector, 3> spacetime_metric_datavector{size};
  tnsr::iaa<DataVector, 3> phi_datavector{size};
  std_vector_to_DataVector(pi_datavector, pi);
  std_vector_to_DataVector(spacetime_metric_datavector, spacetime_metric);
  tri_std_vector_to_DataVector(phi_datavector, phi);
  using initialize_action =
      Cce::Actions::InitializeCharacteristicEvolutionVariables<
          MyEvolutionMetavars>;
  auto spectre_box = db::create<
      db::AddSimpleTags<initialize_action::simple_tags_for_evolution>>();

  size_t boundary_size = get_vector_size(l_max);
  using boundary_value_variables_tag = ::Tags::Variables<tmpl::append<
      typename MyEvolutionMetavars::cce_boundary_communication_tags,
      typename MyEvolutionMetavars::cce_gauge_boundary_tags>>;
  Initialization::mutate_assign<tmpl::list<boundary_value_variables_tag>>(
      make_not_null(&spectre_box),
      typename boundary_value_variables_tag::type{boundary_size});

  db::mutate<initialize_action::boundary_value_variables_tag>(
      [&l_max, &phi_datavector, &pi_datavector, &spacetime_metric_datavector](
          const gsl::not_null<
              initialize_action::boundary_value_variables_tag::type*>
              boundary_variables) {
        auto blah =
            (*boundary_variables)
                .reference_subset<
                    MyEvolutionMetavars::cce_boundary_communication_tags>();
        Cce::create_bondi_boundary_data(make_not_null(&blah), phi_datavector,
                                        pi_datavector,
                                        spacetime_metric_datavector, 10, l_max);
      },
      make_not_null(&spectre_box));

  beta = get<Cce::Tags::BoundaryValue<Cce::Tags::BondiBeta>>(spectre_box);
  dr_j = get<Cce::Tags::BoundaryValue<Cce::Tags::Dr<Cce::Tags::BondiJ>>>(
      spectre_box);
  du_r = get<Cce::Tags::BoundaryValue<Cce::Tags::Du<Cce::Tags::BondiR>>>(
      spectre_box);
  bondih = get<Cce::Tags::BoundaryValue<Cce::Tags::BondiH>>(spectre_box);
  bondij = get<Cce::Tags::BoundaryValue<Cce::Tags::BondiJ>>(spectre_box);
  bondiq = get<Cce::Tags::BoundaryValue<Cce::Tags::BondiQ>>(spectre_box);
  bondir = get<Cce::Tags::BoundaryValue<Cce::Tags::BondiR>>(spectre_box);
  bondiu = get<Cce::Tags::BoundaryValue<Cce::Tags::BondiU>>(spectre_box);
  bondiw = get<Cce::Tags::BoundaryValue<Cce::Tags::BondiW>>(spectre_box);
}
