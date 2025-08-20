// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "hello_world/hello_world.hpp"

#include <boost/preprocessor.hpp>
#include <complex>
#include <iostream>
#include <queue>
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
#include "Evolution/Systems/Cce/ScriPlusInterpolationManager.hpp"
#include "Evolution/Systems/Cce/SwshDerivatives.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Evolution/Systems/Cce/WorldtubeBufferUpdater.hpp"
#include "Evolution/Systems/Cce/WorldtubeDataManager.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/CubicSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/LinearSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshTags.hpp"
#include "Options/Options.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Time/StepChoosers/Factory.hpp"
#include "Time/TimeSteppers/Factory.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Utilities/MakeString.hpp"

void std_vector_to_DataVector(tnsr::aa<DataVector, 3>& pi,
                              const std::vector<std::vector<double>>& data);
void std_vector_to_DataVector(DataVector& pi, const std::vector<double>& data);
void tri_std_vector_to_DataVector(
    tnsr::iaa<DataVector, 3>& pi,
    const std::vector<std::vector<std::vector<double>>>& data);
void gh_to_bondi(Scalar<SpinWeighted<ComplexDataVector, 0>>& beta,
                 Scalar<SpinWeighted<ComplexDataVector, 2>>& dr_j,
                 Scalar<SpinWeighted<ComplexDataVector, 2>>& du_j,
                 Scalar<SpinWeighted<ComplexDataVector, 0>>& du_r,
                 Scalar<SpinWeighted<ComplexDataVector, 2>>& bondih,
                 Scalar<SpinWeighted<ComplexDataVector, 2>>& bondij,
                 Scalar<SpinWeighted<ComplexDataVector, 1>>& bondiq,
                 Scalar<SpinWeighted<ComplexDataVector, 0>>& bondir,
                 Scalar<SpinWeighted<ComplexDataVector, 1>>& bondiu,
                 Scalar<SpinWeighted<ComplexDataVector, 1>>& dr_u,
                 Scalar<SpinWeighted<ComplexDataVector, 0>>& bondiw,
                 Scalar<SpinWeighted<ComplexDataVector, 0>>& du_r_r,
                 Scalar<SpinWeighted<ComplexDataVector, 0>>& spec_norm,
                 const std::vector<std::vector<double>>& spacetime_metric,
                 const std::vector<std::vector<double>>& pi,
                 const std::vector<std::vector<std::vector<double>>>& phi,
                 const size_t l_max, const double radius);

void insert_bondi_variables(
    const gsl::not_null<Cce::Tags::BoundaryValue<Cce::Tags::BondiBeta>::type*>
        bondi_beta,
    const gsl::not_null<
        Cce::Tags::BoundaryValue<Cce::Tags::Dr<Cce::Tags::BondiJ>>::type*>
        bondi_dr_j,
    const gsl::not_null<
        Cce::Tags::BoundaryValue<Cce::Tags::Du<Cce::Tags::BondiR>>::type*>
        bondi_du_r,
    const gsl::not_null<Cce::Tags::BoundaryValue<Cce::Tags::BondiH>::type*>
        bondi_h,
    const gsl::not_null<Cce::Tags::BoundaryValue<Cce::Tags::BondiJ>::type*>
        bondi_j,
    const gsl::not_null<Cce::Tags::BoundaryValue<Cce::Tags::BondiQ>::type*>
        bondi_q,
    const gsl::not_null<Cce::Tags::BoundaryValue<Cce::Tags::BondiR>::type*>
        bondi_r,
    const gsl::not_null<Cce::Tags::BoundaryValue<Cce::Tags::BondiU>::type*>
        bondi_u,
    const gsl::not_null<Cce::Tags::BoundaryValue<Cce::Tags::BondiW>::type*>
        bondi_w,
    const gsl::not_null<
        Cce::Tags::BoundaryValue<Cce::Tags::Dr<Cce::Tags::BondiU>>::type*>
        dr_u,
    const gsl::not_null<
        Cce::Tags::BoundaryValue<Cce::Tags::Du<Cce::Tags::BondiJ>>::type*>
        du_j,
    const gsl::not_null<
        Cce::Tags::BoundaryValue<Cce::Tags::DuRDividedByR>::type*>
        du_r_r,
    const gsl::not_null<
        Cce::Tags::BoundaryValue<Cce::Tags::SpECNormalization>::type*>
        spec_norm,
    const size_t l_max, std::vector<double>& ccm_sender) {
  size_t surface_size = get_vector_size(l_max);
  size_t total_size_ccm_sender = ccm_sender.size();
  /********************************spec_norm_new*************************/
  std::vector<double> spec_norm_new;
  spec_norm_new.insert(spec_norm_new.end(), ccm_sender.end() - surface_size,
                       ccm_sender.end());
  for (size_t xx = 0; xx < surface_size; xx++) {
    get(*spec_norm).data()[xx] = spec_norm_new[xx];
  }

  /*********************************du_r_r******************************/
  total_size_ccm_sender -= surface_size;
  ccm_sender.resize(total_size_ccm_sender);
  std::vector<double> du_r_r_new;
  du_r_r_new.insert(du_r_r_new.end(), ccm_sender.end() - surface_size,
                    ccm_sender.end());
  for (size_t xx = 0; xx < surface_size; xx++) {
    get(*du_r_r).data()[xx] = du_r_r_new[xx];
  }

  /***********************************du_r*******************************/
  total_size_ccm_sender -= surface_size;
  ccm_sender.resize(total_size_ccm_sender);
  std::vector<double> du_r_new;
  du_r_new.insert(du_r_new.end(), ccm_sender.end() - surface_size,
                  ccm_sender.end());
  for (size_t xx = 0; xx < surface_size; xx++) {
    get(*bondi_du_r).data()[xx] = du_r_new[xx];
  }

  /*******************************bondi_r******************************/
  total_size_ccm_sender -= surface_size;
  ccm_sender.resize(total_size_ccm_sender);
  std::vector<double> bondir_new;
  bondir_new.insert(bondir_new.end(), ccm_sender.end() - surface_size,
                    ccm_sender.end());
  for (size_t xx = 0; xx < surface_size; xx++) {
    get(*bondi_r).data()[xx] = bondir_new[xx];
  }

  /*******************************du_j******************************/
  total_size_ccm_sender -= surface_size;
  ccm_sender.resize(total_size_ccm_sender);
  std::vector<double> du_j_im_new;
  du_j_im_new.insert(du_j_im_new.end(), ccm_sender.end() - surface_size,
                     ccm_sender.end());

  total_size_ccm_sender -= surface_size;
  ccm_sender.resize(total_size_ccm_sender);
  std::vector<double> du_j_re_new;
  du_j_re_new.insert(du_j_re_new.end(), ccm_sender.end() - surface_size,
                     ccm_sender.end());

  for (size_t xx = 0; xx < surface_size; xx++) {
    get(*du_j).data()[xx] =
        std::complex<double>(du_j_re_new[xx], du_j_im_new[xx]);
  }

  /*******************************h******************************/
  total_size_ccm_sender -= surface_size;
  ccm_sender.resize(total_size_ccm_sender);
  std::vector<double> bondih_im_new;
  bondih_im_new.insert(bondih_im_new.end(), ccm_sender.end() - surface_size,
                       ccm_sender.end());

  total_size_ccm_sender -= surface_size;
  ccm_sender.resize(total_size_ccm_sender);
  std::vector<double> bondih_re_new;
  bondih_re_new.insert(bondih_re_new.end(), ccm_sender.end() - surface_size,
                       ccm_sender.end());

  for (size_t xx = 0; xx < surface_size; xx++) {
    get(*bondi_h).data()[xx] =
        std::complex<double>(bondih_re_new[xx], bondih_im_new[xx]);
  }

  /*******************************dr_j******************************/
  total_size_ccm_sender -= surface_size;
  ccm_sender.resize(total_size_ccm_sender);
  std::vector<double> dr_j_im_new;
  dr_j_im_new.insert(dr_j_im_new.end(), ccm_sender.end() - surface_size,
                     ccm_sender.end());

  total_size_ccm_sender -= surface_size;
  ccm_sender.resize(total_size_ccm_sender);
  std::vector<double> dr_j_re_new;
  dr_j_re_new.insert(dr_j_re_new.end(), ccm_sender.end() - surface_size,
                     ccm_sender.end());

  for (size_t xx = 0; xx < surface_size; xx++) {
    get(*bondi_dr_j).data()[xx] =
        std::complex<double>(dr_j_re_new[xx], dr_j_im_new[xx]);
  }

  /*******************************j******************************/
  total_size_ccm_sender -= surface_size;
  ccm_sender.resize(total_size_ccm_sender);
  std::vector<double> j_im_new;
  j_im_new.insert(j_im_new.end(), ccm_sender.end() - surface_size,
                  ccm_sender.end());

  total_size_ccm_sender -= surface_size;
  ccm_sender.resize(total_size_ccm_sender);
  std::vector<double> j_re_new;
  j_re_new.insert(j_re_new.end(), ccm_sender.end() - surface_size,
                  ccm_sender.end());

  for (size_t xx = 0; xx < surface_size; xx++) {
    get(*bondi_j).data()[xx] = std::complex<double>(j_re_new[xx], j_im_new[xx]);
  }

  /*******************************w******************************/
  total_size_ccm_sender -= surface_size;
  ccm_sender.resize(total_size_ccm_sender);
  std::vector<double> w_new;
  w_new.insert(w_new.end(), ccm_sender.end() - surface_size, ccm_sender.end());
  for (size_t xx = 0; xx < surface_size; xx++) {
    get(*bondi_w).data()[xx] = w_new[xx];
  }

  /*******************************q******************************/
  total_size_ccm_sender -= surface_size;
  ccm_sender.resize(total_size_ccm_sender);
  std::vector<double> q_im_new;
  q_im_new.insert(q_im_new.end(), ccm_sender.end() - surface_size,
                  ccm_sender.end());

  total_size_ccm_sender -= surface_size;
  ccm_sender.resize(total_size_ccm_sender);
  std::vector<double> q_re_new;
  q_re_new.insert(q_re_new.end(), ccm_sender.end() - surface_size,
                  ccm_sender.end());

  for (size_t xx = 0; xx < surface_size; xx++) {
    get(*bondi_q).data()[xx] = std::complex<double>(q_re_new[xx], q_im_new[xx]);
  }

  /*******************************dr_u******************************/
  total_size_ccm_sender -= surface_size;
  ccm_sender.resize(total_size_ccm_sender);
  std::vector<double> dr_u_im_new;
  dr_u_im_new.insert(dr_u_im_new.end(), ccm_sender.end() - surface_size,
                     ccm_sender.end());

  total_size_ccm_sender -= surface_size;
  ccm_sender.resize(total_size_ccm_sender);
  std::vector<double> dr_u_re_new;
  dr_u_re_new.insert(dr_u_re_new.end(), ccm_sender.end() - surface_size,
                     ccm_sender.end());

  for (size_t xx = 0; xx < surface_size; xx++) {
    get(*dr_u).data()[xx] =
        std::complex<double>(dr_u_re_new[xx], dr_u_im_new[xx]);
  }

  /*******************************u******************************/
  total_size_ccm_sender -= surface_size;
  ccm_sender.resize(total_size_ccm_sender);
  std::vector<double> u_im_new;
  u_im_new.insert(u_im_new.end(), ccm_sender.end() - surface_size,
                  ccm_sender.end());

  total_size_ccm_sender -= surface_size;
  ccm_sender.resize(total_size_ccm_sender);
  std::vector<double> u_re_new;
  u_re_new.insert(u_re_new.end(), ccm_sender.end() - surface_size,
                  ccm_sender.end());

  for (size_t xx = 0; xx < surface_size; xx++) {
    get(*bondi_u).data()[xx] = std::complex<double>(u_re_new[xx], u_im_new[xx]);
  }

  /*******************************beta******************************/
  total_size_ccm_sender -= surface_size;
  ccm_sender.resize(total_size_ccm_sender);
  std::vector<double> beta_new = std::move(ccm_sender);
  for (size_t xx = 0; xx < surface_size; xx++) {
    get(*bondi_beta).data()[xx] = beta_new[xx];
  }
  ASSERT(total_size_ccm_sender == surface_size, "wrong size");
}

// Charm looks for this function but since we build without a main function
// or main module we just have it be empty
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

template <typename Tag, int Spin>
static void transform_and_write_new(
    const ComplexDataVector& data, const double time,
    const gsl::not_null<ComplexModalVector*> goldberg_mode_buffer,
    const gsl::not_null<std::vector<double>*> data_to_write_buffer,
    const size_t l_max, const size_t observation_l_max) {
  const SpinWeighted<ComplexDataVector, Spin> to_transform;
  make_const_view(make_not_null(&to_transform.data()), data, 0, data.size());
  SpinWeighted<ComplexModalVector, Spin> goldberg_modes;
  goldberg_modes.set_data_ref(goldberg_mode_buffer);
  Spectral::Swsh::libsharp_to_goldberg_modes(
      make_not_null(&goldberg_modes),
      Spectral::Swsh::swsh_transform(l_max, 1, to_transform), l_max);

  (*data_to_write_buffer)[0] = time;
  for (size_t i = 0; i < square(observation_l_max + 1); ++i) {
    (*data_to_write_buffer)[2 * i + 1] = real(goldberg_modes.data()[i]);
    (*data_to_write_buffer)[2 * i + 2] = imag(goldberg_modes.data()[i]);
  }
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

void transpose_wt_data(std::vector<double>& data_transposed,
                       const std::vector<double>& data, const size_t l_max) {
  const size_t theta_extent = l_max + 1;
  const size_t phi_extent = 2 * l_max + 1;
  data_transposed.resize(data.size());
  transpose(make_not_null(&data_transposed), data, theta_extent, phi_extent);
}

std::vector<double> transpose_ccm_data(const std::vector<double>& data,
                                       const size_t l_max) {
  const size_t theta_extent = l_max + 1;
  const size_t phi_extent = 2 * l_max + 1;
  auto data_transposed = data;
  // TODO: check this
  transpose(make_not_null(&data_transposed), data, phi_extent, theta_extent);
  return data_transposed;
}

void initialize_j(std::vector<std::complex<double>>& finalbondij,
                  std::vector<double>& cauchy_x, std::vector<double>& cauchy_y,
                  std::vector<double>& cauchy_z,
                  std::vector<double>& inertial_x,
                  std::vector<double>& inertial_y,
                  std::vector<double>& inertial_z,
                  std::vector<double>& ccm_sender, const size_t l_max,
                  const size_t number_of_radial_points, const double radius) {
  // const DataVector gh_read{const_cast<double*>(gh.data()), gh.size()};

  const size_t filter_l_max = l_max - 2;
  const size_t scri_interpolation_order = 5;
  const double radial_filter_alpha = 35.0;
  const size_t radial_filter_half_power = 24;

  using Metavariables = MyEvolutionMetavars;

  using initialize_action =
      Cce::Actions::InitializeCharacteristicEvolutionVariables<Metavariables>;
  using simple_tags_for_evolution =
      initialize_action::simple_tags_for_evolution;
  using from_cache =
      tmpl::list<Cce::InitializationTags::ScriInterpolationOrder,
                 Cce::Tags::LMax, Cce::Tags::NumberOfRadialPoints,
                 Cce::Tags::FilterLMax, Cce::Tags::RadialFilterAlpha,
                 Cce::Tags::RadialFilterHalfPower>;
  using simple_tags = tmpl::append<from_cache, simple_tags_for_evolution>;

  auto spectre_box = db::create<db::AddSimpleTags<simple_tags>>();

  /****************************Initialization*************************************/
  Initialization::mutate_assign<from_cache>(
      make_not_null(&spectre_box),
      Cce::InitializationTags::ScriInterpolationOrder::type{
          scri_interpolation_order},
      Cce::Tags::LMax::type{l_max},
      Cce::Tags::NumberOfRadialPoints::type{number_of_radial_points},
      Cce::Tags::FilterLMax::type{filter_l_max},
      Cce::Tags::RadialFilterAlpha::type{radial_filter_alpha},
      Cce::Tags::RadialFilterHalfPower::type{radial_filter_half_power});
  initialize_action::initialize_impl(make_not_null(&spectre_box));

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
             Cce::Tags::BoundaryValue<Cce::Tags::DuRDividedByR>,
             Cce::Tags::BoundaryValue<Cce::Tags::SpECNormalization>>(
      [&l_max, &ccm_sender](
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
              du_r_r,
          const gsl::not_null<
              Cce::Tags::BoundaryValue<Cce::Tags::SpECNormalization>::type*>
              spec_norm) {
        // gh_to_bondi(*bondi_beta, *bondi_dr_j, *du_j, *bondi_du_r, *bondi_h,
        //             *bondi_j, *bondi_q, *bondi_r, *bondi_u, *dr_u, *bondi_w,
        //             *du_r_r, *spec_norm, spacetime_metric, pi, phi, l_max,
        //             radius);
        insert_bondi_variables(bondi_beta, bondi_dr_j, bondi_du_r, bondi_h,
                               bondi_j, bondi_q, bondi_r, bondi_u, bondi_w,
                               dr_u, du_j, du_r_r, spec_norm, l_max,
                               ccm_sender);
      },
      make_not_null(&spectre_box));

  //   db::mutate<initialize_action::boundary_value_variables_tag>(
  //       [](const gsl::not_null<
  //           initialize_action::boundary_value_variables_tag::type*>
  //              boundary_variables) {
  //         Cce::BondiWorldtubeDataManager q;
  //         auto blah = (*boundary_variables)
  //                         .reference_subset<
  //                             Metavariables::cce_boundary_communication_tags>();
  //         q.populate_hypersurface_boundary_data_spec(make_not_null(&blah));
  //       },
  //       make_not_null(&spectre_box));

  /****************************Construct_Bondi_J*************************************/

  db::mutate_apply<typename Cce::InitializeJ::InitializeJ<true>::mutate_tags,
                   typename Cce::InitializeJ::InitializeJ<true>::argument_tags>(
      Cce::InitializeJ::InverseCubic<true>(), make_not_null(&spectre_box));
  //   std::cout << "final: BondiJ size: "
  //             << get(get<Cce::Tags::BondiJ>(spectre_box)).size() <<
  //             std::endl;
  auto& j_initial_data = get(get<Cce::Tags::BondiJ>(spectre_box));
  for (unsigned int i = 0; i < j_initial_data.size(); i++) {
    finalbondij.push_back(j_initial_data.data()[i]);
  }

  /****************************Construct_coordinates*************************************/
  auto& cauchy_cart = db::get<Cce::Tags::CauchyCartesianCoords>(spectre_box);
  auto& inertial_cart =
      db::get<Cce::Tags::PartiallyFlatCartesianCoords>(spectre_box);
  for (unsigned int i = 0; i < cauchy_cart.get(0).size(); i++) {
    cauchy_x.push_back(cauchy_cart.get(0)[i]);
    cauchy_y.push_back(cauchy_cart.get(1)[i]);
    cauchy_z.push_back(cauchy_cart.get(2)[i]);
    inertial_x.push_back(inertial_cart.get(0)[i]);
    inertial_y.push_back(inertial_cart.get(1)[i]);
    inertial_z.push_back(inertial_cart.get(2)[i]);
  }
}

void ccm_functions(
    std::vector<std::complex<double>>& finalbondih,
    std::vector<double>& dt_cauchy_x, std::vector<double>& dt_cauchy_y,
    std::vector<double>& dt_cauchy_z, std::vector<double>& dt_inertial_x,
    std::vector<double>& dt_inertial_y, std::vector<double>& dt_inertial_z,
    std::vector<std::complex<double>>& eth_inertial_retarded_time,
    std::vector<std::complex<double>>& news,
    std::vector<std::complex<double>>& strain,
    std::vector<std::complex<double>>& psi0,
    std::vector<std::complex<double>>& psi1,
    std::vector<std::complex<double>>& psi2,
    std::vector<std::complex<double>>& psi3,
    std::vector<std::complex<double>>& psi4, std::vector<double>& dt_u_scri,
    std::vector<std::complex<double>>& psi0_ccm,
    // std::vector<std::complex<double>>& wxx_test_for_spec,
    std::vector<double>& coeff_theta,
    std::vector<std::complex<double>>& coeff_phi, double& ccmconstraintomega,
    double& ccmconstraintc, double& ccmconstraintd,
    std::vector<double>& ccm_sender, const size_t l_max,
    const size_t number_of_radial_points, const double radius,
    const std::vector<std::complex<double>>& bondij,
    const std::vector<std::vector<double>>& cauchy_cart,
    const std::vector<std::vector<double>>& inertial_cart,
    const std::vector<double>& intertial_time) {
  // const DataVector gh_read{const_cast<double*>(gh.data()), gh.size()};

  // TODO this is hardcoded
  const size_t filter_l_max = l_max - 2;
  const size_t scri_interpolation_order = 5;
  const double radial_filter_alpha = 35.0;
  const size_t radial_filter_half_power = 24;

  using Metavariables = MyEvolutionMetavars;

  using initialize_action =
      Cce::Actions::InitializeCharacteristicEvolutionVariables<Metavariables>;
  using simple_tags_for_evolution =
      initialize_action::simple_tags_for_evolution;
  using from_cache =
      tmpl::list<Cce::InitializationTags::ScriInterpolationOrder,
                 Cce::Tags::LMax, Cce::Tags::NumberOfRadialPoints,
                 Cce::Tags::FilterLMax, Cce::Tags::RadialFilterAlpha,
                 Cce::Tags::RadialFilterHalfPower>;
  using simple_tags = tmpl::append<from_cache, simple_tags_for_evolution>;

  auto spectre_box = db::create<db::AddSimpleTags<simple_tags>>();

  /****************************Initialization*************************************/
  Initialization::mutate_assign<from_cache>(
      make_not_null(&spectre_box),
      Cce::InitializationTags::ScriInterpolationOrder::type{
          scri_interpolation_order},
      Cce::Tags::LMax::type{l_max},
      Cce::Tags::NumberOfRadialPoints::type{number_of_radial_points},
      Cce::Tags::FilterLMax::type{filter_l_max},
      Cce::Tags::RadialFilterAlpha::type{radial_filter_alpha},
      Cce::Tags::RadialFilterHalfPower::type{radial_filter_half_power});
  initialize_action::initialize_impl(make_not_null(&spectre_box));

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
             Cce::Tags::BoundaryValue<Cce::Tags::DuRDividedByR>,
             Cce::Tags::BoundaryValue<Cce::Tags::SpECNormalization>>(
      [&l_max, &ccm_sender](
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
              du_r_r,
          const gsl::not_null<
              Cce::Tags::BoundaryValue<Cce::Tags::SpECNormalization>::type*>
              spec_norm) {
        // gh_to_bondi(*bondi_beta, *bondi_dr_j, *du_j, *bondi_du_r, *bondi_h,
        //             *bondi_j, *bondi_q, *bondi_r, *bondi_u, *dr_u, *bondi_w,
        //             *du_r_r, *spec_norm, spacetime_metric, pi, phi, l_max,
        //             radius);

        insert_bondi_variables(bondi_beta, bondi_dr_j, bondi_du_r, bondi_h,
                               bondi_j, bondi_q, bondi_r, bondi_u, bondi_w,
                               dr_u, du_j, du_r_r, spec_norm, l_max,
                               ccm_sender);

      },
      make_not_null(&spectre_box));

  /****************************Construct_Bondi_J*************************************/
  //   std::cout << get(db::get<Cce::Tags::BondiJ>(spectre_box)).data() <<
  //   std::endl;

  //   db::mutate_apply<typename
  //   Cce::InitializeJ::InitializeJ<true>::mutate_tags,
  //                    typename
  //                    Cce::InitializeJ::InitializeJ<true>::argument_tags>(
  //       Cce::InitializeJ::InverseCubic<true>(), make_not_null(&spectre_box));
  db::mutate<Cce::Tags::BondiJ, Cce::Tags::CauchyCartesianCoords,
             Cce::Tags::PartiallyFlatCartesianCoords,
             Cce::Tags::InertialRetardedTime>(
      [&cauchy_cart, &inertial_cart, &bondij, &intertial_time](
          const gsl::not_null<Cce::Tags::BondiJ::type*> bondi_j,
          const gsl::not_null<Cce::Tags::CauchyCartesianCoords::type*>
              spectre_cauchy_cart,
          const gsl::not_null<Cce::Tags::PartiallyFlatCartesianCoords::type*>
              spectre_inertial_cart,
          const gsl::not_null<Cce::Tags::InertialRetardedTime::type*>
              inertial_retarded_time_assign) {
        for (size_t kkd = 0; kkd < 3; kkd++) {
          std::memcpy((*spectre_cauchy_cart).get(kkd).data(),
                      cauchy_cart[kkd].data(),
                      sizeof(double) * cauchy_cart[kkd].size());
        }

        for (size_t kkd = 0; kkd < 3; kkd++) {
          std::memcpy((*spectre_inertial_cart).get(kkd).data(),
                      inertial_cart[kkd].data(),
                      sizeof(double) * inertial_cart[kkd].size());
        }
        std::memcpy(get(*bondi_j).data().data(), bondij.data(),
                    sizeof(std::complex<double>) * bondij.size());
        std::memcpy(get(*inertial_retarded_time_assign).data(),
                    intertial_time.data(),
                    sizeof(double) * intertial_time.size());
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

  /****************************CalculatePsi0AndDerivAtInnerBoundary*************************************/
  tmpl::for_each<Cce::Actions::CalculatePsi0AndDerivAtInnerBoundary::mutators>(
      [&spectre_box](auto mutator_v) {
        using mutator = typename decltype(mutator_v)::type;
        db::mutate_apply<mutator>(make_not_null(&spectre_box));
      });

  db::mutate<Cce::Tags::BoundaryValue<Cce::Tags::TetradCoeffTheta>>(
      [&radius](const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
                    theta_tetrad) { get(*theta_tetrad).data() *= radius; },
      make_not_null(&spectre_box));
  db::mutate<Cce::Tags::BoundaryValue<Cce::Tags::TetradCoeffTheta>>(
      [&l_max](const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
                   theta_tetrad) {
        Spectral::Swsh::filter_swsh_boundary_quantity(
            make_not_null(&get(*theta_tetrad)), l_max, l_max - 3);
      },
      make_not_null(&spectre_box));
  db::mutate<Cce::Tags::BoundaryValue<Cce::Tags::TetradCoeffPhi>>(
      [&radius](const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
                    phi_tetrad) { get(*phi_tetrad).data() *= radius; },
      make_not_null(&spectre_box));
  db::mutate<Cce::Tags::BoundaryValue<Cce::Tags::TetradCoeffPhi>>(
      [&l_max](const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
                   phi_tetrad) {
        Spectral::Swsh::filter_swsh_boundary_quantity(
            make_not_null(&get(*phi_tetrad)), l_max, l_max - 3);
      },
      make_not_null(&spectre_box));
  db::mutate<Cce::Tags::BoundaryValue<Cce::Tags::Psi0Match>>(
      [&l_max](const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
                   psi_0_bound) {
        Spectral::Swsh::filter_swsh_boundary_quantity(
            make_not_null(&get(*psi_0_bound)), l_max, l_max - 3);
      },
      make_not_null(&spectre_box));

  /****************************hypersurface_computation*************************************/
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

  /*************************for_test*****************************/

  // db::mutate_apply<Cce::GetWijForTest>(make_not_null(&spectre_box));
  /*************************after_cce*****************************/
  //   std::cout << "final: BondiH size: "
  //             << get(get<Cce::Tags::BondiH>(spectre_box)).size() <<
  //             std::endl;
  auto& final_h = get(get<Cce::Tags::BondiH>(spectre_box));
  finalbondih.insert(finalbondih.end(), final_h.data().begin(),
                     final_h.data().end());

  auto& dt_cauchy_cart =
      db::get<::Tags::dt<Cce::Tags::CauchyCartesianCoords>>(spectre_box);
  auto& dt_inertial_cart =
      db::get<::Tags::dt<Cce::Tags::PartiallyFlatCartesianCoords>>(spectre_box);

  dt_cauchy_x.insert(dt_cauchy_x.end(), dt_cauchy_cart.get(0).begin(),
                     dt_cauchy_cart.get(0).end());
  dt_cauchy_y.insert(dt_cauchy_y.end(), dt_cauchy_cart.get(1).begin(),
                     dt_cauchy_cart.get(1).end());
  dt_cauchy_z.insert(dt_cauchy_z.end(), dt_cauchy_cart.get(2).begin(),
                     dt_cauchy_cart.get(2).end());

  dt_inertial_x.insert(dt_inertial_x.end(), dt_inertial_cart.get(0).begin(),
                       dt_inertial_cart.get(0).end());
  dt_inertial_y.insert(dt_inertial_y.end(), dt_inertial_cart.get(1).begin(),
                       dt_inertial_cart.get(1).end());
  dt_inertial_z.insert(dt_inertial_z.end(), dt_inertial_cart.get(2).begin(),
                       dt_inertial_cart.get(2).end());

  auto& du_t = get<::Tags::dt<Cce::Tags::InertialRetardedTime>>(spectre_box);

  dt_u_scri.insert(dt_u_scri.end(), du_t.get().begin(), du_t.get().end());

  auto& eth_inertial_retarded_time_from_cce =
      get<Cce::Tags::EthInertialRetardedTime>(spectre_box);
  auto& news_from_cce = get<Cce::Tags::News>(spectre_box);
  auto& strain_from_cce =
      get<Cce::Tags::ScriPlus<Cce::Tags::Strain>>(spectre_box);
  auto& psi0_from_cce = get<Cce::Tags::ScriPlus<Cce::Tags::Psi0>>(spectre_box);
  auto& psi1_from_cce = get<Cce::Tags::ScriPlus<Cce::Tags::Psi1>>(spectre_box);
  auto& psi2_from_cce = get<Cce::Tags::ScriPlus<Cce::Tags::Psi2>>(spectre_box);
  auto& psi3_from_cce = get<Cce::Tags::ScriPlus<Cce::Tags::Psi3>>(spectre_box);
  auto& psi4_from_cce =
      get<Cce::Tags::TimeIntegral<Cce::Tags::ScriPlus<Cce::Tags::Psi4>>>(
          spectre_box);

  auto& psi0_for_ccm_from_spectre =
      get<Cce::Tags::BoundaryValue<Cce::Tags::Psi0Match>>(spectre_box);

  // auto& wij_ccm = get<Cce::Tags::WxxForSpECTest>(spectre_box);
  auto& ccm_tetrad_coeff_theta =
      get<Cce::Tags::BoundaryValue<Cce::Tags::TetradCoeffTheta>>(spectre_box);
  auto& ccm_tetrad_coeff_phi =
      get<Cce::Tags::BoundaryValue<Cce::Tags::TetradCoeffPhi>>(spectre_box);

  eth_inertial_retarded_time.insert(
      eth_inertial_retarded_time.end(),
      get(eth_inertial_retarded_time_from_cce).data().begin(),
      get(eth_inertial_retarded_time_from_cce).data().end());
  // for (unsigned int i = 0; i <
  // get(eth_inertial_retarded_time_from_cce).size();
  //      i++) {
  //   std::cout << eth_inertial_retarded_time[i] -
  //                    (get(eth_inertial_retarded_time_from_cce).data()[i])
  //             << std::endl;
  // }

  news.insert(news.end(), get(news_from_cce).data().begin(),
              get(news_from_cce).data().end());
  // for (unsigned int i = 0; i < get(news_from_cce).size(); i++) {
  //   std::cout << news[i] - (get(news_from_cce).data()[i]) << std::endl;
  // }

  strain.insert(strain.end(), get(strain_from_cce).data().begin(),
                get(strain_from_cce).data().end());
  // for (unsigned int i = 0; i < get(strain_from_cce).size(); i++) {
  //   std::cout << strain[i] - (get(strain_from_cce).data()[i]) << std::endl;
  // }

  psi0.insert(psi0.end(), get(psi0_from_cce).data().begin(),
              get(psi0_from_cce).data().end());
  // for (unsigned int i = 0; i < get(psi0_from_cce).size(); i++) {
  //   std::cout << psi0[i] - (get(psi0_from_cce).data()[i]) << std::endl;
  // }

  psi1.insert(psi1.end(), get(psi1_from_cce).data().begin(),
              get(psi1_from_cce).data().end());
  // for (unsigned int i = 0; i < get(psi1_from_cce).size(); i++) {
  //   std::cout << psi1[i] - (get(psi1_from_cce).data()[i]) << std::endl;
  // }

  psi2.insert(psi2.end(), get(psi2_from_cce).data().begin(),
              get(psi2_from_cce).data().end());
  // for (unsigned int i = 0; i < get(psi2_from_cce).size(); i++) {
  //   std::cout << psi2[i] - (get(psi2_from_cce).data()[i]) << std::endl;
  // }

  psi3.insert(psi3.end(), get(psi3_from_cce).data().begin(),
              get(psi3_from_cce).data().end());
  // for (unsigned int i = 0; i < get(psi3_from_cce).size(); i++) {
  //   std::cout << psi3[i] - (get(psi3_from_cce).data()[i]) << std::endl;
  // }

  psi4.insert(psi4.end(), get(psi4_from_cce).data().begin(),
              get(psi4_from_cce).data().end());
  // for (unsigned int i = 0; i < get(psi4_from_cce).size(); i++) {
  //   std::cout << psi4[i] - (get(psi4_from_cce).data()[i]) << std::endl;
  // }

  psi0_ccm.insert(psi0_ccm.end(), get(psi0_for_ccm_from_spectre).data().begin(),
                  get(psi0_for_ccm_from_spectre).data().end());
  // for (unsigned int i = 0; i < get(psi0_for_ccm_from_spectre).size(); i++) {
  //   std::cout << psi0_ccm[i] - get(psi0_for_ccm_from_spectre).data()[i]
  //             << std::endl;
  // }

  // wxx_test_for_spec.insert(wxx_test_for_spec.end(),
  // get(wij_ccm).data().begin(),
  //                          get(wij_ccm).data().end());
  // for (unsigned int i = 0; i < get(wij_ccm).size(); i++) {
  //   std::cout << wxx_test_for_spec[i] - (get(wij_ccm).data()[i]) <<
  //   std::endl;
  // }
  auto& real_coeff_theta = real(get(ccm_tetrad_coeff_theta).data());

  coeff_theta.insert(coeff_theta.end(), real_coeff_theta.begin(),
                     real_coeff_theta.end());
  // for (unsigned int i = 0; i < get(ccm_tetrad_coeff_theta).size(); i++) {
  //   std::cout << coeff_theta[i] - (get(ccm_tetrad_coeff_theta).data()[i])
  //             << std::endl;
  // }

  coeff_phi.insert(coeff_phi.end(), get(ccm_tetrad_coeff_phi).data().begin(),
                   get(ccm_tetrad_coeff_phi).data().end());
  // for (unsigned int i = 0; i < get(ccm_tetrad_coeff_phi).size(); i++) {
  //   std::cout << coeff_phi[i] - get(ccm_tetrad_coeff_phi).data()[i]
  //             << std::endl;
  // }

  //   std::cout << real(get(psi3).data())[0] << " " <<
  //   imag(get(psi3).data())[0]
  //             << std::endl;

  // DataVector dv_psi0 = gh_read * 2.;

  // for (unsigned int i = 0; i < dv_psi0.size(); i++) {
  //   psi0.push_back(dv_psi0.at(i));
  // }

  auto& l2norm = get<Cce::Tags::CCMConstraintOmega>(spectre_box);
  auto& l2normc = get<Cce::Tags::CCMConstraintc>(spectre_box);
  auto& l2normd = get<Cce::Tags::CCMConstraintd>(spectre_box);
  ccmconstraintomega = abs(get(l2norm).data()[0]);
  ccmconstraintc = abs(get(l2normc).data()[0]);
  ccmconstraintd = abs(get(l2normd).data()[0]);
}

void ccm_interpolation(std::vector<std::complex<double>>& psi0_ccm_interpolated,
                       const std::vector<double>& cauchy_theta,
                       const std::vector<double>& cauchy_phi,
                       const size_t l_max,
                       const std::vector<std::complex<double>>& psi0_ccm) {
  DataVector cauchy_theta_dv(cauchy_theta.size());
  DataVector cauchy_phi_dv(cauchy_phi.size());
  std_vector_to_DataVector(cauchy_theta_dv, cauchy_theta);
  std_vector_to_DataVector(cauchy_phi_dv, cauchy_phi);

  Spectral::Swsh::SwshInterpolator interpolator{cauchy_theta_dv, cauchy_phi_dv,
                                                l_max};
  SpinWeighted<ComplexDataVector, 2> psi0_for_ccm_from_spectre{psi0_ccm.size()};
  SpinWeighted<ComplexDataVector, 2> psi0_for_ccm_from_spectre_interpolated;

  std::memcpy(psi0_for_ccm_from_spectre.data().data(), psi0_ccm.data(),
              sizeof(std::complex<double>) * psi0_ccm.size());

  interpolator.interpolate(
      make_not_null(&psi0_for_ccm_from_spectre_interpolated),
      psi0_for_ccm_from_spectre);

  psi0_ccm_interpolated.insert(
      psi0_ccm_interpolated.end(),
      psi0_for_ccm_from_spectre_interpolated.data().begin(),
      psi0_for_ccm_from_spectre_interpolated.data().end());
}

void ccm_interpolation0(
    std::vector<std::complex<double>>& psi0_ccm_interpolated,
    const std::vector<double>& cauchy_theta,
    const std::vector<double>& cauchy_phi, const size_t l_max,
    const std::vector<double>& psi0_ccm) {
  DataVector cauchy_theta_dv(cauchy_theta.size());
  DataVector cauchy_phi_dv(cauchy_phi.size());
  std_vector_to_DataVector(cauchy_theta_dv, cauchy_theta);
  std_vector_to_DataVector(cauchy_phi_dv, cauchy_phi);

  Spectral::Swsh::SwshInterpolator interpolator{cauchy_theta_dv, cauchy_phi_dv,
                                                l_max};
  SpinWeighted<ComplexDataVector, 0> psi0_for_ccm_from_spectre{psi0_ccm.size()};
  SpinWeighted<ComplexDataVector, 0> psi0_for_ccm_from_spectre_interpolated;

  for (size_t i = 0; i < psi0_ccm.size(); i++)
    psi0_for_ccm_from_spectre.data()[i] = psi0_ccm[i];

  interpolator.interpolate(
      make_not_null(&psi0_for_ccm_from_spectre_interpolated),
      psi0_for_ccm_from_spectre);

  psi0_ccm_interpolated.insert(
      psi0_ccm_interpolated.end(),
      psi0_for_ccm_from_spectre_interpolated.data().begin(),
      psi0_for_ccm_from_spectre_interpolated.data().end());
}

void std_vector_to_DataVector(DataVector& pi, const std::vector<double>& data) {
  const auto size = data.size();
  std::memcpy(pi.data(), data.data(), sizeof(double) * size);
  // for (unsigned int i = 0; i < size; i++) {
  //   std::cout << pi[i] << " " << data[i] << " " << pi[i] - data[i] <<
  //   std::endl;
  // }
}

void std_vector_to_DataVector(tnsr::aa<DataVector, 3>& pi,
                              const std::vector<std::vector<double>>& data) {
  const auto size = data.at(0).size();
  std::memcpy(get<0, 0>(pi).data(), data.at(0).data(), sizeof(double) * size);
  std::memcpy(get<0, 1>(pi).data(), data.at(1).data(), sizeof(double) * size);
  std::memcpy(get<0, 2>(pi).data(), data.at(2).data(), sizeof(double) * size);
  std::memcpy(get<0, 3>(pi).data(), data.at(3).data(), sizeof(double) * size);
  std::memcpy(get<1, 1>(pi).data(), data.at(4).data(), sizeof(double) * size);
  std::memcpy(get<1, 2>(pi).data(), data.at(5).data(), sizeof(double) * size);
  std::memcpy(get<1, 3>(pi).data(), data.at(6).data(), sizeof(double) * size);
  std::memcpy(get<2, 2>(pi).data(), data.at(7).data(), sizeof(double) * size);
  std::memcpy(get<2, 3>(pi).data(), data.at(8).data(), sizeof(double) * size);
  std::memcpy(get<3, 3>(pi).data(), data.at(9).data(), sizeof(double) * size);
  // for (unsigned int i = 0; i < size; i++) {
  //   std::cout << get<0, 1>(pi)[i] - data.at(1)[i] << std::endl;
  //   std::cout << get<0, 2>(pi)[i] - data.at(2)[i] << std::endl;
  //   std::cout << get<0, 3>(pi)[i] - data.at(3)[i] << std::endl;
  //   std::cout << get<1, 1>(pi)[i] - data.at(4)[i] << std::endl;
  //   std::cout << get<1, 2>(pi)[i] - data.at(5)[i] << std::endl;
  //   std::cout << get<1, 3>(pi)[i] - data.at(6)[i] << std::endl;
  //   std::cout << get<2, 2>(pi)[i] - data.at(7)[i] << std::endl;
  //   std::cout << get<2, 3>(pi)[i] - data.at(8)[i] << std::endl;
  //   std::cout << get<3, 3>(pi)[i] - data.at(9)[i] << std::endl;
  // }
}

void tri_std_vector_to_DataVector(
    tnsr::iaa<DataVector, 3>& pi,
    const std::vector<std::vector<std::vector<double>>>& data) {
  const auto size = data.at(0).at(0).size();
  for (size_t ijj = 0; ijj < 3; ++ijj) {
    std::memcpy(pi.get(ijj, 0, 0).data(), data.at(ijj).at(0).data(),
                sizeof(double) * size);
    std::memcpy(pi.get(ijj, 0, 1).data(), data.at(ijj).at(1).data(),
                sizeof(double) * size);
    std::memcpy(pi.get(ijj, 0, 2).data(), data.at(ijj).at(2).data(),
                sizeof(double) * size);
    std::memcpy(pi.get(ijj, 0, 3).data(), data.at(ijj).at(3).data(),
                sizeof(double) * size);

    std::memcpy(pi.get(ijj, 1, 1).data(), data.at(ijj).at(4).data(),
                sizeof(double) * size);
    std::memcpy(pi.get(ijj, 1, 2).data(), data.at(ijj).at(5).data(),
                sizeof(double) * size);
    std::memcpy(pi.get(ijj, 1, 3).data(), data.at(ijj).at(6).data(),
                sizeof(double) * size);

    std::memcpy(pi.get(ijj, 2, 2).data(), data.at(ijj).at(7).data(),
                sizeof(double) * size);
    std::memcpy(pi.get(ijj, 2, 3).data(), data.at(ijj).at(8).data(),
                sizeof(double) * size);
    std::memcpy(pi.get(ijj, 3, 3).data(), data.at(ijj).at(9).data(),
                sizeof(double) * size);
  }
}

void gh_to_bondi(Scalar<SpinWeighted<ComplexDataVector, 0>>& beta,
                 Scalar<SpinWeighted<ComplexDataVector, 2>>& dr_j,
                 Scalar<SpinWeighted<ComplexDataVector, 2>>& du_j,
                 Scalar<SpinWeighted<ComplexDataVector, 0>>& du_r,
                 Scalar<SpinWeighted<ComplexDataVector, 2>>& bondih,
                 Scalar<SpinWeighted<ComplexDataVector, 2>>& bondij,
                 Scalar<SpinWeighted<ComplexDataVector, 1>>& bondiq,
                 Scalar<SpinWeighted<ComplexDataVector, 0>>& bondir,
                 Scalar<SpinWeighted<ComplexDataVector, 1>>& bondiu,
                 Scalar<SpinWeighted<ComplexDataVector, 1>>& dr_u,
                 Scalar<SpinWeighted<ComplexDataVector, 0>>& bondiw,
                 Scalar<SpinWeighted<ComplexDataVector, 0>>& du_r_r,
                 Scalar<SpinWeighted<ComplexDataVector, 0>>& spec_norm,
                 const std::vector<std::vector<double>>& spacetime_metric,
                 const std::vector<std::vector<double>>& pi,
                 const std::vector<std::vector<std::vector<double>>>& phi,
                 const size_t l_max, const double radius) {
  // create_bondi_boundary_data
  const auto size = pi.at(0).size();
  tnsr::aa<DataVector, 3> pi_datavector{size};
  tnsr::aa<DataVector, 3> spacetime_metric_datavector{size};
  tnsr::iaa<DataVector, 3> phi_datavector{size};
  std_vector_to_DataVector(pi_datavector, pi);
  std_vector_to_DataVector(spacetime_metric_datavector, spacetime_metric);
  tri_std_vector_to_DataVector(phi_datavector, phi);

  size_t boundary_size = get_vector_size(l_max);

  Variables<typename MyEvolutionMetavars::cce_boundary_communication_tags>
      blahblah{boundary_size};
  Cce::create_bondi_boundary_data(make_not_null(&blahblah), phi_datavector,
                                  pi_datavector, spacetime_metric_datavector,
                                  radius, l_max);

  beta = get<Cce::Tags::BoundaryValue<Cce::Tags::BondiBeta>>(blahblah);
  bondiu = get<Cce::Tags::BoundaryValue<Cce::Tags::BondiU>>(blahblah);
  dr_u =
      get<Cce::Tags::BoundaryValue<Cce::Tags::Dr<Cce::Tags::BondiU>>>(blahblah);
  bondiq = get<Cce::Tags::BoundaryValue<Cce::Tags::BondiQ>>(blahblah);
  bondiw = get<Cce::Tags::BoundaryValue<Cce::Tags::BondiW>>(blahblah);
  bondij = get<Cce::Tags::BoundaryValue<Cce::Tags::BondiJ>>(blahblah);
  dr_j =
      get<Cce::Tags::BoundaryValue<Cce::Tags::Dr<Cce::Tags::BondiJ>>>(blahblah);
  bondih = get<Cce::Tags::BoundaryValue<Cce::Tags::BondiH>>(blahblah);
  du_j =
      get<Cce::Tags::BoundaryValue<Cce::Tags::Du<Cce::Tags::BondiJ>>>(blahblah);
  bondir = get<Cce::Tags::BoundaryValue<Cce::Tags::BondiR>>(blahblah);
  du_r =
      get<Cce::Tags::BoundaryValue<Cce::Tags::Du<Cce::Tags::BondiR>>>(blahblah);
  du_r_r = get<Cce::Tags::BoundaryValue<Cce::Tags::DuRDividedByR>>(blahblah);
  spec_norm =
      get<Cce::Tags::BoundaryValue<Cce::Tags::SpECNormalization>>(blahblah);

  //   beta = get<Cce::Tags::BoundaryValue<Cce::Tags::BondiBeta>>(spectre_box);
  //   dr_j = get<Cce::Tags::BoundaryValue<Cce::Tags::Dr<Cce::Tags::BondiJ>>>(
  //       spectre_box);
  //   du_r = get<Cce::Tags::BoundaryValue<Cce::Tags::Du<Cce::Tags::BondiR>>>(
  //       spectre_box);
  //   bondih = get<Cce::Tags::BoundaryValue<Cce::Tags::BondiH>>(spectre_box);
  //   bondij = get<Cce::Tags::BoundaryValue<Cce::Tags::BondiJ>>(spectre_box);
  //   bondiq = get<Cce::Tags::BoundaryValue<Cce::Tags::BondiQ>>(spectre_box);
  //   bondir = get<Cce::Tags::BoundaryValue<Cce::Tags::BondiR>>(spectre_box);
  //   bondiu = get<Cce::Tags::BoundaryValue<Cce::Tags::BondiU>>(spectre_box);
  //   bondiw = get<Cce::Tags::BoundaryValue<Cce::Tags::BondiW>>(spectre_box);
}

void gh_to_bondi_spec(std::vector<double>& final_array,
                      const std::vector<std::vector<double>>& spacetime_metric,
                      const std::vector<std::vector<double>>& pi,
                      const std::vector<std::vector<std::vector<double>>>& phi,
                      const size_t l_max, const double radius) {
  // create_bondi_boundary_data
  const auto size = pi.at(0).size();
  tnsr::aa<DataVector, 3> pi_datavector{size};
  tnsr::aa<DataVector, 3> spacetime_metric_datavector{size};
  tnsr::iaa<DataVector, 3> phi_datavector{size};
  std_vector_to_DataVector(pi_datavector, pi);
  std_vector_to_DataVector(spacetime_metric_datavector, spacetime_metric);
  tri_std_vector_to_DataVector(phi_datavector, phi);

  size_t boundary_size = get_vector_size(l_max);

  Variables<typename MyEvolutionMetavars::cce_boundary_communication_tags>
      blahblah{boundary_size};
  Cce::create_bondi_boundary_data(make_not_null(&blahblah), phi_datavector,
                                  pi_datavector, spacetime_metric_datavector,
                                  radius, l_max);

  auto& beta_spectre =
      real(get(get<Cce::Tags::BoundaryValue<Cce::Tags::BondiBeta>>(blahblah))
               .data());
  final_array.insert(final_array.end(), beta_spectre.begin(),
                     beta_spectre.end());

  auto& bondiu_spectre =
      get(get<Cce::Tags::BoundaryValue<Cce::Tags::BondiU>>(blahblah)).data();
  final_array.insert(final_array.end(), real(bondiu_spectre).begin(),
                     real(bondiu_spectre).end());
  final_array.insert(final_array.end(), imag(bondiu_spectre).begin(),
                     imag(bondiu_spectre).end());

  auto& dr_u_spectre =
      get(get<Cce::Tags::BoundaryValue<Cce::Tags::Dr<Cce::Tags::BondiU>>>(
              blahblah))
          .data();
  final_array.insert(final_array.end(), real(dr_u_spectre).begin(),
                     real(dr_u_spectre).end());
  final_array.insert(final_array.end(), imag(dr_u_spectre).begin(),
                     imag(dr_u_spectre).end());

  auto& bondiq_spectre =
      get(get<Cce::Tags::BoundaryValue<Cce::Tags::BondiQ>>(blahblah)).data();
  final_array.insert(final_array.end(), real(bondiq_spectre).begin(),
                     real(bondiq_spectre).end());
  final_array.insert(final_array.end(), imag(bondiq_spectre).begin(),
                     imag(bondiq_spectre).end());

  auto& bondiw_spectre = real(
      get(get<Cce::Tags::BoundaryValue<Cce::Tags::BondiW>>(blahblah)).data());
  final_array.insert(final_array.end(), bondiw_spectre.begin(),
                     bondiw_spectre.end());

  auto& bondij_spectre =
      get(get<Cce::Tags::BoundaryValue<Cce::Tags::BondiJ>>(blahblah)).data();
  final_array.insert(final_array.end(), real(bondij_spectre).begin(),
                     real(bondij_spectre).end());
  final_array.insert(final_array.end(), imag(bondij_spectre).begin(),
                     imag(bondij_spectre).end());

  auto& dr_j_spectre =
      get(get<Cce::Tags::BoundaryValue<Cce::Tags::Dr<Cce::Tags::BondiJ>>>(
              blahblah))
          .data();
  final_array.insert(final_array.end(), real(dr_j_spectre).begin(),
                     real(dr_j_spectre).end());
  final_array.insert(final_array.end(), imag(dr_j_spectre).begin(),
                     imag(dr_j_spectre).end());

  auto& bondih_spectre =
      get(get<Cce::Tags::BoundaryValue<Cce::Tags::BondiH>>(blahblah)).data();
  final_array.insert(final_array.end(), real(bondih_spectre).begin(),
                     real(bondih_spectre).end());
  final_array.insert(final_array.end(), imag(bondih_spectre).begin(),
                     imag(bondih_spectre).end());

  auto& du_j_spectre =
      get(get<Cce::Tags::BoundaryValue<Cce::Tags::Du<Cce::Tags::BondiJ>>>(
              blahblah))
          .data();
  final_array.insert(final_array.end(), real(du_j_spectre).begin(),
                     real(du_j_spectre).end());
  final_array.insert(final_array.end(), imag(du_j_spectre).begin(),
                     imag(du_j_spectre).end());

  auto& bondir_spectre = real(
      get(get<Cce::Tags::BoundaryValue<Cce::Tags::BondiR>>(blahblah)).data());
  final_array.insert(final_array.end(), bondir_spectre.begin(),
                     bondir_spectre.end());

  auto& du_r_spectre =
      real(get(get<Cce::Tags::BoundaryValue<Cce::Tags::Du<Cce::Tags::BondiR>>>(
                   blahblah))
               .data());
  final_array.insert(final_array.end(), du_r_spectre.begin(),
                     du_r_spectre.end());

  auto& du_r_r_spectre = real(
      get(get<Cce::Tags::BoundaryValue<Cce::Tags::DuRDividedByR>>(blahblah))
          .data());
  final_array.insert(final_array.end(), du_r_r_spectre.begin(),
                     du_r_r_spectre.end());

  auto& spec_norm_spectre = real(
      get(get<Cce::Tags::BoundaryValue<Cce::Tags::SpECNormalization>>(blahblah))
          .data());
  final_array.insert(final_array.end(), spec_norm_spectre.begin(),
                     spec_norm_spectre.end());
}

namespace spectre {
struct MyScriPlusInterpolationManager {
  MyScriPlusInterpolationManager(size_t target_number_of_points, size_t l_max)
      : target_number_of_points_(target_number_of_points),
        vector_size_(Spectral::Swsh::number_of_swsh_collocation_points(l_max)),
        interpolator1_(
            std::make_unique<intrp::BarycentricRationalSpanInterpolator>(
                2 * target_number_of_points - 1,
                2 * target_number_of_points + 2)),
        interpolator2_(
            std::make_unique<intrp::BarycentricRationalSpanInterpolator>(
                2 * target_number_of_points - 1,
                2 * target_number_of_points + 2)),
        interpolator3_(
            std::make_unique<intrp::BarycentricRationalSpanInterpolator>(
                2 * target_number_of_points - 1,
                2 * target_number_of_points + 2)),
        interpolator4_(
            std::make_unique<intrp::BarycentricRationalSpanInterpolator>(
                2 * target_number_of_points - 1,
                2 * target_number_of_points + 2)),
        interpolator5_(
            std::make_unique<intrp::BarycentricRationalSpanInterpolator>(
                2 * target_number_of_points - 1,
                2 * target_number_of_points + 2)),
        interpolator6_(
            std::make_unique<intrp::BarycentricRationalSpanInterpolator>(
                2 * target_number_of_points - 1,
                2 * target_number_of_points + 2)),
        interpolator7_(
            std::make_unique<intrp::BarycentricRationalSpanInterpolator>(
                2 * target_number_of_points - 1,
                2 * target_number_of_points + 2)),
        interpolator8_(
            std::make_unique<intrp::BarycentricRationalSpanInterpolator>(
                2 * target_number_of_points - 1,
                2 * target_number_of_points + 2)),
        manager_news_(target_number_of_points, vector_size_,
                      std::move(interpolator1_)),
        manager_strain_(target_number_of_points, vector_size_,
                        std::move(interpolator2_)),
        manager_psi3_(target_number_of_points, vector_size_,
                      std::move(interpolator3_)),
        manager_psi2_(target_number_of_points, vector_size_,
                      std::move(interpolator4_)),
        manager_psi1_(target_number_of_points, vector_size_,
                      std::move(interpolator5_)),
        manager_psi0_(target_number_of_points, vector_size_,
                      std::move(interpolator6_)),
        manager_eth_inertial_retarded_time_(
            target_number_of_points, vector_size_, std::move(interpolator7_)),
        manager_psi4_(target_number_of_points, vector_size_,
                      std::move(interpolator8_)){

        };

 private:
  size_t target_number_of_points_, vector_size_;
  std::unique_ptr<intrp::BarycentricRationalSpanInterpolator> interpolator1_;
  std::unique_ptr<intrp::BarycentricRationalSpanInterpolator> interpolator2_;
  std::unique_ptr<intrp::BarycentricRationalSpanInterpolator> interpolator3_;
  std::unique_ptr<intrp::BarycentricRationalSpanInterpolator> interpolator4_;
  std::unique_ptr<intrp::BarycentricRationalSpanInterpolator> interpolator5_;
  std::unique_ptr<intrp::BarycentricRationalSpanInterpolator> interpolator6_;
  std::unique_ptr<intrp::BarycentricRationalSpanInterpolator> interpolator7_;
  std::unique_ptr<intrp::BarycentricRationalSpanInterpolator> interpolator8_;

 public:
  Cce::ScriPlusInterpolationManager<ComplexDataVector, Cce::Tags::News>
      manager_news_;
  Cce::ScriPlusInterpolationManager<ComplexDataVector,
                                    Cce::Tags::ScriPlus<Cce::Tags::Strain>>
      manager_strain_;
  Cce::ScriPlusInterpolationManager<ComplexDataVector,
                                    Cce::Tags::ScriPlus<Cce::Tags::Psi3>>
      manager_psi3_;
  Cce::ScriPlusInterpolationManager<ComplexDataVector,
                                    Cce::Tags::ScriPlus<Cce::Tags::Psi2>>
      manager_psi2_;
  Cce::ScriPlusInterpolationManager<ComplexDataVector,
                                    Cce::Tags::ScriPlus<Cce::Tags::Psi1>>
      manager_psi1_;
  Cce::ScriPlusInterpolationManager<ComplexDataVector,
                                    Cce::Tags::ScriPlus<Cce::Tags::Psi0>>
      manager_psi0_;
  Cce::ScriPlusInterpolationManager<ComplexDataVector,
                                    Cce::Tags::EthInertialRetardedTime>
      manager_eth_inertial_retarded_time_;
  Cce::ScriPlusInterpolationManager<ComplexDataVector,
                                    Cce::Tags::Du<Cce::Tags::TimeIntegral<
                                        Cce::Tags::ScriPlus<Cce::Tags::Psi4>>>>
      manager_psi4_;
};

InterpolationInterface::InterpolationInterface(size_t target_number_of_points,
                                               size_t l_max,
                                               size_t scri_output_density,
                                               size_t observation_l_max)
    : my_scri_plus_interpolation_manager_(nullptr),
      scri_output_density_(scri_output_density),
      l_max_(l_max),
      observation_l_max_(observation_l_max) {
  my_scri_plus_interpolation_manager_ =
      new MyScriPlusInterpolationManager(target_number_of_points, l_max);
}

InterpolationInterface::~InterpolationInterface() {
  delete my_scri_plus_interpolation_manager_;
}
void InterpolationInterface::clear() {
  delete my_scri_plus_interpolation_manager_;
  my_scri_plus_interpolation_manager_ = nullptr;
}

std::deque<std::pair<double, double>>
InterpolationInterface::get_u_bondi_ranges() {
  return my_scri_plus_interpolation_manager_->manager_psi0_
      .get_u_bondi_ranges();
}

std::deque<double> InterpolationInterface::get_target_times() const {
  return my_scri_plus_interpolation_manager_->manager_psi0_.get_target_times();
}

std::deque<std::vector<std::complex<double>>> InterpolationInterface::get_psi0() const {
  auto psi0 = my_scri_plus_interpolation_manager_->manager_psi0_.get_data();
  std::deque<std::vector<std::complex<double>>> test;
  for (const auto& element : psi0) {
    std::vector<std::complex<double>> temp(element.size());
    for (size_t i = 0; i < element.size(); i++) {
      temp[i] = element[i];
    }
    test.push_back(temp);
  }
  return test;
}

std::deque<std::vector<std::complex<double>>> InterpolationInterface::get_psi1()
    const {
  auto psi0 = my_scri_plus_interpolation_manager_->manager_psi1_.get_data();
  std::deque<std::vector<std::complex<double>>> test;
  for (const auto& element : psi0) {
    std::vector<std::complex<double>> temp(element.size());
    for (size_t i = 0; i < element.size(); i++) {
      temp[i] = element[i];
    }
    test.push_back(temp);
  }
  return test;
}

std::deque<std::vector<std::complex<double>>> InterpolationInterface::get_psi2()
    const {
  auto psi0 = my_scri_plus_interpolation_manager_->manager_psi2_.get_data();
  std::deque<std::vector<std::complex<double>>> test;
  for (const auto& element : psi0) {
    std::vector<std::complex<double>> temp(element.size());
    for (size_t i = 0; i < element.size(); i++) {
      temp[i] = element[i];
    }
    test.push_back(temp);
  }
  return test;
}

std::deque<std::vector<std::complex<double>>> InterpolationInterface::get_psi3()
    const {
  auto psi0 = my_scri_plus_interpolation_manager_->manager_psi3_.get_data();
  std::deque<std::vector<std::complex<double>>> test;
  for (const auto& element : psi0) {
    std::vector<std::complex<double>> temp(element.size());
    for (size_t i = 0; i < element.size(); i++) {
      temp[i] = element[i];
    }
    test.push_back(temp);
  }
  return test;
}

std::deque<std::vector<std::complex<double>>> InterpolationInterface::get_psi4()
    const {
  auto psi0 = my_scri_plus_interpolation_manager_->manager_psi4_.get_data();
  std::deque<std::vector<std::complex<double>>> test;
  for (const auto& element : psi0) {
    std::vector<std::complex<double>> temp(element.size());
    for (size_t i = 0; i < element.size(); i++) {
      temp[i] = element[i];
    }
    test.push_back(temp);
  }
  return test;
}

std::deque<std::vector<std::complex<double>>> InterpolationInterface::get_news()
    const {
  auto psi0 = my_scri_plus_interpolation_manager_->manager_news_.get_data();
  std::deque<std::vector<std::complex<double>>> test;
  for (const auto& element : psi0) {
    std::vector<std::complex<double>> temp(element.size());
    for (size_t i = 0; i < element.size(); i++) {
      temp[i] = element[i];
    }
    test.push_back(temp);
  }
  return test;
}

std::deque<std::vector<std::complex<double>>>
InterpolationInterface::get_strain() const {
  auto psi0 = my_scri_plus_interpolation_manager_->manager_strain_.get_data();
  std::deque<std::vector<std::complex<double>>> test;
  for (const auto& element : psi0) {
    std::vector<std::complex<double>> temp(element.size());
    for (size_t i = 0; i < element.size(); i++) {
      temp[i] = element[i];
    }
    test.push_back(temp);
  }
  return test;
}

std::deque<std::vector<std::complex<double>>>
InterpolationInterface::get_eth_inertial_retarded_time() const {
  auto psi0 = my_scri_plus_interpolation_manager_
                  ->manager_eth_inertial_retarded_time_.get_data();
  std::deque<std::vector<std::complex<double>>> test;
  for (const auto& element : psi0) {
    std::vector<std::complex<double>> temp(element.size());
    for (size_t i = 0; i < element.size(); i++) {
      temp[i] = element[i];
    }
    test.push_back(temp);
  }
  return test;
}

std::deque<std::vector<double>> InterpolationInterface::get_u_bondi_values()
    const {
  auto u_bondi_values =
      my_scri_plus_interpolation_manager_->manager_psi0_.get_u_bondi_values();

  std::deque<std::vector<double>> test;

  for (const auto& element : u_bondi_values) {
    std::vector<double> temp(element.size());
    for (size_t i = 0; i < element.size(); i++) {
      temp[i] = element[i];
    }
    test.push_back(temp);
  }
  return test;
}

void InterpolationInterface::InsertTargetTime(double time) {
  my_scri_plus_interpolation_manager_->manager_psi0_.insert_target_time(time);
  my_scri_plus_interpolation_manager_->manager_psi1_.insert_target_time(time);
  my_scri_plus_interpolation_manager_->manager_psi2_.insert_target_time(time);
  my_scri_plus_interpolation_manager_->manager_psi3_.insert_target_time(time);
  my_scri_plus_interpolation_manager_->manager_psi4_.insert_target_time(time);
  my_scri_plus_interpolation_manager_->manager_strain_.insert_target_time(time);
  my_scri_plus_interpolation_manager_->manager_news_.insert_target_time(time);
  my_scri_plus_interpolation_manager_->manager_eth_inertial_retarded_time_
      .insert_target_time(time);
}

void InterpolationInterface::InsertInterpolationScriData(
    std::vector<double>& inertial_time, std::vector<std::complex<double>>& psi0,
    std::vector<std::complex<double>>& psi1,
    std::vector<std::complex<double>>& psi2,
    std::vector<std::complex<double>>& psi3,
    std::vector<std::complex<double>>& psi4,
    std::vector<std::complex<double>>& strain,
    std::vector<std::complex<double>>& news,
    std::vector<std::complex<double>>& eth_inertial_retarded_time) {
  const ComplexDataVector spectre_psi0 =
      ComplexDataVector(psi0.data(), psi0.size());
  const ComplexDataVector spectre_psi1 =
      ComplexDataVector(psi1.data(), psi1.size());
  const ComplexDataVector spectre_psi2 =
      ComplexDataVector(psi2.data(), psi2.size());
  const ComplexDataVector spectre_psi3 =
      ComplexDataVector(psi3.data(), psi3.size());
  const ComplexDataVector spectre_psi4 =
      ComplexDataVector(psi4.data(), psi4.size());
  const ComplexDataVector spectre_strain =
      ComplexDataVector(strain.data(), strain.size());
  const ComplexDataVector spectre_news =
      ComplexDataVector(news.data(), news.size());
  const ComplexDataVector spectre_eth_inertial_retarded_time =
      ComplexDataVector(eth_inertial_retarded_time.data(),
                        eth_inertial_retarded_time.size());

  const DataVector spectre_inertial_time =
      DataVector(inertial_time.data(), inertial_time.size());

  my_scri_plus_interpolation_manager_->manager_psi0_.insert_data(
      spectre_inertial_time, spectre_psi0);
  my_scri_plus_interpolation_manager_->manager_psi1_.insert_data(
      spectre_inertial_time, spectre_psi1);
  my_scri_plus_interpolation_manager_->manager_psi2_.insert_data(
      spectre_inertial_time, spectre_psi2);
  my_scri_plus_interpolation_manager_->manager_psi3_.insert_data(
      spectre_inertial_time, spectre_psi3);
  my_scri_plus_interpolation_manager_->manager_psi4_.insert_data(
      spectre_inertial_time, spectre_psi4);
  my_scri_plus_interpolation_manager_->manager_strain_.insert_data(
      spectre_inertial_time, spectre_strain);
  my_scri_plus_interpolation_manager_->manager_news_.insert_data(
      spectre_inertial_time, spectre_news);
  my_scri_plus_interpolation_manager_->manager_eth_inertial_retarded_time_
      .insert_data(spectre_inertial_time, spectre_eth_inertial_retarded_time);
}

void InterpolationInterface::ScriObserveInterpolated(
    std::queue<std::vector<double>>& psi0_to_write_final,
    std::queue<std::vector<double>>& psi1_to_write_final,
    std::queue<std::vector<double>>& psi2_to_write_final,
    std::queue<std::vector<double>>& psi3_to_write_final,
    std::queue<std::vector<double>>& psi4_to_write_final,
    std::queue<std::vector<double>>& strain_to_write_final,
    std::queue<std::vector<double>>& news_to_write_final) {
  std::vector<double> eth_inertial_retarded_time_to_write(
      2 * square(observation_l_max_ + 1) + 1);
  std::vector<double> psi0_to_write(2 * square(observation_l_max_ + 1) + 1);
  std::vector<double> psi1_to_write(2 * square(observation_l_max_ + 1) + 1);
  std::vector<double> psi2_to_write(2 * square(observation_l_max_ + 1) + 1);
  std::vector<double> psi3_to_write(2 * square(observation_l_max_ + 1) + 1);
  std::vector<double> psi4_to_write(2 * square(observation_l_max_ + 1) + 1);
  std::vector<double> strain_to_write(2 * square(observation_l_max_ + 1) + 1);
  std::vector<double> news_to_write(2 * square(observation_l_max_ + 1) + 1);

  ComplexModalVector goldberg_modes{square(l_max_ + 1)};

  Variables<Cce::Actions::detail::weyl_correction_list>
      corrected_scri_plus_weyl{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max_)};

  while (my_scri_plus_interpolation_manager_->manager_news_
             .first_time_is_ready_to_interpolate()) {
    double interpolation_time = 0.0;
    std::pair<double, ComplexDataVector> interpolation;

    {
      interpolation = my_scri_plus_interpolation_manager_->manager_psi4_
                          .interpolate_and_pop_first_time();
      interpolation_time = interpolation.first;
      get(get<Cce::Tags::Du<
              Cce::Tags::TimeIntegral<Cce::Tags::ScriPlus<Cce::Tags::Psi4>>>>(
              corrected_scri_plus_weyl))
          .data() = interpolation.second;
    }

    {
      interpolation = my_scri_plus_interpolation_manager_->manager_psi3_
                          .interpolate_and_pop_first_time();
      interpolation_time = interpolation.first;
      get(get<Cce::Tags::ScriPlus<Cce::Tags::Psi3>>(corrected_scri_plus_weyl))
          .data() = interpolation.second;
    }

    {
      interpolation = my_scri_plus_interpolation_manager_->manager_psi2_
                          .interpolate_and_pop_first_time();
      interpolation_time = interpolation.first;
      get(get<Cce::Tags::ScriPlus<Cce::Tags::Psi2>>(corrected_scri_plus_weyl))
          .data() = interpolation.second;
    }

    {
      interpolation = my_scri_plus_interpolation_manager_->manager_psi1_
                          .interpolate_and_pop_first_time();
      interpolation_time = interpolation.first;
      get(get<Cce::Tags::ScriPlus<Cce::Tags::Psi1>>(corrected_scri_plus_weyl))
          .data() = interpolation.second;
    }

    {
      interpolation = my_scri_plus_interpolation_manager_->manager_psi0_
                          .interpolate_and_pop_first_time();
      interpolation_time = interpolation.first;
      get(get<Cce::Tags::ScriPlus<Cce::Tags::Psi0>>(corrected_scri_plus_weyl))
          .data() = interpolation.second;
    }

    {
      interpolation = my_scri_plus_interpolation_manager_
                          ->manager_eth_inertial_retarded_time_
                          .interpolate_and_pop_first_time();
      interpolation_time = interpolation.first;
      get(get<Cce::Tags::EthInertialRetardedTime>(corrected_scri_plus_weyl))
          .data() = interpolation.second;
    }

    Cce::Actions::detail::correct_weyl_scalars_for_inertial_time(
        make_not_null(&corrected_scri_plus_weyl));

    {
      using tag = Cce::Tags::ScriPlus<Cce::Tags::Psi0>;
      transform_and_write_new<tag, tag::type::type::spin>(
          get(get<tag>(corrected_scri_plus_weyl)).data(), interpolation_time,
          make_not_null(&goldberg_modes), make_not_null(&psi0_to_write), l_max_,
          observation_l_max_);
      psi0_to_write_final.push(psi0_to_write);
    }
    {
      using tag = Cce::Tags::ScriPlus<Cce::Tags::Psi1>;
      transform_and_write_new<tag, tag::type::type::spin>(
          get(get<tag>(corrected_scri_plus_weyl)).data(), interpolation_time,
          make_not_null(&goldberg_modes), make_not_null(&psi1_to_write), l_max_,
          observation_l_max_);
      psi1_to_write_final.push(psi1_to_write);
    }
    {
      using tag = Cce::Tags::ScriPlus<Cce::Tags::Psi2>;
      transform_and_write_new<tag, tag::type::type::spin>(
          get(get<tag>(corrected_scri_plus_weyl)).data(), interpolation_time,
          make_not_null(&goldberg_modes), make_not_null(&psi2_to_write), l_max_,
          observation_l_max_);
      psi2_to_write_final.push(psi2_to_write);
    }

    {
      using tag = Cce::Tags::ScriPlus<Cce::Tags::Psi3>;
      transform_and_write_new<tag, tag::type::type::spin>(
          get(get<tag>(corrected_scri_plus_weyl)).data(), interpolation_time,
          make_not_null(&goldberg_modes), make_not_null(&psi3_to_write), l_max_,
          observation_l_max_);
      psi3_to_write_final.push(psi3_to_write);
    }

    {
      using tag = Cce::Tags::Du<
          Cce::Tags::TimeIntegral<Cce::Tags::ScriPlus<Cce::Tags::Psi4>>>;
      transform_and_write_new<tag, tag::type::type::spin>(
          get(get<tag>(corrected_scri_plus_weyl)).data(), interpolation_time,
          make_not_null(&goldberg_modes), make_not_null(&psi4_to_write), l_max_,
          observation_l_max_);
      psi4_to_write_final.push(psi4_to_write);
    }

    {
      using tag = Cce::Tags::ScriPlus<Cce::Tags::Strain>;
      interpolation = my_scri_plus_interpolation_manager_->manager_strain_
                          .interpolate_and_pop_first_time();
      transform_and_write_new<tag, tag::type::type::spin>(
          interpolation.second, interpolation.first,
          make_not_null(&goldberg_modes), make_not_null(&strain_to_write),
          l_max_, observation_l_max_);
      strain_to_write_final.push(strain_to_write);
    }
    {
      using tag = Cce::Tags::News;
      interpolation = my_scri_plus_interpolation_manager_->manager_news_
                          .interpolate_and_pop_first_time();
      transform_and_write_new<tag, tag::type::type::spin>(
          interpolation.second, interpolation.first,
          make_not_null(&goldberg_modes), make_not_null(&news_to_write), l_max_,
          observation_l_max_);
      news_to_write_final.push(news_to_write);
    }
  }
}

}  // namespace spectre
