// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "hello_world/hello_world.hpp"

#include <boost/preprocessor.hpp>
#include <iostream>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "Evolution/Executables/Cce/CharacteristicExtractBase.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeCharacteristicEvolutionVariables.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Evolution/Systems/Cce/WorldtubeBufferUpdater.hpp"
#include "Evolution/Systems/Cce/WorldtubeDataManager.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/CubicSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/LinearSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
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

void ccm_functions(std::vector<double>& psi0,
                   const std::vector<double>& bondi_beta_spec,
                   const std::vector<double>& bondi_dr_j_spec,
                   const std::vector<double>& bondi_du_r_spec,
                   const std::vector<double>& bondi_h_spec,
                   const std::vector<double>& bondi_j_spec,
                   const std::vector<double>& bondi_q_spec,
                   const std::vector<double>& bondi_r_spec,
                   const std::vector<double>& bondi_u_spec,
                   const std::vector<double>& bondi_w_spec) {
  // const DataVector gh_read{const_cast<double*>(gh.data()), gh.size()};

  const size_t l_max = 1;
  const size_t boundary_size =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  const size_t number_of_radial_points = 2;
  const size_t volume_size = boundary_size * number_of_radial_points;
  const size_t transform_buffer_size =
      number_of_radial_points *
      Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max);
  using spec_tags = Cce::Tags::characteristic_worldtube_boundary_tags<
      Cce::Tags::BoundaryValue>;

  using Metavariables = MyEvolutionMetavars;

  using initialize_action =
      Cce::Actions::InitializeCharacteristicEvolutionVariables<Metavariables>;
  using simple_tags_for_evolution =
      initialize_action::simple_tags_for_evolution;

  auto spectre_box = db::create<db::AddSimpleTags<simple_tags_for_evolution>>();

  //   Initialization::mutate_assign<
  //       tmpl::list<initialize_action::boundary_value_variables_tag>>(
  //       make_not_null(&spectre_box),
  //       typename initialize_action::boundary_value_variables_tag::type{
  //           boundary_size});
  Initialization::mutate_assign<simple_tags_for_evolution>(
      make_not_null(&spectre_box),
      typename initialize_action::boundary_value_variables_tag::type{
          boundary_size},
      typename initialize_action::coordinate_variables_tag::type{boundary_size},
      typename initialize_action::dt_coordinate_variables_tag::type{
          boundary_size},
      typename initialize_action::evolved_swsh_variables_tag::type{volume_size},
      typename initialize_action::evolved_swsh_dt_variables_tag::type{
          volume_size},
      typename initialize_action::angular_coordinates_variables_tag::type{
          boundary_size},
      typename initialize_action::scri_variables_tag::type{boundary_size},
      typename initialize_action::volume_variables_tag::type{volume_size},
      typename initialize_action::pre_swsh_derivatives_variables_tag::type{
          volume_size, 0.0},
      typename initialize_action::transform_buffer_variables_tag::type{
          transform_buffer_size, 0.0},
      typename initialize_action::swsh_derivative_variables_tag::type{
          volume_size, 0.0},
      Spectral::Swsh::SwshInterpolator{}, Spectral::Swsh::SwshInterpolator{},
      typename initialize_action::ccm_tag::type{boundary_size});

  //   db::mutate<Cce::Tags::BoundaryValue<Cce::Tags::BondiBeta>>(
  //       make_not_null(&spectre_box),
  //       [&bondi_beta_spec](
  //           const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector,
  //           0>>*>
  //               bondi_beta) {
  //         for (unsigned int i = 0; i < bondi_beta_spec.size(); i++) {
  //           //   get(*bondi_beta).data()[i] =
  //           //   bondi_beta_spec.at(i) * std::complex<double>(1.0, 0.0);
  //         }
  //       });

  Variables<spec_tags> boundary_variables{
      Spectral::Swsh::number_of_swsh_collocation_points(l_max)};

  auto& bondi_beta = get(
      get<Cce::Tags::BoundaryValue<Cce::Tags::BondiBeta>>(boundary_variables));
  auto& bondi_dr_j =
      get(get<Cce::Tags::BoundaryValue<Cce::Tags::Dr<Cce::Tags::BondiJ>>>(
          boundary_variables));
  auto& bondi_du_r =
      get(get<Cce::Tags::BoundaryValue<Cce::Tags::Du<Cce::Tags::BondiR>>>(
          boundary_variables));
  auto& bondi_h =
      get(get<Cce::Tags::BoundaryValue<Cce::Tags::BondiH>>(boundary_variables));
  auto& bondi_j =
      get(get<Cce::Tags::BoundaryValue<Cce::Tags::BondiJ>>(boundary_variables));
  auto& bondi_q =
      get(get<Cce::Tags::BoundaryValue<Cce::Tags::BondiQ>>(boundary_variables));
  auto& bondi_r =
      get(get<Cce::Tags::BoundaryValue<Cce::Tags::BondiR>>(boundary_variables));
  auto& bondi_u =
      get(get<Cce::Tags::BoundaryValue<Cce::Tags::BondiU>>(boundary_variables));
  auto& bondi_w =
      get(get<Cce::Tags::BoundaryValue<Cce::Tags::BondiW>>(boundary_variables));

  for (unsigned int i = 0; i < bondi_j_spec.size(); i++) {
    bondi_beta.data()[i] =
        bondi_beta_spec.at(i) * std::complex<double>(1.0, 0.0);
    bondi_dr_j.data()[i] =
        bondi_dr_j_spec.at(i) * std::complex<double>(1.0, 0.0);
    bondi_du_r.data()[i] =
        bondi_du_r_spec.at(i) * std::complex<double>(1.0, 0.0);
    bondi_h.data()[i] = bondi_h_spec.at(i) * std::complex<double>(1.0, 0.0);
    bondi_j.data()[i] = bondi_j_spec.at(i) * std::complex<double>(1.0, 0.0);
    bondi_q.data()[i] = bondi_q_spec.at(i) * std::complex<double>(1.0, 0.0);
    bondi_r.data()[i] = bondi_r_spec.at(i) * std::complex<double>(1.0, 0.0);
    bondi_u.data()[i] = bondi_u_spec.at(i) * std::complex<double>(1.0, 0.0);
    bondi_w.data()[i] = bondi_w_spec.at(i) * std::complex<double>(1.0, 0.0);
  }
  std::cout << get(get<Cce::Tags::BoundaryValue<Cce::Tags::DuRDividedByR>>(
                       boundary_variables))
                   .data()
            << std::endl;
  Cce::BondiWorldtubeDataManager q;
  q.populate_hypersurface_boundary_data_spec(&boundary_variables);
  std::cout << get(get<Cce::Tags::BoundaryValue<Cce::Tags::DuRDividedByR>>(
                       boundary_variables))
                   .data()
            << std::endl;
  // std::cout << q.get_l_max() << std::endl;
  // Cce::Tags::BondiBeta q;

  // DataVector dv_psi0 = gh_read * 2.;

  // for (unsigned int i = 0; i < dv_psi0.size(); i++) {
  //   psi0.push_back(dv_psi0.at(i));
  // }
}
