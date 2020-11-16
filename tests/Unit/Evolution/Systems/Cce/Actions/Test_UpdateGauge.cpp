// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <utility>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/Actions/UpdateGauge.hpp"
#include "Evolution/Systems/Cce/Components/CharacteristicEvolution.hpp"
#include "Evolution/Systems/Cce/GaugeTransformBoundaryData.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/NumericalAlgorithms/Spectral/SwshTestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/SwshCoefficients.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshFiltering.hpp"
#include "NumericalAlgorithms/Spectral/SwshTransform.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

#include "Parallel/Printf.hpp"

namespace Cce {

namespace {
using real_tags_to_generate = tmpl::list<Tags::CauchyCartesianCoords,
                                         Tags::InertialCartesianCoords>;

using real_tags_to_compute = tmpl::list<Tags::CauchyAngularCoords,
                                        Tags::InertialAngularCoords>;

using swsh_tags_to_compute =
    tmpl::list<Tags::GaugeC, Tags::GaugeD, Tags::GaugeOmega,
               Tags::GaugeCnohat, Tags::GaugeDnohat, Tags::GaugeOmeganohat,
               Spectral::Swsh::Tags::Derivative<Tags::GaugeOmega,
                                                Spectral::Swsh::Tags::Eth>,
               Spectral::Swsh::Tags::Derivative<Tags::GaugeOmeganohat,
                                                Spectral::Swsh::Tags::Eth>>;

using swsh_ccm_tags_to_generate = tmpl::list<Tags::BondiJ,Tags::OneMinusY>;

using swsh_ccm_tags_to_compute = tmpl::list<Tags::BondiJ_Cauchyview,
                                 Tags::Dy<Tags::BondiJ_Cauchyview>,
                                 Tags::Dy<Tags::Dy<Tags::BondiJ_Cauchyview>>,
                                 Tags::Psi0Match, Tags::Dy<Tags::Psi0Match>,
                                 Tags::BoundaryValue<Tags::Psi0Match>,
                                 Tags::BoundaryValue<Tags::Dlambda
                                                     <Tags::Psi0Match>>>;

using swsh_boundary_ccm_tags_to_generate = tmpl::list<Tags::BondiR,
                                                      Tags::BondiBeta>;

template <typename Metavariables>
struct mock_characteristic_evolution {
  using simple_tags = tmpl::push_back<
      db::AddSimpleTags<::Tags::Variables<tmpl::append<
                            real_tags_to_generate, real_tags_to_compute>>,
                        ::Tags::Variables<swsh_tags_to_compute>,
                        ::Tags::Variables<tmpl::append
                            <swsh_ccm_tags_to_generate,
                            swsh_ccm_tags_to_compute>>,
      ::Tags::Variables<swsh_boundary_ccm_tags_to_generate>>,
      Spectral::Swsh::Tags::SwshInterpolator<Tags::CauchyAngularCoords>,
      Spectral::Swsh::Tags::SwshInterpolator<Tags::InertialAngularCoords>>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Evolve,
                             tmpl::list<Actions::UpdateGauge,
                             Actions::CalculatePsi0>>>;
};

struct metavariables {
  using component_list =
      tmpl::list<mock_characteristic_evolution<metavariables>>;
  enum class Phase { Initialization, Evolve, Exit };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.Actions.UpdateGauge",
                  "[Unit][Cce]") {
  MAKE_GENERATOR(gen);
  // limited l_max distribution because test depends on an analytic
  // basis function with factorials.
  UniformCustomDistribution<size_t> sdist{7, 10};
  const size_t l_max = sdist(gen);
  UniformCustomDistribution<size_t> sdist1{2, 3};
  const size_t number_of_radial_points = sdist1(gen);
  UniformCustomDistribution<double> coefficient_distribution{-2.0, 2.0};
  CAPTURE(l_max);

  using component = mock_characteristic_evolution<metavariables>;
  ActionTesting::MockRuntimeSystem<metavariables> runner{{l_max}};
  Variables<tmpl::append<real_tags_to_generate, real_tags_to_compute>>
      real_variables{Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
  Variables<swsh_tags_to_compute> swsh_variables{
      Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
  Variables<tmpl::append<swsh_ccm_tags_to_generate,
      swsh_ccm_tags_to_compute>> swsh_variables_ccm{
      Spectral::Swsh::number_of_swsh_collocation_points(l_max) *
      number_of_radial_points};
  Variables<swsh_boundary_ccm_tags_to_generate> swsh_boundary_variables_ccm{
      Spectral::Swsh::number_of_swsh_collocation_points(l_max)};

  tmpl::for_each<real_tags_to_generate>([&real_variables, &gen,
                                         &coefficient_distribution,
                                         &l_max](auto tag_v) noexcept {
    using tag = typename decltype(tag_v)::type;
    SpinWeighted<ComplexModalVector, 0> generated_modes{
        Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)};
    SpinWeighted<ComplexDataVector, 0> generated_data{
        Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
    for (auto& tensor_component : get<tag>(real_variables)) {
      Spectral::Swsh::TestHelpers::generate_swsh_modes<0>(
          make_not_null(&generated_modes.data()), make_not_null(&gen),
          make_not_null(&coefficient_distribution), 1, l_max);
      Spectral::Swsh::inverse_swsh_transform(
          l_max, 1, make_not_null(&generated_data), generated_modes);
      // aggressive filter to make the uniformly generated random modes
      // somewhat reasonable
      Spectral::Swsh::filter_swsh_boundary_quantity(
          make_not_null(&generated_data), l_max, l_max / 2);
      tensor_component = real(generated_data.data());
    }
  });

  tmpl::for_each<swsh_ccm_tags_to_generate>([&swsh_variables_ccm, &gen,
                                         &coefficient_distribution,
                                         &l_max,
                           &number_of_radial_points](auto tag_v) noexcept {
    using tag = typename decltype(tag_v)::type;
    const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    SpinWeighted<ComplexModalVector, 0> generated_modes{
        Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)};
    SpinWeighted<ComplexDataVector, 0> generated_data{
        Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
    for (auto& tensor_component : get<tag>(swsh_variables_ccm)) {
      SpinWeighted<ComplexDataVector, 2> angular_view_j;
     for(size_t i=0; i < number_of_radial_points; ++i) {
      Spectral::Swsh::TestHelpers::generate_swsh_modes<0>(
          make_not_null(&generated_modes.data()), make_not_null(&gen),
          make_not_null(&coefficient_distribution), 1, l_max);
      Spectral::Swsh::inverse_swsh_transform(
          l_max, 1, make_not_null(&generated_data), generated_modes);
      // aggressive filter to make the uniformly generated random modes
      // somewhat reasonable
      Spectral::Swsh::filter_swsh_boundary_quantity(
          make_not_null(&generated_data), l_max, l_max / 2);
      angular_view_j.set_data_ref(
                 tensor_component.data().data() + i * number_of_angular_points,
                 number_of_angular_points);
      angular_view_j.data() = generated_data.data();
      }
    }
  });

  tmpl::for_each<swsh_boundary_ccm_tags_to_generate>(
                                         [&swsh_boundary_variables_ccm, &gen,
                                         &coefficient_distribution,
                                         &l_max](auto tag_v) noexcept {
    using tag = typename decltype(tag_v)::type;
    SpinWeighted<ComplexModalVector, 0> generated_modes{
        Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)};
    SpinWeighted<ComplexDataVector, 0> generated_data{
        Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
    for (auto& tensor_component : get<tag>(swsh_boundary_variables_ccm)) {
      Spectral::Swsh::TestHelpers::generate_swsh_modes<0>(
          make_not_null(&generated_modes.data()), make_not_null(&gen),
          make_not_null(&coefficient_distribution), 1, l_max);
      Spectral::Swsh::inverse_swsh_transform(
          l_max, 1, make_not_null(&generated_data), generated_modes);
      // aggressive filter to make the uniformly generated random modes
      // somewhat reasonable
      Spectral::Swsh::filter_swsh_boundary_quantity(
          make_not_null(&generated_data), l_max, l_max / 2);
      tensor_component = generated_data.data();
    }
  });

  ActionTesting::emplace_component_and_initialize<component>(
      &runner, 0,
      {real_variables, swsh_variables, swsh_variables_ccm,
       swsh_boundary_variables_ccm, Spectral::Swsh::SwshInterpolator{},
       Spectral::Swsh::SwshInterpolator{}});
  auto expected_box = db::create<
      tmpl::append<component::simple_tags, db::AddSimpleTags<Tags::LMax>>>(
      std::move(real_variables), std::move(swsh_variables),
      std::move(swsh_variables_ccm), std::move(swsh_boundary_variables_ccm),
      Spectral::Swsh::SwshInterpolator{},
      Spectral::Swsh::SwshInterpolator{}, l_max);

  runner.set_phase(metavariables::Phase::Evolve);
  // apply the `UpdateGauge` action
  ActionTesting::next_action<component>(make_not_null(&runner), 0);

  // apply the corresponding mutators to the `expected_box`
  db::mutate_apply<GaugeUpdateAngularFromCartesian<
      Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords>>(
      make_not_null(&expected_box));
  db::mutate_apply<GaugeUpdateAngularFromCartesian<
      Tags::InertialAngularCoords, Tags::InertialCartesianCoords>>(
      make_not_null(&expected_box));
  db::mutate_apply<GaugeUpdateJacobianFromCoordinates<
      Tags::GaugeC, Tags::GaugeD, Tags::CauchyAngularCoords,
      Tags::CauchyCartesianCoords>>(make_not_null(&expected_box));
  db::mutate_apply<GaugeUpdateJacobianFromCoordinates<
      Tags::GaugeCnohat, Tags::GaugeDnohat, Tags::InertialAngularCoords,
      Tags::InertialCartesianCoords>>(make_not_null(&expected_box));
  db::mutate_apply<GaugeUpdateInterpolator<Tags::CauchyAngularCoords>>(
      make_not_null(&expected_box));
  db::mutate_apply<GaugeUpdateInterpolator<Tags::InertialAngularCoords>>(
      make_not_null(&expected_box));
  db::mutate_apply<GaugeUpdateOmega>(make_not_null(&expected_box));
  db::mutate_apply<GaugeUpdateOmeganohat>(make_not_null(&expected_box));

  tmpl::for_each<
      tmpl::append<real_tags_to_compute, swsh_tags_to_compute>>(
      [&runner, &expected_box](auto tag_v) noexcept {
        using tag = typename decltype(tag_v)::type;
        const auto& test_lhs =
            ActionTesting::get_databox_tag<component, tag>(runner, 0);
        const auto& test_rhs = db::get<tag>(expected_box);
        CHECK_ITERABLE_APPROX(test_lhs, test_rhs);
      });

  ActionTesting::next_action<component>(make_not_null(&runner), 0);

  db::mutate_apply<Interpolate_BondiJ>(make_not_null(&expected_box));
  db::mutate_apply<PreSwshDerivatives<Tags::Dy<
        Tags::BondiJ_Cauchyview>>>(make_not_null(&expected_box));
  db::mutate_apply<PreSwshDerivatives<Tags::Dy<Tags::Dy<
        Tags::BondiJ_Cauchyview>>>>(make_not_null(&expected_box));
  db::mutate_apply<VolumeWeyl<Tags::Psi0Match>>(make_not_null(&expected_box));
  db::mutate_apply<PreSwshDerivatives<Tags::Dy<
        Tags::Psi0Match>>>(make_not_null(&expected_box));
  db::mutate_apply<BoundaryWeyl>(make_not_null(&expected_box));

  tmpl::for_each<swsh_ccm_tags_to_compute>(
      [&runner, &expected_box](auto tag_v) noexcept {
        using tag = typename decltype(tag_v)::type;
        const auto& test_lhs =
            ActionTesting::get_databox_tag<component, tag>(runner, 0);
        const auto& test_rhs = db::get<tag>(expected_box);
        CHECK_ITERABLE_APPROX(test_lhs, test_rhs);
      });

  // verify the interpolators are the same by interpolating a random set of
  // modes.
  SpinWeighted<ComplexModalVector, 2> generated_modes{
      Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)};
  Spectral::Swsh::TestHelpers::generate_swsh_modes<2>(
      make_not_null(&generated_modes.data()), make_not_null(&gen),
      make_not_null(&coefficient_distribution), 1, l_max);

  const Spectral::Swsh::SwshInterpolator& computed_interpolator =
      ActionTesting::get_databox_tag<
          component,
          Spectral::Swsh::Tags::SwshInterpolator<Tags::CauchyAngularCoords>>(
          runner, 0);
  const Spectral::Swsh::SwshInterpolator& expected_interpolator = db::get<
      Spectral::Swsh::Tags::SwshInterpolator<Tags::CauchyAngularCoords>>(
      expected_box);

  SpinWeighted<ComplexDataVector, 2> interpolated_points_from_computed{
      Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
  SpinWeighted<ComplexDataVector, 2> interpolated_points_from_expected{
      Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
  computed_interpolator.interpolate(
      make_not_null(&interpolated_points_from_computed),
      Spectral::Swsh::libsharp_to_goldberg_modes(generated_modes, l_max));
  expected_interpolator.interpolate(
      make_not_null(&interpolated_points_from_expected),
      Spectral::Swsh::libsharp_to_goldberg_modes(generated_modes, l_max));
  CHECK_ITERABLE_APPROX(interpolated_points_from_computed,
                        interpolated_points_from_expected);


  const Spectral::Swsh::SwshInterpolator& computed_interpolator_inertial =
      ActionTesting::get_databox_tag<
          component,
          Spectral::Swsh::Tags::SwshInterpolator<Tags::InertialAngularCoords>>(
          runner, 0);
  const Spectral::Swsh::SwshInterpolator& expected_interpolator_inertial
      = db::get<
      Spectral::Swsh::Tags::SwshInterpolator<Tags::InertialAngularCoords>>(
      expected_box);

  SpinWeighted<ComplexDataVector, 2> interpolated_points_from_computed_inertial{
      Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
  SpinWeighted<ComplexDataVector, 2> interpolated_points_from_expected_inertial{
      Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
  computed_interpolator_inertial.interpolate(
      make_not_null(&interpolated_points_from_computed_inertial),
      Spectral::Swsh::libsharp_to_goldberg_modes(generated_modes, l_max));
  expected_interpolator_inertial.interpolate(
      make_not_null(&interpolated_points_from_expected_inertial),
      Spectral::Swsh::libsharp_to_goldberg_modes(generated_modes, l_max));
  CHECK_ITERABLE_APPROX(interpolated_points_from_computed_inertial,
                        interpolated_points_from_expected_inertial);
}
}  // namespace Cce
