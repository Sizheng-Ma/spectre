// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/AnalyticBoundaryDataManager.hpp"

#include <cstddef>
#include <utility>

#include "Evolution/Systems/Cce/AnalyticSolutions/WorldtubeData.hpp"

namespace Cce {
AnalyticBoundaryDataManager::AnalyticBoundaryDataManager(
    const size_t l_max, const double extraction_radius,
    std::unique_ptr<Solutions::WorldtubeData> generator)
    : l_max_{l_max},
      generator_{std::move(generator)},
      extraction_radius_{extraction_radius} {}

bool AnalyticBoundaryDataManager::populate_hypersurface_boundary_data(
    const gsl::not_null<Variables<
        Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>*>
        boundary_data_variables,
    gsl::not_null<
        Variables<Tags::st_worldtube_boundary_tags<Tags::BoundaryValue>>*>
        st_boundary_data_variables,
    const double time) const {
  const auto boundary_tuple = generator_->variables(
      l_max_, time,
      tmpl::list<gr::Tags::SpacetimeMetric<DataVector, 3>,
                 gh::Tags::Pi<DataVector, 3>, gh::Tags::Phi<DataVector, 3>,
                 Tags::BondiSTPsi, Tags::BondiSTTheta,
                 Tags::Dr<Tags::BondiSTPsi>>{});
  const auto& spacetime_metric =
      get<gr::Tags::SpacetimeMetric<DataVector, 3>>(boundary_tuple);
  const auto& pi = get<gh::Tags::Pi<DataVector, 3>>(boundary_tuple);
  const auto& phi = get<gh::Tags::Phi<DataVector, 3>>(boundary_tuple);
  create_bondi_boundary_data(boundary_data_variables, phi, pi, spacetime_metric,
                             extraction_radius_, l_max_);

  const auto& st_psi = get<Cce::Tags::BondiSTPsi>(boundary_tuple);
  const auto& dt_psi = get<Cce::Tags::BondiSTTheta>(boundary_tuple);
  const auto& dr_psi =
      get<Cce::Tags::Dr<Cce::Tags::BondiSTPsi>>(boundary_tuple);

  get<Tags::BoundaryValue<Tags::BondiSTPsi>>(*st_boundary_data_variables) =
      st_psi;

  create_st_boundary_data(st_boundary_data_variables, phi, pi, spacetime_metric,
                          get(dr_psi).data(), get(dt_psi).data(),
                          extraction_radius_, l_max_);

  // auto& beta =
  //     get<Tags::BoundaryValue<Tags::BondiBeta>>(*boundary_data_variables);

  // get(beta) = get(beta) * 0.;
  // {
  //   auto& bondi_u =
  //       get<Tags::BoundaryValue<Tags::BondiU>>(*boundary_data_variables);

  //   get(bondi_u) = get(bondi_u) * 0.;
  // }

  // auto& bondi_dr_u = get<Tags::BoundaryValue<Tags::Dr<Tags::BondiU>>>(
  //     *boundary_data_variables);

  // get(bondi_dr_u) = get(bondi_dr_u) * 0.;

  // {
  //   auto& bondi_u =
  //       get<Tags::BoundaryValue<Tags::BondiQ>>(*boundary_data_variables);

  //   get(bondi_u) = get(bondi_u) * 0.;
  // }

  // {
  //   auto& bondi_u =
  //       get<Tags::BoundaryValue<Tags::BondiJ>>(*boundary_data_variables);

  //   get(bondi_u) = get(bondi_u) * 0.;
  // }

  // {
  //   auto& bondi_u = get<Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>>(
  //       *boundary_data_variables);

  //   get(bondi_u) = get(bondi_u) * 0.;
  // }

  // {
  //   auto& bondi_u =
  //       get<Tags::BoundaryValue<Tags::BondiH>>(*boundary_data_variables);

  //   get(bondi_u) = get(bondi_u) * 0.;
  // }

  // {
  //   auto& bondi_u = get<Tags::BoundaryValue<Tags::Du<Tags::BondiJ>>>(
  //       *boundary_data_variables);

  //   get(bondi_u) = get(bondi_u) * 0.;
  // }

  // {
  //   auto& bondi_u =
  //       get<Tags::BoundaryValue<Tags::BondiW>>(*boundary_data_variables);

  //   get(bondi_u) = get(bondi_u) * 0. - 2. / square(extraction_radius_);
  // }

  // {
  //   auto& bondi_u =
  //       get<Tags::BoundaryValue<Tags::DuRDividedByR>>(*boundary_data_variables);

  //   get(bondi_u) = get(bondi_u) * 0.;
  // }

  // {
  //   auto& bondi_u = get<Tags::BoundaryValue<Tags::Du<Tags::BondiR>>>(
  //       *boundary_data_variables);

  //   get(bondi_u) = get(bondi_u) * 0.;
  // }

  // {
  //   auto& bondi_u =
  //       get<Tags::BoundaryValue<Tags::BondiR>>(*boundary_data_variables);

  //   get(bondi_u) = get(bondi_u) * 0. + extraction_radius_;
  // }

  return true;
}

void AnalyticBoundaryDataManager::pup(PUP::er& p) {
  p | l_max_;
  p | extraction_radius_;
  p | generator_;
}
}  // namespace Cce
