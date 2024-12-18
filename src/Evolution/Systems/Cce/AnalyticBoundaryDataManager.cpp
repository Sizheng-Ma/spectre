// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/AnalyticBoundaryDataManager.hpp"

#include <cstddef>
#include <utility>

#include "Evolution/Systems/Cce/AnalyticSolutions/WorldtubeData.hpp"

namespace Cce {
AnalyticBoundaryDataManager::AnalyticBoundaryDataManager(
    const size_t l_max, std::unique_ptr<Solutions::WorldtubeData> generator)
    : l_max_{l_max}, generator_{std::move(generator)} {}

bool AnalyticBoundaryDataManager::populate_hypersurface_boundary_data(
    const gsl::not_null<Variables<
        Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>*>
        boundary_data_variables,
    const double time) const {
  const auto boundary_tuple = generator_->variables(
      l_max_, time,
      tmpl::list<gr::Tags::SpacetimeMetric<DataVector, 3>,
                 gh::Tags::Pi<DataVector, 3>, gh::Tags::Phi<DataVector, 3>>{});
  const auto& spacetime_metric =
      get<gr::Tags::SpacetimeMetric<DataVector, 3>>(boundary_tuple);
  const auto& pi = get<gh::Tags::Pi<DataVector, 3>>(boundary_tuple);
  const auto& phi = get<gh::Tags::Phi<DataVector, 3>>(boundary_tuple);
  //   create_bondi_boundary_data(boundary_data_variables, phi, pi,
  //   spacetime_metric,
  //                              extraction_radius_, l_max_);

  create_bondi_boundary_data_spacelike_char(boundary_data_variables, phi, pi,
                                            spacetime_metric, -time, l_max_);
  return true;
}

void AnalyticBoundaryDataManager::pup(PUP::er& p) {
  p | l_max_;
  p | generator_;
}
}  // namespace Cce
