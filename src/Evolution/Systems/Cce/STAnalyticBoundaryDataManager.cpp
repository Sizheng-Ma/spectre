// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/STAnalyticBoundaryDataManager.hpp"

#include <cstddef>
#include <utility>

#include "Evolution/Systems/Cce/AnalyticSolutions/WorldtubeData.hpp"

namespace Cce {
STAnalyticBoundaryDataManager::STAnalyticBoundaryDataManager(
    const size_t l_max, const double extraction_radius)
    : l_max_{l_max}, extraction_radius_{extraction_radius} {}

bool STAnalyticBoundaryDataManager::populate_hypersurface_boundary_data(
    const gsl::not_null<
        Variables<Tags::st_worldtube_boundary_tags<Tags::BoundaryValue>>*>
        boundary_data_variables,
    const double time) const {
  return true;
}

void STAnalyticBoundaryDataManager::pup(PUP::er& p) {
  p | l_max_;
  p | extraction_radius_;
}
}  // namespace Cce
