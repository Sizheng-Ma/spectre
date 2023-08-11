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
    const double time,
    const Scalar<SpinWeighted<ComplexDataVector, 0>> bondi_r) const {
  auto& psi =
      get<Tags::BoundaryValue<Tags::BondiSTPsi>>(*boundary_data_variables);
  auto& theta =
      get<Tags::BoundaryValue<Tags::BondiSTTheta>>(*boundary_data_variables);

  const size_t boundary_size =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max_);

  Spectral::Swsh::SpinWeightedSphericalHarmonic y_22{0, 2, 0};
  const auto& collocation_metadata =
      Spectral::Swsh::cached_collocation_metadata<
          Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max_);
  SpinWeighted<ComplexDataVector, 0> perturbed_j{boundary_size};
  for (const auto collocation_point : collocation_metadata) {
    const std::complex<double> y_22_factor =
        y_22.evaluate(collocation_point.theta, collocation_point.phi);
    perturbed_j.data()[collocation_point.offset] = y_22_factor;
  }
  const double u0 = 20;
  const double sigma0 = 1;
  double psi_boundary = exp(-0.5 * square(time - u0) / square(sigma0));
  get(psi).data() = psi_boundary * perturbed_j.data() / get(bondi_r).data();
  get(theta).data() = -(time - u0) / square(sigma0) * get(psi).data();
  return true;
}

void STAnalyticBoundaryDataManager::pup(PUP::er& p) {
  p | l_max_;
  p | extraction_radius_;
}
}  // namespace Cce
