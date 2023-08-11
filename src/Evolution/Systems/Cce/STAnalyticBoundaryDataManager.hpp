// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "Evolution/Systems/Cce/AnalyticSolutions/WorldtubeData.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/SpecBoundaryData.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Evolution/Systems/Cce/WorldtubeDataManager.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "NumericalAlgorithms/Spectral/SwshCoefficients.hpp"
#include "NumericalAlgorithms/Spectral/SwshTransform.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/System/ParallelInfo.hpp"

namespace Cce {
namespace Tags {
/// \cond
struct ObservationLMax;
/// \endcond
}  // namespace Tags

/// A boundary data manager that constructs the desired boundary data into
/// the `Variables` from the data provided by the analytic solution.
class STAnalyticBoundaryDataManager {
 public:
  // charm needs an empty constructor.
  STAnalyticBoundaryDataManager() = default;

  STAnalyticBoundaryDataManager(size_t l_max, double extraction_radius);

  /*!
   * \brief Update the `boundary_data_variables` entries for all tags in
   * `Tags::characteristic_worldtube_boundary_tags` to the boundary data from
   * the analytic solution at  `time`.
   *
   * \details This class retrieves metric boundary data from the
   * `Cce::Solutions::WorldtubeData` derived class that represents an analytic
   * solution, then dispatches to `Cce::create_bondi_boundary_data()` to
   * construct the Bondi values into the provided `Variables`
   */
  bool populate_hypersurface_boundary_data(
      gsl::not_null<
          Variables<Tags::st_worldtube_boundary_tags<Tags::BoundaryValue>>*>
          boundary_data_variables,
      double time,
      const Scalar<SpinWeighted<ComplexDataVector, 0>> bondi_r) const;

  size_t get_l_max() const { return l_max_; }

  void pup(PUP::er& p);

 private:
  size_t l_max_ = 0;
  double extraction_radius_ = std::numeric_limits<double>::signaling_NaN();
};
}  // namespace Cce
