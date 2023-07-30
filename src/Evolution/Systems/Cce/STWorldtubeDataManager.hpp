// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <utility>

#include "DataStructures/DataBox/Tag.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/STWorldtubeBufferUpdater.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "Parallel/NodeLock.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {

/// \cond
class RealSTWorldtubeDataManager;
/// \endcond

class STWorldtubeDataManager : public PUP::able {
 public:
  using creatable_classes = tmpl::list<RealSTWorldtubeDataManager>;

  WRAPPED_PUPable_abstract(STWorldtubeDataManager);  // NOLINT

  virtual bool populate_hypersurface_boundary_data(
      gsl::not_null<
          Variables<Tags::st_worldtube_boundary_tags<Tags::BoundaryValue>>*>
          boundary_data_variables,
      double time, gsl::not_null<Parallel::NodeLock*> hdf5_lock) const = 0;

  virtual std::unique_ptr<STWorldtubeDataManager> get_clone() const = 0;

  virtual size_t get_l_max() const = 0;

  virtual std::pair<size_t, size_t> get_time_span() const = 0;
};

class RealSTWorldtubeDataManager : public STWorldtubeDataManager {
 public:
  // charm needs an empty constructor.
  RealSTWorldtubeDataManager() = default;

  RealSTWorldtubeDataManager(
      std::unique_ptr<STWorldtubeBufferUpdater<cce_st_input_tags>>
          buffer_updater,
      size_t l_max, size_t buffer_depth,
      std::unique_ptr<intrp::SpanInterpolator> interpolator);

  WRAPPED_PUPable_decl_template(RealSTWorldtubeDataManager);  // NOLINT

  explicit RealSTWorldtubeDataManager(CkMigrateMessage* /*unused*/) {}

  bool populate_hypersurface_boundary_data(
      gsl::not_null<
          Variables<Tags::st_worldtube_boundary_tags<Tags::BoundaryValue>>*>
          boundary_data_variables,
      double time, gsl::not_null<Parallel::NodeLock*> hdf5_lock) const override;

  std::unique_ptr<STWorldtubeDataManager> get_clone() const override;

  size_t get_l_max() const override { return l_max_; }

  std::pair<size_t, size_t> get_time_span() const override;

  /// Serialization for Charm++.
  void pup(PUP::er& p) override;  // NOLINT

 private:
  std::unique_ptr<STWorldtubeBufferUpdater<cce_st_input_tags>> buffer_updater_;
  mutable size_t time_span_start_ = 0;
  mutable size_t time_span_end_ = 0;
  size_t l_max_ = 0;

  mutable Variables<cce_st_input_tags> interpolated_coefficients_;

  mutable Variables<cce_st_input_tags> coefficients_buffers_;

  size_t buffer_depth_ = 0;

  std::unique_ptr<intrp::SpanInterpolator> interpolator_;
};
}  // namespace Cce
