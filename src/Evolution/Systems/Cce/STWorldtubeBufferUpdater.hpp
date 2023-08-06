// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "DataStructures/ComplexModalVector.hpp"
#include "Evolution/Systems/Cce/WorldtubeBufferUpdater.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/Version.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Cce {
// namespace Tags {
// namespace detail {
// // tags for use in the buffers for the modal input worldtube data management
// // classes
// using SpatialMetric = gr::Tags::SpatialMetric<ComplexModalVector, 3>;
// using Shift = gr::Tags::Shift<ComplexModalVector, 3>;
// using Lapse = gr::Tags::Lapse<ComplexModalVector>;

// // radial derivative prefix tag to be used with the modal input worldtube
// data template <typename Tag> struct Dr : db::SimpleTag, db::PrefixTag {
//   using type = typename Tag::type;
//   using tag = Tag;
// };

// // tag for the string for accessing the quantity associated with `Tag` in
// // worldtube h5 file
// template <typename Tag>
// struct InputDataSet : db::SimpleTag, db::PrefixTag {
//   using type = std::string;
//   using tag = Tag;
// };
// }  // namespace detail
// }  // namespace Tags

/// the full set of tensors to be extracted from the worldtube h5 file
/// \cond
class RealSTWorldtubeH5BufferUpdater;
/// \endcond

template <typename BufferTags>
class STWorldtubeBufferUpdater : public PUP::able {
 public:
  using creatable_classes = tmpl::list<RealSTWorldtubeH5BufferUpdater>;

  WRAPPED_PUPable_abstract(STWorldtubeBufferUpdater);  // NOLINT

  virtual double update_buffers_for_time(
      gsl::not_null<Variables<BufferTags>*> buffers,
      gsl::not_null<size_t*> time_span_start,
      gsl::not_null<size_t*> time_span_end, double time,
      size_t computation_l_max, size_t interpolator_length,
      size_t buffer_depth) const = 0;

  virtual std::unique_ptr<STWorldtubeBufferUpdater> get_clone() const = 0;

  virtual bool time_is_outside_range(double time) const = 0;

  virtual size_t get_l_max() const = 0;

  virtual double get_extraction_radius() const = 0;

  virtual bool has_version_history() const = 0;

  virtual DataVector& get_time_buffer() = 0;
};

/// A `WorldtubeBufferUpdater` specialized to the CCE input worldtube H5 file
/// produced by the reduced SpEC format.
class RealSTWorldtubeH5BufferUpdater
    : public STWorldtubeBufferUpdater<cce_bondi_input_tags> {
 public:
  // charm needs the empty constructor
  RealSTWorldtubeH5BufferUpdater() = default;

  /// The constructor takes the filename of the SpEC h5 file that will be used
  /// for boundary data. The extraction radius can either be passed in directly,
  /// or if it takes the value `std::nullopt`, then the extraction radius is
  /// retrieved as an integer in the filename.
  explicit RealSTWorldtubeH5BufferUpdater(
      const std::string& cce_data_filename,
      std::optional<double> extraction_radius = std::nullopt);

  WRAPPED_PUPable_decl_template(RealSTWorldtubeH5BufferUpdater);  // NOLINT

  explicit RealSTWorldtubeH5BufferUpdater(CkMigrateMessage* /*unused*/) {}

  /// update the `buffers`, `time_span_start`, and `time_span_end` with
  /// time-varies-fastest, Goldberg modal data and the start and end index in
  /// the member `time_buffer_` covered by the newly updated `buffers`.
  double update_buffers_for_time(
      gsl::not_null<Variables<cce_bondi_input_tags>*> buffers,
      gsl::not_null<size_t*> time_span_start,
      gsl::not_null<size_t*> time_span_end, double time,
      size_t computation_l_max, size_t interpolator_length,
      size_t buffer_depth) const override;

  std::unique_ptr<STWorldtubeBufferUpdater<cce_bondi_input_tags>> get_clone()
      const override {
    return std::make_unique<RealSTWorldtubeH5BufferUpdater>(filename_);
  }

  /// The time can only be supported in the buffer update if it is between the
  /// first and last time of the input file.
  bool time_is_outside_range(const double time) const override {
    return time < time_buffer_[0] or
           time > time_buffer_[time_buffer_.size() - 1];
  }

  /// retrieves the l_max of the input file
  size_t get_l_max() const override { return l_max_; }

  /// retrieves the extraction radius. In most normal circumstances, this will
  /// not be needed for Bondi data.
  double get_extraction_radius() const override {
    if (not static_cast<bool>(extraction_radius_)) {
      ERROR(
          "Extraction radius has not been set, and was not successfully parsed "
          "from the filename. The extraction radius has been used, so must be "
          "set either by the input file or via the filename.");
    }
    return *extraction_radius_;
  }

  /// The time buffer is supplied by non-const reference to allow views to
  /// easily point into the buffer.
  ///
  /// \warning Altering this buffer outside of the constructor of this class
  /// results in undefined behavior! This should be supplied by const reference
  /// once there is a convenient method of producing a const view of a vector
  /// type.
  DataVector& get_time_buffer() override { return time_buffer_; }

  bool has_version_history() const override { return true; }

  /// Serialization for Charm++.
  void pup(PUP::er& p) override;

 private:
  void update_buffer(gsl::not_null<ComplexModalVector*> buffer_to_update,
                     const h5::Dat& read_data, size_t computation_l_max,
                     size_t time_span_start, size_t time_span_end,
                     bool is_real) const;

  std::optional<double> extraction_radius_ = std::nullopt;
  size_t l_max_ = 0;

  h5::H5File<h5::AccessType::ReadOnly> cce_data_file_;
  std::string filename_;

  tuples::tagged_tuple_from_typelist<
      db::wrap_tags_in<Tags::detail::InputDataSet, cce_bondi_input_tags>>
      dataset_names_;

  // stores all the times in the input file
  DataVector time_buffer_;
};
}  // namespace Cce
