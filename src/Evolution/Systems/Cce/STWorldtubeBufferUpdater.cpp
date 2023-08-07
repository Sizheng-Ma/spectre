// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/STWorldtubeBufferUpdater.hpp"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/Version.hpp"
#include "NumericalAlgorithms/Spectral/SwshCoefficients.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {

RealSTWorldtubeH5BufferUpdater::RealSTWorldtubeH5BufferUpdater(
    const std::string& cce_data_filename,
    const std::optional<double> extraction_radius)
    : cce_data_file_{cce_data_filename}, filename_{cce_data_filename} {
  get<Tags::detail::InputDataSet<
      Spectral::Swsh::Tags::SwshTransform<Tags::BondiSTPsi>>>(dataset_names_) =
      "KGPsi";

  // We assume that the filename has the extraction radius encoded as an
  // integer between the last occurrence of 'R' and the last occurrence of
  // '.'. This is the format provided by SpEC.
  const size_t r_pos = cce_data_filename.find_last_of('R');
  const size_t dot_pos = cce_data_filename.find_last_of('.');
  const std::string text_radius =
      cce_data_filename.substr(r_pos + 1, dot_pos - r_pos - 1);
  try {
    extraction_radius_ = static_cast<bool>(extraction_radius)
                             ? *extraction_radius
                             : std::stod(text_radius);
  } catch (const std::invalid_argument&) {
    // the extraction radius is typically not used in the Bondi system, so we
    // don't error if it isn't parsed from the filename. Instead, we'll just
    // error if the invalid extraction radius value is ever retrieved using
    // `get_extraction_radius`.
  }

  const auto& u_data = cce_data_file_.get<h5::Dat>("/KGPsi");
  const auto data_table_dimensions = u_data.get_dimensions();
  const Matrix time_matrix = u_data.get_data_subset(std::vector<size_t>{0}, 0,
                                                    data_table_dimensions[0]);
  time_buffer_ = DataVector{data_table_dimensions[0]};
  for (size_t i = 0; i < data_table_dimensions[0]; ++i) {
    time_buffer_[i] = time_matrix(i, 0);
  }
  l_max_ = sqrt(data_table_dimensions[1] / 2) - 1;
  cce_data_file_.close_current_object();
}

double RealSTWorldtubeH5BufferUpdater::update_buffers_for_time(
    const gsl::not_null<Variables<cce_st_input_tags>*> buffers,
    const gsl::not_null<size_t*> time_span_start,
    const gsl::not_null<size_t*> time_span_end, const double time,
    const size_t computation_l_max, const size_t interpolator_length,
    const size_t buffer_depth) const {
  if (*time_span_end >= time_buffer_.size()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (*time_span_end > interpolator_length and
      time_buffer_[*time_span_end - interpolator_length] > time) {
    // the next time an update will be required
    return time_buffer_[*time_span_end - interpolator_length + 1];
  }
  // find the time spans that are needed
  auto new_span_pair = detail::create_span_for_time_value(
      time, buffer_depth, interpolator_length, 0, time_buffer_.size(),
      time_buffer_);
  *time_span_start = new_span_pair.first;
  *time_span_end = new_span_pair.second;
  // load the desired time spans into the buffers
  tmpl::for_each<cce_st_input_tags>([this, &buffers, &time_span_start,
                                     &time_span_end,
                                     &computation_l_max](auto tag_v) {
    using tag = typename decltype(tag_v)::type;
    this->update_buffer(
        make_not_null(&get(get<tag>(*buffers)).data()),
        cce_data_file_.get<h5::Dat>(
            "/" + get<Tags::detail::InputDataSet<tag>>(dataset_names_)),
        computation_l_max, *time_span_start, *time_span_end,
        tag::type::type::spin == 0);
    cce_data_file_.close_current_object();
  });
  // the next time an update will be required
  return time_buffer_[std::min(*time_span_end - interpolator_length + 1,
                               time_buffer_.size() - 1)];
}

void RealSTWorldtubeH5BufferUpdater::update_buffer(
    const gsl::not_null<ComplexModalVector*> buffer_to_update,
    const h5::Dat& read_data, const size_t computation_l_max,
    const size_t time_span_start, const size_t time_span_end,
    const bool is_real) const {
  size_t number_of_columns = read_data.get_dimensions()[1];
  if (UNLIKELY(buffer_to_update->size() !=
               square(computation_l_max + 1) *
                   (time_span_end - time_span_start))) {
    ERROR("Incorrect storage size for the data to be loaded in.");
  }
  std::vector<size_t> cols(number_of_columns - 1);
  std::iota(cols.begin(), cols.end(), 1);
  Matrix data_matrix = read_data.get_data_subset(
      cols, time_span_start, time_span_end - time_span_start);
  *buffer_to_update = 0.0;
  for (size_t time_row = 0; time_row < time_span_end - time_span_start;
       ++time_row) {
    for (int l = 0; l <= static_cast<int>(std::min(computation_l_max, l_max_));
         ++l) {
      for (int m = -l; m <= l; ++m) {
        if (is_real) {
          if (m == 0) {
            (*buffer_to_update)[Spectral::Swsh::goldberg_mode_index(
                                    computation_l_max, static_cast<size_t>(l),
                                    m) *
                                    (time_span_end - time_span_start) +
                                time_row] =
                std::complex<double>(
                    data_matrix(time_row, static_cast<size_t>(square(l))), 0.0);
          } else if (m > 0) {
            (*buffer_to_update)[Spectral::Swsh::goldberg_mode_index(
                                    computation_l_max, static_cast<size_t>(l),
                                    m) *
                                    (time_span_end - time_span_start) +
                                time_row] =
                std::complex<double>(
                    data_matrix(time_row,
                                static_cast<size_t>(square(l) + 2 * m - 1)),
                    data_matrix(
                        time_row,
                        static_cast<size_t>(square(l) + 2 * m)));  // NOLINT
          } else {
            (*buffer_to_update)[Spectral::Swsh::goldberg_mode_index(
                                    computation_l_max, static_cast<size_t>(l),
                                    m) *
                                    (time_span_end - time_span_start) +
                                time_row] =
                (-m % 2 == 0 ? 1.0 : -1.0) *
                std::complex<double>(
                    data_matrix(time_row,
                                static_cast<size_t>(square(l) + 2 * -m - 1)),
                    -data_matrix(
                        time_row,
                        static_cast<size_t>(square(l) + 2 * -m)));  // NOLINT
          }
        } else {
          (*buffer_to_update)[Spectral::Swsh::goldberg_mode_index(
                                  computation_l_max, static_cast<size_t>(l),
                                  m) *
                                  (time_span_end - time_span_start) +
                              time_row] =
              std::complex<double>(
                  data_matrix(time_row,
                              2 * Spectral::Swsh::goldberg_mode_index(
                                      l_max_, static_cast<size_t>(l), m)),
                  data_matrix(time_row,
                              2 * Spectral::Swsh::goldberg_mode_index(
                                      l_max_, static_cast<size_t>(l), m) +
                                  1));
        }
      }
    }
  }
}

void RealSTWorldtubeH5BufferUpdater::pup(PUP::er& p) {
  p | time_buffer_;
  p | filename_;
  p | l_max_;
  p | extraction_radius_;
  p | dataset_names_;
  if (p.isUnpacking()) {
    cce_data_file_ = h5::H5File<h5::AccessType::ReadOnly>{filename_};
  }
}

PUP::able::PUP_ID RealSTWorldtubeH5BufferUpdater::my_PUP_ID = 0;
}  // namespace Cce
