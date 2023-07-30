// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/STWorldtubeDataManager.hpp"

#include <complex>
#include <cstddef>
#include <memory>
#include <mutex>
#include <utility>

#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/SpecBoundaryData.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "NumericalAlgorithms/Spectral/SwshCoefficients.hpp"
#include "NumericalAlgorithms/Spectral/SwshTransform.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Evolution/Systems/Cce/WorldtubeBufferUpdater.hpp"

namespace Cce {
RealSTWorldtubeDataManager::RealSTWorldtubeDataManager(
    std::unique_ptr<STWorldtubeBufferUpdater<cce_st_input_tags>> buffer_updater,
    const size_t l_max, const size_t buffer_depth,
    std::unique_ptr<intrp::SpanInterpolator> interpolator)
    : buffer_updater_{std::move(buffer_updater)},
      l_max_{l_max},
      interpolated_coefficients_{
          Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)},
      buffer_depth_{buffer_depth},
      interpolator_{std::move(interpolator)} {
  if (UNLIKELY(
          buffer_updater_->get_time_buffer().size() <
          2 * interpolator_->required_number_of_points_before_and_after())) {
    ERROR(
        "The specified buffer updater doesn't have enough time points to "
        "supply the requested interpolator. This almost certainly "
        "indicates that the corresponding file hasn't been created properly, "
        "but might indicate that the specified Interpolator requests too many "
        "points");
  }
  // This will actually change the buffer depth in the case where the buffer
  // depth passed to the constructor is too large for the worldtube file size.
  // In that case, the worldtube data wouldn't be able to fill the buffer, so
  // here we shrink the buffer depth down to be no larger than the length of the
  // worldtube file.
  if (UNLIKELY(buffer_updater_->get_time_buffer().size() <
               2 * interpolator_->required_number_of_points_before_and_after() +
                   buffer_depth_)) {
    buffer_depth_ =
        buffer_updater_->get_time_buffer().size() -
        2 * interpolator_->required_number_of_points_before_and_after();
  }
  coefficients_buffers_ = Variables<cce_st_input_tags>{
      square(l_max + 1) *
      (buffer_depth_ +
       2 * interpolator_->required_number_of_points_before_and_after())};
}

bool RealSTWorldtubeDataManager::populate_hypersurface_boundary_data(
    const gsl::not_null<
        Variables<Tags::st_worldtube_boundary_tags<Tags::BoundaryValue>>*>
        boundary_data_variables,
    const double time,
    const gsl::not_null<Parallel::NodeLock*> hdf5_lock) const {
  if (buffer_updater_->time_is_outside_range(time)) {
    return false;
  }
  {
    const std::lock_guard hold_lock(*hdf5_lock);
    buffer_updater_->update_buffers_for_time(
        make_not_null(&coefficients_buffers_), make_not_null(&time_span_start_),
        make_not_null(&time_span_end_), time, l_max_,
        interpolator_->required_number_of_points_before_and_after(),
        buffer_depth_);
  }
  auto interpolation_time_span = detail::create_span_for_time_value(
      time, 0, interpolator_->required_number_of_points_before_and_after(),
      time_span_start_, time_span_end_, buffer_updater_->get_time_buffer());

  const size_t buffer_span_size = time_span_end_ - time_span_start_;
  const size_t interpolation_span_size =
      interpolation_time_span.second - interpolation_time_span.first;

  DataVector time_points{
      buffer_updater_->get_time_buffer().data() + interpolation_time_span.first,
      interpolation_span_size};

  auto interpolate_from_column =
      [&time, &time_points, &buffer_span_size, &interpolation_time_span,
       &interpolation_span_size, this](auto data, size_t column) {
        const auto interp_val = interpolator_->interpolate(
            gsl::span<const double>(time_points.data(), time_points.size()),
            gsl::span<const std::complex<double>>(
                data + column * (buffer_span_size) +
                    (interpolation_time_span.first - time_span_start_),
                interpolation_span_size),
            time);
        return interp_val;
      };

  // the ComplexModalVectors should be provided from the buffer_updater_ in
  // 'Goldberg' format, so we iterate over modes and convert to libsharp
  // format.
  for (const auto libsharp_mode :
       Spectral::Swsh::cached_coefficients_metadata(l_max_)) {
    tmpl::for_each<cce_st_input_tags>(
        [this, &libsharp_mode, &interpolate_from_column](auto tag_v) {
          using tag = typename decltype(tag_v)::type;
          Spectral::Swsh::goldberg_modes_to_libsharp_modes_single_pair(
              libsharp_mode,
              make_not_null(&get(get<tag>(interpolated_coefficients_))), 0,
              interpolate_from_column(
                  get(get<tag>(coefficients_buffers_)).data().data(),
                  Spectral::Swsh::goldberg_mode_index(
                      l_max_, libsharp_mode.l,
                      static_cast<int>(libsharp_mode.m))),
              interpolate_from_column(
                  get(get<tag>(coefficients_buffers_)).data().data(),
                  Spectral::Swsh::goldberg_mode_index(
                      l_max_, libsharp_mode.l,
                      -static_cast<int>(libsharp_mode.m))));
        });
  }
  // just inverse transform the 'direct' tags
  tmpl::for_each<tmpl::transform<cce_st_input_tags,
                                 tmpl::bind<db::remove_tag_prefix, tmpl::_1>>>(
      [this, &boundary_data_variables](auto tag_v) {
        using tag = typename decltype(tag_v)::type;
        Spectral::Swsh::inverse_swsh_transform(
            l_max_, 1,
            make_not_null(
                &get(get<Tags::BoundaryValue<tag>>(*boundary_data_variables))),
            get(get<Spectral::Swsh::Tags::SwshTransform<tag>>(
                interpolated_coefficients_)));
      });
  return true;
}

std::unique_ptr<STWorldtubeDataManager> RealSTWorldtubeDataManager::get_clone()
    const {
  return std::make_unique<RealSTWorldtubeDataManager>(
      buffer_updater_->get_clone(), l_max_, buffer_depth_,
      interpolator_->get_clone());
}

std::pair<size_t, size_t> RealSTWorldtubeDataManager::get_time_span() const {
  return std::make_pair(time_span_start_, time_span_end_);
}

void RealSTWorldtubeDataManager::pup(PUP::er& p) {
  p | buffer_updater_;
  p | time_span_start_;
  p | time_span_end_;
  p | l_max_;
  p | buffer_depth_;
  p | interpolator_;
  if (p.isUnpacking()) {
    time_span_start_ = 0;
    time_span_end_ = 0;
    const size_t size_of_buffer =
        square(l_max_ + 1) *
        (buffer_depth_ +
         2 * interpolator_->required_number_of_points_before_and_after());
    coefficients_buffers_ = Variables<cce_st_input_tags>{size_of_buffer};
    interpolated_coefficients_ = Variables<cce_st_input_tags>{
        Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max_)};
  }
}

PUP::able::PUP_ID RealSTWorldtubeDataManager::my_PUP_ID = 0;
}  // namespace Cce
