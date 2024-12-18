# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  BouncingBlackHole.cpp
  WorldtubeData.cpp
  SphericalMetricData.cpp
  GaugeWave.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BouncingBlackHole.hpp
  WorldtubeData.hpp
  SphericalMetricData.hpp
  GaugeWave.hpp
  )
