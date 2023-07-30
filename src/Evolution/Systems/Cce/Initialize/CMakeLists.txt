# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ConformalFactor.cpp
  InitializeJ.cpp
  InverseCubic.cpp
  NoIncomingRadiation.cpp
  ZeroNonSmooth.cpp
  InitializeSTPsi.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ConformalFactor.hpp
  InitializeJ.hpp
  InverseCubic.hpp
  InitializeSTPsi.hpp
  NoIncomingRadiation.hpp
  RegisterInitializeJWithCharm.hpp
  ZeroNonSmooth.hpp
  )
