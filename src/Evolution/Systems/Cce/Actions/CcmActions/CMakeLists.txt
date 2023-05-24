# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  InitializeCcmTagsinGH.hpp
  Psi0Matching.hpp
  SendPsi0ToEvolution.hpp
  )
