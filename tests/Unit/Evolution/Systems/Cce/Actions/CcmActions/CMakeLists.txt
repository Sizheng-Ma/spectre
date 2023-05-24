# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Cce_Actions_CcmActions")

set(LIBRARY_SOURCES
  Test_Psi0Matching.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/Cce/Actions/CcmActions/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Cce
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )
