# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY SizhengHelloWorld)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  hello_world.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  hello_world.hpp
  )

target_link_libraries(
 ${LIBRARY}
 PUBLIC
 Utilities
 Parallel
 DataStructures
 Charmxx::charmxx
 SystemUtilities
 CharmModuleInit
 )
