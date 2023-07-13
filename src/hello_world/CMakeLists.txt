# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY SizhengHelloWorld)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  # hello_world.cpp
  hello_world_test.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  # hello_world.hpp
  hello_world_test.hpp
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
 Boost::boost
 Spectral
 H5
 Libxsmm
 Cce
 )

# # add_custom_command
# add_custom_command(
# OUTPUT "libnew.a"
# COMMAND ar cqt libnew.a libSystemUtilities.a
# #COMMAND ${CMAKE_COMMAND} -E ar cqT -libSystemUtilities.a -l${CHARM_LDXX_FLAGS_FOR_ARCHIVE}
# )

# add_library(SizhengHelloWorld libnew.a)

