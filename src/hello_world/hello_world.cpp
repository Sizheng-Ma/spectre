// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "hello_world/hello_world.hpp"

#include <boost/preprocessor.hpp>
#include <iostream>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Evolution/Systems/Cce/WorldtubeDataManager.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "Parallel/Printf.hpp"

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

std::string link_date() { return std::string(__TIMESTAMP__); }

std::string executable_name() {
  return std::string(BOOST_PP_STRINGIZE(EXECUTABLE_NAME));
}

std::string git_description() {
  return std::string(BOOST_PP_STRINGIZE(GIT_DESCRIPTION));
}

std::string git_branch() { return std::string(BOOST_PP_STRINGIZE(GIT_BRANCH)); }

void myprint() {
  std::cout << "hello"
            << "\n"
            << std::endl;
}

int mynewfunction(int a) {
  std::cout << a << "\n" << std::endl;
  return a + 2;
}

void print_data_vector() {
  DataVector a{1.0, 2.3, 8.9};
  Parallel::printf("%s\n", a);
}

void ccm_functions(std::vector<double>& psi0, const std::vector<double>& gh) {
  const DataVector gh_read{const_cast<double*>(gh.data()), gh.size()};

  Cce::BondiWorldtubeDataManager q;
  // Cce::Tags::BondiBeta q;

  DataVector dv_psi0 = gh_read * 2.;

  for (unsigned int i = 0; i < dv_psi0.size(); i++) {
    psi0.push_back(dv_psi0.at(i));
  }
}
