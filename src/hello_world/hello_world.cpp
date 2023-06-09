// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "hello_world/hello_world.hpp"

#include <iostream>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "Parallel/Printf.hpp"
// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

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
  DataVector gh_read{gh.size()};
    Parallel::printf("%d\n", gh.size());
  for(unsigned int i = 0; i < gh.size(); i++) {
    gh_read.at(i)=gh.at(i);
    Parallel::printf("%f\n", gh.at(i));
  }
  DataVector dv_psi0{psi0.data(), psi0.size()};
  dv_psi0=gh_read;
  Parallel::printf("%s\n", gh_read);
}
