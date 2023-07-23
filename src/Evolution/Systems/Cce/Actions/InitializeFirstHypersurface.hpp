// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Systems/Cce/Components/WorldtubeBoundary.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/ScriPlusValues.hpp"
#include "IO/Observer/Actions/GetLockPointer.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "hello_world/hello_world.hpp"

namespace Cce {
namespace Actions {

/*!
 * \ingroup ActionsGroup
 * \brief Given initial boundary data for \f$J\f$ and \f$\partial_r J\f$,
 * computes the initial hypersurface quantities \f$J\f$ and gauge values.
 *
 * \details This action is to be called after boundary data has been received,
 * but before the time-stepping evolution loop. So, it should be either late in
 * an initialization phase or early (before a `Actions::Goto` loop or similar)
 * in the `Evolve` phase.
 *
 * Internally, this dispatches to the call function of
 * `Tags::InitializeJ`, which designates a hypersurface initial data generator
 * chosen by input file options, `InitializeGauge`, and
 * `InitializeScriPlusValue<Tags::InertialRetardedTime>` to perform the
 * computations. Refer to the documentation for those mutators for mathematical
 * details.
 *
 * \note This action accesses the base tag `Cce::Tags::InitializeJBase`,
 * trusting that a tag that inherits from that base tag is present in the box or
 * the global cache. Typically, this tag should be added by the worldtube
 * boundary component, as the type of initial data is decided by the type of the
 * worldtube boundary data.
 */
template <bool EvolveCcm, typename BoundaryComponent>
struct InitializeFirstHypersurface {
  using const_global_cache_tags =
      tmpl::list<Tags::LMax, Tags::NumberOfRadialPoints>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    // In some contexts, this action may get re-run (e.g. self-start procedure)
    // In those cases, we do not want to alter the existing hypersurface data,
    // so we just exit. However, we do want to re-run the action each time
    // the self start 'reset's from the beginning
    if (db::get<::Tags::TimeStepId>(box).slab_number() > 0 or
        not db::get<::Tags::TimeStepId>(box).is_at_slab_boundary()) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }
    // some initialization schemes need the hdf5_lock so that they can read
    // their own input data from disk.
    auto hdf5_lock = Parallel::local_branch(
                         Parallel::get_parallel_component<
                             observers::ObserverWriter<Metavariables>>(cache))
                         ->template local_synchronous_action<
                             observers::Actions::GetLockPointer<
                                 observers::Tags::H5FileLock>>();
    if constexpr (tt::is_a_v<AnalyticWorldtubeBoundary, BoundaryComponent>) {
      db::mutate_apply<typename InitializeJ::InitializeJ<false>::mutate_tags,
                       typename InitializeJ::InitializeJ<false>::argument_tags>(
          db::get<Tags::InitializeJBase>(box), make_not_null(&box),
          make_not_null(hdf5_lock));
    } else {
      db::mutate_apply<
          typename InitializeJ::InitializeJ<EvolveCcm>::mutate_tags,
          typename InitializeJ::InitializeJ<EvolveCcm>::argument_tags>(
          db::get<Tags::InitializeJBase>(box), make_not_null(&box),
          make_not_null(hdf5_lock));
    }

    std::vector<std::vector<double>> pi;
    std::vector<std::vector<double>> spacetime_metric;
    std::vector<std::vector<std::vector<double>>> phi;

    auto& my_space_time = db::get<Cce::Tags::TestSpaceTimeMetric>(box);
    auto& my_phi = db::get<Cce::Tags::TestPhi>(box);
    auto& my_pi = db::get<Cce::Tags::TestPi>(box);

    ThisThisDataVector_to_std_vector(my_space_time, spacetime_metric);
    ThisThisDataVector_to_std_vector(my_pi, pi);
    ThisThisDataVector_to_tri_std_vector(my_phi, phi);

    std::vector<double> re_j;
    std::vector<double> im_j;

    std::vector<double> cauchy_x;
    std::vector<double> cauchy_y;
    std::vector<double> cauchy_z;
    std::vector<double> inertial_x;
    std::vector<double> inertial_y;
    std::vector<double> inertial_z;

    auto l_max = db::get<Tags::LMax>(box);
    auto number_of_radial_points = db::get<Tags::NumberOfRadialPoints>(box);

    double radius = 20.;

    initialize_j(re_j, im_j, cauchy_x, cauchy_y, cauchy_z, inertial_x,
                 inertial_y, inertial_z, l_max, number_of_radial_points,
                 spacetime_metric, pi, phi, radius);

    auto bondi_j = db::get<Tags::BondiJ>(box);

    double resres = 0;
    for (size_t iii = 0; iii < re_j.size(); iii++) {
      resres += pow(im_j.at(iii) - imag(get(bondi_j).data())[iii], 2);
      resres += pow(re_j.at(iii) - real(get(bondi_j).data())[iii], 2);
    }

    resres /= re_j.size();
    std::cout << "first time " << std::setprecision(30) << sqrt(resres) << " ";

    auto& dt_cauchy_cart = db::get<Cce::Tags::CauchyCartesianCoords>(box);

    double resres_cauchy = 0;
    for (size_t iii = 0; iii < get<0>(dt_cauchy_cart).size(); iii++) {
      resres_cauchy += pow(cauchy_x.at(iii) - get<0>(dt_cauchy_cart)[iii], 2);
      resres_cauchy += pow(cauchy_y.at(iii) - get<1>(dt_cauchy_cart)[iii], 2);
      resres_cauchy += pow(cauchy_z.at(iii) - get<2>(dt_cauchy_cart)[iii], 2);
    }
    resres_cauchy /= get<0>(dt_cauchy_cart).size();
    std::cout  << std::setprecision(30) << sqrt(resres_cauchy)
              << " "<< std::endl;

    db::mutate_apply<InitializeScriPlusValue<Tags::InertialRetardedTime>>(
        make_not_null(&box), db::get<::Tags::TimeStepId>(box).substep_time());
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }

  static void ThisThisDataVector_to_std_vector(
      const tnsr::aa<DataVector, 3>& pi,
      std::vector<std::vector<double>>& data) {
    const auto size = get<0, 0>(pi).size();
    std::vector<double> datattt(size);
    std::vector<double> datattx(size);
    std::vector<double> datatty(size);
    std::vector<double> datattz(size);
    std::vector<double> datatxx(size);
    std::vector<double> datatxy(size);
    std::vector<double> datatxz(size);
    std::vector<double> datatyy(size);
    std::vector<double> datatyz(size);
    std::vector<double> datatzz(size);
    for (unsigned int i = 0; i < size; i++) {
      datattt[i] = get<0, 0>(pi)[i];
      datattx[i] = get<0, 1>(pi)[i];
      datatty[i] = get<0, 2>(pi)[i];
      datattz[i] = get<0, 3>(pi)[i];
      datatxx[i] = get<1, 1>(pi)[i];
      datatxy[i] = get<1, 2>(pi)[i];
      datatxz[i] = get<1, 3>(pi)[i];
      datatyy[i] = get<2, 2>(pi)[i];
      datatyz[i] = get<2, 3>(pi)[i];
      datatzz[i] = get<3, 3>(pi)[i];
    }
    data.push_back(datattt);
    data.push_back(datattx);
    data.push_back(datatty);
    data.push_back(datattz);
    data.push_back(datatxx);
    data.push_back(datatxy);
    data.push_back(datatxz);
    data.push_back(datatyy);
    data.push_back(datatyz);
    data.push_back(datatzz);
  }

  static void ThisThisDataVector_to_tri_std_vector(
      const tnsr::iaa<DataVector, 3>& pi,
      std::vector<std::vector<std::vector<double>>>& data) {
    const auto size = pi.get(0, 0, 0).size();
    std::vector<double> datattt(size);
    std::vector<double> datattx(size);
    std::vector<double> datatty(size);
    std::vector<double> datattz(size);
    std::vector<double> datatxx(size);
    std::vector<double> datatxy(size);
    std::vector<double> datatxz(size);
    std::vector<double> datatyy(size);
    std::vector<double> datatyz(size);
    std::vector<double> datatzz(size);
    for (size_t ijj = 0; ijj < 3; ++ijj) {
      for (size_t i = 0; i < size; i++) {
        datattt[i] = pi.get(ijj, 0, 0)[i];
        datattx[i] = pi.get(ijj, 0, 1)[i];
        datatty[i] = pi.get(ijj, 0, 2)[i];
        datattz[i] = pi.get(ijj, 0, 3)[i];
        datatxx[i] = pi.get(ijj, 1, 1)[i];
        datatxy[i] = pi.get(ijj, 1, 2)[i];
        datatxz[i] = pi.get(ijj, 1, 3)[i];
        datatyy[i] = pi.get(ijj, 2, 2)[i];
        datatyz[i] = pi.get(ijj, 2, 3)[i];
        datatzz[i] = pi.get(ijj, 3, 3)[i];
      }
      std::vector<std::vector<double>> datafinal{
          datattt, datattx, datatty, datattz, datatxx,
          datatxy, datatxz, datatyy, datatyz, datatzz};
      data.push_back(datafinal);
    }
  }
};
}  // namespace Actions
}  // namespace Cce
