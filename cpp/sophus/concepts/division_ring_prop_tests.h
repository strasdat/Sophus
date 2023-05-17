// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/concepts/division_ring.h"

namespace sophus {
namespace test {

template <concepts::DivisionRingConcept TRing>
struct DivisionRingTestSuite {
  using Ring = TRing;
  using Scalar = typename TRing::Scalar;
  static int constexpr kNumParams = TRing::kNumParams;
  static decltype(Ring::Impl::paramsExamples()) const kParamsExamples;

  using Params = Eigen::Vector<Scalar, kNumParams>;

  static void associativityTests(std::string ring_name) {
    for (size_t params_id = 0; params_id < kParamsExamples.size();
         ++params_id) {
      Params params = SOPHUS_AT(kParamsExamples, params_id);
      Ring g1 = Ring::fromParams(params);
      for (size_t params_id2 = 0; params_id2 < kParamsExamples.size();
           ++params_id2) {
        Params params2 = SOPHUS_AT(kParamsExamples, params_id2);
        Ring g2 = Ring::fromParams(params2);
        for (size_t params_id3 = 0; params_id3 < kParamsExamples.size();
             ++params_id3) {
          Params params3 = SOPHUS_AT(kParamsExamples, params_id3);
          Ring g3 = Ring::fromParams(params3);

          Ring left_hugging = (g1 * g2) * g3;
          Ring right_hugging = g1 * (g2 * g3);

          SOPHUS_ASSERT_NEAR(
              left_hugging.params(),
              right_hugging.params(),
              10.0 * kEpsilonSqrt<Scalar>,
              "`(g1*g2)*g3 == g1*(g2*g3)` Test for {}, #{}/#{}/#{}",
              ring_name,
              params_id,
              params_id2,
              params_id2);
        }
      }
    }
  }

  static void commutativityTests(std::string ring_name) {
    if (Ring::Impl::kIsCommutative) {
      for (size_t params_id = 0; params_id < kParamsExamples.size();
           ++params_id) {
        Params params = SOPHUS_AT(kParamsExamples, params_id);
        Ring g1 = Ring::fromParams(params);
        for (size_t params_id2 = 0; params_id2 < kParamsExamples.size();
             ++params_id2) {
          Params params2 = SOPHUS_AT(kParamsExamples, params_id2);
          Ring g2 = Ring::fromParams(params2);

          Ring left_hugging = g1 * g2;
          Ring right_hugging = g2 * g1;

          SOPHUS_ASSERT_NEAR(
              left_hugging.params(),
              right_hugging.params(),
              kEpsilonSqrt<Scalar>,
              "`g1 * g2 == g2 * g3` Test for {}, #{}/#{}",
              ring_name,
              params_id,
              params_id2);
        }
      }
    } else {
      size_t num_cases = 0;
      size_t num_commutativity = 0;
      for (size_t params_id = 0; params_id < kParamsExamples.size();
           ++params_id) {
        Params params = SOPHUS_AT(kParamsExamples, params_id);
        Ring g1 = Ring::fromParams(params);
        for (size_t params_id2 = 0; params_id2 < kParamsExamples.size();
             ++params_id2) {
          Params params2 = SOPHUS_AT(kParamsExamples, params_id2);
          Ring g2 = Ring::fromParams(params2);
          Ring left_hugging = g1 * g2;
          Ring right_hugging = g2 * g1;
          ++num_cases;
          if ((left_hugging.params() - right_hugging.params()).norm() <
              kEpsilonSqrt<Scalar>) {
            ++num_commutativity;
          }
        }
      }
      if (num_cases > 0) {
        Scalar commutativity_percentage =
            Scalar(num_commutativity) / Scalar(num_cases);
        SOPHUS_ASSERT_LE(commutativity_percentage, 0.75);
      }
    }
  }

  static void additionTests(std::string ring_name) {
    for (size_t params_id = 0; params_id < kParamsExamples.size();
         ++params_id) {
      Params params = SOPHUS_AT(kParamsExamples, params_id);
      Ring g = Ring::fromParams(params);

      SOPHUS_ASSERT_NEAR(
          g.params(),
          (g + Ring::zero()).params(),
          kEpsilonSqrt<Scalar>,
          "`g + 0 == g` Test for {}, #{}",
          ring_name,
          params_id);
    }
  }

  static void multiplicationTests(std::string ring_name) {
    for (size_t params_id = 0; params_id < kParamsExamples.size();
         ++params_id) {
      Params params = SOPHUS_AT(kParamsExamples, params_id);
      Ring g = Ring::fromParams(params);

      SOPHUS_ASSERT_NEAR(
          g.params(),
          (g * Ring::one()).params(),
          kEpsilonSqrt<Scalar>,
          "`g * 1 == g` Test for {}, #{}",
          ring_name,
          params_id);

      SOPHUS_ASSERT_NEAR(
          (g * g.inverse()).params(),
          Ring::one().params(),
          kEpsilonSqrt<Scalar>,
          "`g * 1 == g` Test for {}, #{}",
          ring_name,
          params_id);
    }
  }

  static void runAllTests(std::string ring_name) {
    associativityTests(ring_name);
    commutativityTests(ring_name);
    additionTests(ring_name);
    multiplicationTests(ring_name);
  }
};

template <concepts::DivisionRingConcept TRing>
decltype(TRing::Impl::paramsExamples())
    const DivisionRingTestSuite<TRing>::kParamsExamples =
        TRing::Impl::paramsExamples();

}  // namespace test
}  // namespace sophus
