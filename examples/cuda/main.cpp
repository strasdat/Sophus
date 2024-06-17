#include <cmath>
#include <iostream>
#include <vector>

extern void cudaVecAdd(float* A, float* B, float* C, int N);

int main() {
  int N = 1024;
  std::vector<float> h_A(N, 0);
  std::vector<float> h_B(N, 0);
  std::vector<float> h_C(N, 0);

  // Initialize vectors
  for (int i = 0; i < N; ++i) {
    h_A[i] = sin(i) * sin(i);
    h_B[i] = cos(i) * cos(i);
  }

  // Call the CUDA kernel wrapper function
  cudaVecAdd(h_A.data(), h_B.data(), h_C.data(), N);

  // Check the result
  for (int i = 0; i < N; ++i) {
    float expected = h_A[i] + h_B[i];
    if (abs(h_C[i] - expected) > 1e-5) {
      std::cerr << "Result verification failed at element " << i << "!\n";
      return EXIT_FAILURE;
    }
  }

  std::cout << "Test PASSED\n";
  return 0;
}
