
#include "Util.hpp"
#include <cassert>
#include <cmath>
#include <cstdlib>

double Util::RandInterval(double s, double e) { return s + (e - s) * (rand() / (double)RAND_MAX); }

double Util::GaussianSample(double mean, double sd) {
  // Taken from GSL Library Gaussian random distribution.
  double x, y, r2;

  do {
    // choose x,y in uniform square (-1,-1) to (+1,+1)
    x = -1 + 2 * RandInterval(0.0, 1.0);
    y = -1 + 2 * RandInterval(0.0, 1.0);

    // see if it is in the unit circle
    r2 = x * x + y * y;
  } while (r2 > 1.0 || r2 == 0);

  // Box-Muller transform
  return mean + sd * y * sqrt(-2.0 * log(r2) / r2);
}

std::vector<float> Util::SoftmaxWeights(const std::vector<float> &in) {
  assert(in.size() > 0);

  std::vector<float> result(in.size());

  float maxVal = in[0];
  for (unsigned r = 0; r < in.size(); r++) {
    maxVal = fmax(maxVal, in[r]);
  }

  float sum = 0.0f;
  for (unsigned i = 0; i < in.size(); i++) {
    result[i] = expf(in[i] - maxVal);
    sum += result[i];
  }

  for (unsigned i = 0; i < result.size(); i++) {
    result[i] /= sum;
  }

  return result;
}

float Util::SoftmaxWeightedAverage(const std::vector<float> &in, float temperature) {
  assert(temperature > 0.0f);

  std::vector<float> tempAdjusted(in.size());
  for (unsigned i = 0; i < in.size(); i++) {
    tempAdjusted[i] = in[i] / temperature;
  }

  auto weights = SoftmaxWeights(tempAdjusted);

  float result = 0.0f;
  for (unsigned i = 0; i < in.size(); i++) {
    result += weights[i] * in[i];
  }
  return result;
}
