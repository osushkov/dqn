
#pragma once

#include <vector>

namespace Util {

double RandInterval(double s, double e);
double GaussianSample(double mean, double sd);

std::vector<float> SoftmaxWeights(const std::vector<float> &in);
float SoftmaxWeightedAverage(const std::vector<float> &in, float temperature);
}
