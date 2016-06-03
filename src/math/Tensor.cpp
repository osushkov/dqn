
#include "Tensor.hpp"
#include <algorithm>
#include <cassert>

using namespace math;

Tensor Tensor::Read(std::istream &in) {
  Tensor result;

  unsigned numLayers;
  in >> numLayers;

  for (unsigned i = 0; i < numLayers; i++) {
    int rows, cols;
    in >> rows;
    in >> cols;

    assert(rows > 0 && cols > 0);

    EMatrix layer(rows, cols);
    for (int y = 0; y < rows; y++) {
      for (int x = 0; x < cols; x++) {
        in >> layer(y, x);
      }
    }

    result.AddLayer(layer);
  }

  return result;
}

void Tensor::Write(std::ostream &out) {
  out << data.size() << std::endl;
  for (const auto &elem : data) {
    out << elem.rows() << std::endl;
    out << elem.cols() << std::endl;

    for (int y = 0; y < elem.rows(); y++) {
      for (int x = 0; x < elem.cols(); x++) {
        out << elem(y, x) << std::endl;
      }
    }
  }
}

unsigned Tensor::NumLayers(void) const { return this->data.size(); }
void Tensor::AddLayer(const EMatrix &m) { this->data.push_back(m); }

EMatrix &Tensor::operator()(unsigned index) {
  assert(index < data.size());
  return data[index];
}

const EMatrix &Tensor::operator()(unsigned index) const {
  assert(index < data.size());
  return data[index];
}

Tensor Tensor::operator*(const Tensor &t) const {
  Tensor result(*this);
  result *= t;
  return result;
}

Tensor Tensor::operator+(const Tensor &t) const {
  Tensor result(*this);
  result += t;
  return result;
}

Tensor Tensor::operator-(const Tensor &t) const {
  Tensor result(*this);
  result -= t;
  return result;
}

Tensor Tensor::operator*(float s) const {
  Tensor result(*this);
  result *= s;
  return result;
}

Tensor Tensor::operator/(float s) const {
  Tensor result(*this);
  result /= s;
  return result;
}

Tensor &Tensor::operator*=(const Tensor &t) {
  assert(this->NumLayers() == t.NumLayers());
  for (unsigned i = 0; i < NumLayers(); i++) {
    assert(data[i].rows() == t.data[i].rows());
    assert(data[i].cols() == t.data[i].cols());

    for (int y = 0; y < data[i].rows(); y++) {
      for (int x = 0; x < data[i].cols(); x++) {
        data[i](y, x) *= t.data[i](y, x);
      }
    }
  }
  return *this;
}

Tensor &Tensor::operator+=(const Tensor &t) {
  assert(this->NumLayers() == t.NumLayers());
  for (unsigned i = 0; i < NumLayers(); i++) {
    data[i] += t.data[i];
  }
  return *this;
}

Tensor &Tensor::operator-=(const Tensor &t) {
  assert(this->NumLayers() == t.NumLayers());
  for (unsigned i = 0; i < NumLayers(); i++) {
    data[i] -= t.data[i];
  }
  return *this;
}

Tensor &Tensor::operator*=(float s) {
  for_each(data, [=](EMatrix &m) { m *= s; });
  return *this;
}

Tensor &Tensor::operator/=(float s) {
  float inv = 1.0f / s;
  for_each(data, [=](EMatrix &m) { m *= inv; });
  return *this;
}
