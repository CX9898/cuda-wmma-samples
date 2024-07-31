#pragma once

namespace dev {

template<typename T>
class vector {
 public:
  inline vector() : size_(0), capacity_(0), data_(nullptr) {};
  inline vector(int size);
  inline ~vector();

  inline __host__ __device__ size_t size() {
      return size_;
  }
  inline __host__ __device__ const T *data() const {
      return data_;
  }
  inline __host__ __device__ T *data() {
      return data_;
  }
  inline __device__ const T &operator[](size_t idx) const {
      return data_[idx];
  }
  inline __device__ T &operator[](size_t idx) {
      return data_[idx];
  }

 private:
  size_t size_;
  size_t capacity_;
  T *data_;
};

template<typename T>
inline vector<T>::vector(const int size) : vector() {
    size_ = size;
    capacity_ = size;
    cudaMalloc(reinterpret_cast<void **>(data_), size);
}

template<typename T>
inline vector<T>::~vector() {
    if (data_) { cudaFree(data_); }
}
} // namespace dev
