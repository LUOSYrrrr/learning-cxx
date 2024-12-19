#include "../exercise.h"
#include <cstring> // Add this line
// READ: 类模板 <https://zh.cppreference.com/w/cpp/language/class_template>

template<class T>

struct Tensor4D {
    unsigned int shape[4];
    T *data;

    Tensor4D(unsigned int const shape_[4], T const *data_) {
        unsigned int size = 1;
        for (int i = 0; i < 4; i++) {
            if (shape_[i] == 0) {
                throw std::invalid_argument("Shape dimensions must be >= 1");
            }
            this->shape[i] = shape_[i];
            size *= shape_[i];
        }
        if (!data_) {
            throw std::invalid_argument("Input data pointer is null");
        }
        data = new T[size];
        std::memcpy(data, data_, size * sizeof(T));
    }
    ~Tensor4D() {
        delete[] data;
    }

    Tensor4D(Tensor4D const &) = delete;
    Tensor4D(Tensor4D &&) noexcept = delete;
    // 这个加法需要支持“单向广播”。
    // 具体来说，others 可以具有与 this 不同的形状，形状不同的维度长度必须为 1。
    // others 长度为 1 但 this 长度不为 1 的维度将发生广播计算。
    // 例如，this 形状为 [1, 2, 3, 4]，others 形状为 [1, 2, 1, 4]，
    // 则 this 与 others 相加时，3 个形状为 [1, 2, 1, 4] 的子张���各自与 others 对应项相加。
    Tensor4D &operator+=(Tensor4D const &others) {
        for (int i = 0; i < 4; i++) {
            if (others.shape[i] != 1 && this->shape[i] != others.shape[i]) {
                throw std::invalid_argument("Shapes are not compatible for broadcasting");
            }
        }
        unsigned int size = shape[0] * shape[1] * shape[2] * shape[3];
        for (unsigned int idx = 0; idx < size; ++idx) {
            unsigned int linearIdx = idx;
            unsigned int broadcastIdx = 0;
            unsigned int stride = 1;

            for (int dim = 3; dim >= 0; --dim) {
                unsigned int thisIdx = linearIdx % shape[dim];
                linearIdx /= shape[dim];

                if (others.shape[dim] != 1) {
                    broadcastIdx += thisIdx * stride;
                }
                stride *= others.shape[dim];
            }
            data[idx] += others.data[broadcastIdx];
        }
        return *this;
    }
};

template <typename T>
Tensor4D(unsigned int const shape[4], T const* data) -> Tensor4D<T>;
// ---- 不要修改以下代码 ----
int main(int argc, char **argv) {
    {
        unsigned int shape[]{1, 2, 3, 4};
        // clang-format off
        int data[]{
             1,  2,  3,  4,
             5,  6,  7,  8,
             9, 10, 11, 12,

            13, 14, 15, 16,
            17, 18, 19, 20,
            21, 22, 23, 24};
        // clang-format on
        auto t0 = Tensor4D(shape, data);
        auto t1 = Tensor4D(shape, data);
        t0 += t1;
        for (auto i = 0u; i < sizeof(data) / sizeof(*data); ++i) {
            ASSERT(t0.data[i] == data[i] * 2, "Tensor doubled by plus its self.");
        }
    }
    {
        unsigned int s0[]{1, 2, 3, 4};
        // clang-format off
        float d0[]{
            1, 1, 1, 1,
            2, 2, 2, 2,
            3, 3, 3, 3,

            4, 4, 4, 4,
            5, 5, 5, 5,
            6, 6, 6, 6};
        // clang-format on
        unsigned int s1[]{1, 2, 3, 1};
        // clang-format off
        float d1[]{
            6,
            5,
            4,

            3,
            2,
            1};
        // clang-format on

        auto t0 = Tensor4D(s0, d0);
        auto t1 = Tensor4D(s1, d1);
        t0 += t1;
        for (auto i = 0u; i < sizeof(d0) / sizeof(*d0); ++i) {
            ASSERT(t0.data[i] == 7.f, "Every element of t0 should be 7 after adding t1 to it.");
        }
    }
    {
        unsigned int s0[]{1, 2, 3, 4};
        // clang-format off
        double d0[]{
             1,  2,  3,  4,
             5,  6,  7,  8,
             9, 10, 11, 12,

            13, 14, 15, 16,
            17, 18, 19, 20,
            21, 22, 23, 24};
        // clang-format on
        unsigned int s1[]{1, 1, 1, 1};
        double d1[]{1};

        auto t0 = Tensor4D(s0, d0);
        auto t1 = Tensor4D(s1, d1);
        t0 += t1;
        for (auto i = 0u; i < sizeof(d0) / sizeof(*d0); ++i) {
            ASSERT(t0.data[i] == d0[i] + 1, "Every element of t0 should be incremented by 1 after adding t1 to it.");
        }
    }
}
