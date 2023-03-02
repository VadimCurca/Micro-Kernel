#include <cstdint>
#include <cstdlib>

extern "C"
void linear_impl(int32_t in_features, int32_t out_features, float* input, float* weights, float *biases, float* output) {
    for (int out = 0; out < out_features; out++) {
        output[out] = biases[out];
        for (int in = 0; in < in_features; in++) {
            output[out] += input[in] * weights[in];
        }
        weights += in_features;
    }
}
