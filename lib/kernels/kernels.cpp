#include <cstdint>
#include <cstdlib>

typedef struct MicroTensor {
    float* address;
    int* shape;
    int size;
};

extern "C"
void linear_impl(int32_t in_features, int32_t out_features, MicroTensor* inputTensor, MicroTensor* weightsTensor, MicroTensor* biasesTensor, MicroTensor* outputTensor) {
    float* input = inputTensor->address;
    float* weights = weightsTensor->address;
    float* biases = biasesTensor->address;
    float* output = outputTensor->address;

    for (int out = 0; out < out_features; out++) {
        output[out] = biases[out];
        for (int in = 0; in < in_features; in++) {
            output[out] += input[in] * weights[in];
        }
        weights += in_features;
    }
}
