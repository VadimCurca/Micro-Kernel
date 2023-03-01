#include <cstdint>
#include <cstdlib>

extern "C"
int add(int a, int b) {
    return a + b;
}

extern "C"
void increment(int32_t* buffer, size_t size) {
    for (int i = 0; i < size; i++) {
        buffer[i] += 1;
    }
}

extern "C"
void add_buffers(int32_t* buffer1, int32_t* buffer2, int32_t* output, size_t size) {
    for (int i = 0; i < size; i++) {
        output[i] = buffer1[i] + buffer2[i];
    }
}

struct Point {
    int32_t x;
    int32_t y;
};


extern "C"
void incrementStruct(Point* p) {
    p->x++;
    p->y++;
}

