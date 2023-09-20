#version 460

#ifndef DATA_TYPE
#define DATA_TYPE uvec4
#endif

#include <prelude.glsl>

#include <raw_get_data/storage_buffer.glsl>
#include <get_data/loop.glsl>
#include <writer/buffer.glsl>

bool condition(int x, int y, int z, DATA_TYPE data) {
    return true;
}

DATA_TYPE get_identity() {
    return DATA_TYPE(0);
}
DATA_TYPE accumulate(DATA_TYPE acc, DATA_TYPE data) {
    return acc + data;
}
