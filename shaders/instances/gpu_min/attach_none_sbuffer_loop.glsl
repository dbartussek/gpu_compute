#version 460

#include <prelude.glsl>

#include <raw_get_data/storage_buffer.glsl>
#include <get_data/loop.glsl>
#include <writer/attachment.glsl>

bool condition(int x, int y, int z, DATA_TYPE data) {
    return true;
}

DATA_TYPE get_identity() {
    return pos_infinity;
}
DATA_TYPE accumulate(DATA_TYPE acc, DATA_TYPE data) {
    return min(acc, data);
}
