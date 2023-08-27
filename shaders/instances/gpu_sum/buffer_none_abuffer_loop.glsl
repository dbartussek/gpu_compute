#version 460

#extension GL_KHR_shader_subgroup_basic: require

#include <prelude.glsl>

#include <raw_get_data/storage_buffer.glsl>
#include <get_data/loop.glsl>
#include <discarder/none.glsl>
#include <writer/atomic_buffer.glsl>

bool condition(int x, int y, int z, DATA_TYPE data) {
    return true;
}

DATA_TYPE get_identity() {
    return 0;
}
DATA_TYPE accumulate(DATA_TYPE acc, DATA_TYPE data) {
    return acc + data;
}
