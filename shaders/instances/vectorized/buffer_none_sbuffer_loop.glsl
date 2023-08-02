#version 460

#define DATA_TYPE uvec4

#include "../../pluggable/prelude.glsl"

#include "../../pluggable/raw_get_data/storage_buffer.glsl"
#include "../../pluggable/get_data/loop.glsl"
#include "../../pluggable/discarder/none.glsl"
#include "../../pluggable/writer/buffer.glsl"

bool condition(int x, int y, int z, uvec4 data) {
    return true;
}

uvec4 get_identity() {
    return uvec4(0);
}
uvec4 accumulate(uvec4 acc, uvec4 data) {
    return acc + data;
}
