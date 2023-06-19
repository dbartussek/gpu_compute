#version 460

#include "../pluggable/raw_get_data/sampled_1D.glsl"
#include "../pluggable/get_data/many_calls.glsl"
#include "../pluggable/discarder/discard.glsl"
#include "../pluggable/writer/attachment.glsl"

bool condition(int x, int y, int z, uint data) {
    return true;
}
