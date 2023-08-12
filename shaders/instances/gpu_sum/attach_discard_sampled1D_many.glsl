#version 460

#include <prelude.glsl>

#include <raw_get_data/sampled_1D.glsl>
#include <get_data/many_calls.glsl>
#include <discarder/discard.glsl>
#include <writer/attachment.glsl>

bool condition(int x, int y, int z, DATA_TYPE data) {
    return true;
}
