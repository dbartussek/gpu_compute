#version 460

#include "../pluggable/prelude.glsl"

#include "../pluggable/raw_get_data/storage_buffer.glsl"
#include "../pluggable/get_data/loop.glsl"
#include "../pluggable/discarder/discard.glsl"
#include "../pluggable/writer/attachment.glsl"

bool condition(int x, int y, int z, uint data) {
    return true;
}

uint get_identity() {
    return 0;
}
uint accumulate(uint acc, uint data) {
    return acc + data;
}
