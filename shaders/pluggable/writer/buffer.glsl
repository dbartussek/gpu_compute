#include "../constants.glsl"

layout(set = 1, binding = 0, std430) writeonly buffer out_buffer {
    uint out_values[];
};

void main() {
    vec2 coord = tex_coord * vec2(TEXTURE_SIZE_X, TEXTURE_SIZE_Y);

    GetData d = get_data_discarder(int(coord.x), int(coord.y));
    out_values[int(coord.x)] = d.data;
}
