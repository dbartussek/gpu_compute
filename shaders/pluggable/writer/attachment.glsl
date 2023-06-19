#include "../constants.glsl"

void main() {
    vec2 coord = tex_coord * vec2(TEXTURE_SIZE_X, TEXTURE_SIZE_Y);

    GetData d = get_data_discarder(int(coord.x), int(coord.y));
    f_color = uvec4(d.data, 0, 0, 0);
}
