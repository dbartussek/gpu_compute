#include "../constants.glsl"

layout(location = 0) out uvec4 f_color;

void main() {
    ivec2 coord = get_coord();

    GetData d = get_data(coord.x, coord.y);
    f_color = uvec4(d.data, 0, 0, 0);
}
