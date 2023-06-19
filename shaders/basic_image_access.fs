#version 460
layout(location = 0) in float tex_coord;
layout(location = 0) out uvec4 f_color;
layout(set = 0, binding = 0, r32ui) uniform readonly uimage1D img;

void main() {
    f_color = imageLoad(img, int(tex_coord * imageSize(img).x));
}
