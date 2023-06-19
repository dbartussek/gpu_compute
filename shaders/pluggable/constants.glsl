#ifndef CONSTANTS
#define CONSTANTS

layout (constant_id = 0) const int TEXTURE_SIZE_X = 1;
layout (constant_id = 1) const int TEXTURE_SIZE_Y = 1;

layout(location = 0) in vec2 tex_coord;
layout(location = 1) flat in uint instance_id;
layout(location = 0) out uvec4 f_color;

#endif
