#ifndef CONSTANTS
#define CONSTANTS

layout (constant_id = 0) const int TEXTURE_SIZE_X = 1;
layout (constant_id = 1) const int TEXTURE_SIZE_Y = 1;

#define pos_infinity uintBitsToFloat(0x7F800000)
#define neg_infinity uintBitsToFloat(0xFF800000)

#endif
