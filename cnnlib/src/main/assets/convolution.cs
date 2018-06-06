layout (local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

uniform float k[KENNEL_SIZE];
layout(binding = 0, rgba32f) readonly uniform  image2D input_image;
layout(binding = 1, rgba32f) writeonly uniform  image2D output_image;

shared vec4 scanline[32][32];

void main(void)
{
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    scanline[pos.x][pos.y] = imageLoad(input_image, pos);
    barrier();
    vec4 data = scanline[pos.x][pos.y];
    data.r = data.r + k[0] ;
    data.g = data.g + k[1];
    data.b = data.b + k[2];
    data.a = data.a + k[KENNEL_SIZE - 1];
    imageStore(output_image, pos.xy, data);
}