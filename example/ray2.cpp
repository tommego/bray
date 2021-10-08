#include <iostream>
#include "../bray/bray.h"
#include "float.h"

int main()
{
    int nx = 200;
    int ny = 100;
    int ns = 100;
    PPM_START(nx, ny);
    vec3 lower_left_corner(-2.0, -1.0, -1.0);
    vec3 horizontal(4.0, 0.0, 0.0);
    vec3 vertical(0.0, 2.0, 0.0);
    vec3 origin(0.0, 0.0, 0.0);

    checker_texture tex(new constant_texture(vec3(0.3,0.4,0.7)), new constant_texture(vec3(0.7,0.6,0.7)));

    hitable *hlist[7];
    // surfaces
    // hlist[0] = new sphere(vec3(0, -1000, -1), 1000, new lambert(new pernlin_noise_texture()));
    hlist[0] = new sphere(vec3(0, 3, -9), 3, new lambert(&tex));
    hlist[1] = new sphere(vec3(-6, 3, -9), 3, new metal(vec3(1.0, 0.1, 0.3), 0.2));
    hlist[2] = new sphere(vec3(6, 3, -9), 3, new dielectric(1.5));

    // lights
    hlist[3] = new sphere(vec3(10, 12, -9), 3, new diffuse_light(new constant_texture(vec3(1.0, 1.0, 1.0))));
    hlist[4] = new sphere(vec3(-10, 12, -9), 6, new diffuse_light(new constant_texture(vec3(1.0, 0.5, 1.0))));
    
    // planes
    hlist[5] = new xz_plane(new lambert(new pernlin_noise_texture()), vec3(0, 0, 0), 100, 100);
    hlist[6] = new yz_plane(new lambert(&tex), vec3(16, 0, 0), 1000, 1000);
    
    hitable *world = new hitable_list(hlist, 7);
    camera cam(vec3(-5, 16, 16), vec3(0, 0, -9), vec3(0, 1, 0), 45, float(nx) / float(ny));
    for(int j = ny - 1; j >= 0; j--)
    {
        for(int i = 0; i < nx; i++)
        {
            vec3 col(0, 0, 0);
            for(int s = 0; s < ns; s++)
            {
                float u = float(i + drand48()) / float(nx);
                float v = float(j + drand48()) / float(ny);
                ray r = cam.get_ray(u, v);
                col += color(r, world, 0);
            }
            col /= float(ns);
            col = vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));
            PPM_PIX_COLOR(col.r(), col.g(), col.b());
        }
    }
}
