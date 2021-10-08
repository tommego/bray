#ifndef BRAY_H
#define BRAY_H

#include <math.h>
#include <stdlib.h>
#include <iostream>

#define MAX_HIT_DEPTH 100

// ppm macros
#define PPM_START(nx, ny) \
    std::cout << "P3\n" << nx << " " << ny << "\n255\n"

#define PPM_PIX_COLOR(fr, fg, fb) \
    std::cout << int(255.5*std::max(fr, float(0))) << " " << int(255.5*std::max(fg, float(0))) << " " << int(255.5*std::max(fb, float(0))) << "\n";


/* -------------------------------------------- vector -------------------------------------*/

class vec3 {
    public:
    vec3() {}
    vec3(float e0, float e1, float e2){ e[0] = e0; e[1] = e1; e[2] = e2; }
    inline float x() const { return e[0]; }
    inline float y() const { return e[1]; }
    inline float z() const { return e[2]; }
    inline float r() const { return e[0]; }
    inline float g() const { return e[1]; }
    inline float b() const { return e[2]; }

    inline const vec3& operator+() const { return *this; };
    inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    inline float operator[](int i) const { return e[i]; }
    inline float& operator[](int i) { return e[i]; }

    inline vec3& operator+=(const vec3 &v2) { 
        e[0] += v2.e[0]; e[1] += v2.e[1]; e[2] += v2.e[2]; return *this; 
    }
    inline vec3& operator-=(const vec3 &v2){ 
        e[0] -= v2.e[0]; e[1] -= v2.e[1]; e[2] -= v2.e[2]; return *this; 
    }
    inline vec3& operator*=(const vec3 &v2){ 
        e[0] *= v2.e[0]; e[1] *= v2.e[1]; e[2] *= v2.e[2]; return *this; 
    }
    inline vec3& operator/=(const vec3 &v2){ 
        e[0] /= v2.e[0]; e[1] /= v2.e[1]; e[2] /= v2.e[2]; return *this; 
    }
    inline vec3& operator+=(const float t){ 
        e[0] += t; e[1] += t; e[2] += t; return *this; 
    }
    inline vec3& operator-=(const float t){ 
        e[0] -= t; e[1] -= t; e[2] -= t; return *this; 
    }
    inline vec3& operator*=(const float t){ 
        e[0] *= t; e[1] *= t; e[2] *= t; return *this; 
    }
    inline vec3& operator/=(const float t){ 
        e[0] /= t; e[1] /= t; e[2] /= t; return *this; 
    }

    inline float length() const { return sqrt(e[0]*e[0]+e[1]*e[1]+e[2]*e[2]); }
    inline float squared_length() const { return e[0]*e[0]+e[1]*e[1]+e[2]*e[2]; };
    inline void make_unit_vector(){
        float k = 1.0 / sqrt(e[0]*e[0]+e[1]*e[1]+e[2]*e[2]);
        e[0] *= k; e[1] *= k; e[2] *= k;
    }
    float e[3];
};

inline vec3 unit_vector(const vec3& v) {
    float k = 1.0 / sqrt(v.e[0]*v.e[0]+v.e[1]*v.e[1]+v.e[2]*v.e[2]);
    return vec3(v.e[0] * k, v.e[1] * k, v.e[2] * k);
}

inline vec3 operator+(const vec3 &v1, const vec3 &v2){
    return vec3(v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]);
}

inline vec3 operator-(const vec3 &v1, const vec3 &v2){
    return vec3(v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]);
}

inline vec3 operator*(const vec3 &v1, const vec3 &v2){
    return vec3(v1[0] * v2[0], v1[1] * v2[1], v1[2] * v2[2]);
}

inline vec3 operator/(const vec3 &v1, const vec3 &v2){
    return vec3(v1[0] / v2[0], v1[1] / v2[1], v1[2] / v2[2]);
}

inline vec3 operator+(const vec3& v, const float& t)
{
    return vec3(v.e[0] + t, v.e[1] + t, v.e[2] + t);
}

inline vec3 operator+(const float& t, const vec3& v)
{
    return vec3(v.e[0] + t, v.e[1] + t, v.e[2] + t);
}

inline vec3 operator-(const vec3& v, const float& t)
{
    return vec3(v.e[0] - t, v.e[1] - t, v.e[2] - t);
}

inline vec3 operator-(const float& t, const vec3& v)
{
    return vec3(v.e[0] - t, v.e[1] - t, v.e[2] - t);
}

inline vec3 operator*(const vec3& v, const float& t)
{
    return vec3(v.e[0] * t, v.e[1] * t, v.e[2] * t);
}

inline vec3 operator*(const float& t, const vec3& v)
{
    return vec3(v.e[0] * t, v.e[1] * t, v.e[2] * t);
}

inline vec3 operator/(const vec3& v, const float& t)
{
    return vec3(v.e[0] / t, v.e[1] / t, v.e[2] / t);
}
inline vec3 operator/(const float& t, const vec3& v)
{
    return vec3(v.e[0] / t, v.e[1] / t, v.e[2] / t);
}

inline float dot(const vec3 &v1, const vec3 &v2)
{
    return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

inline vec3 cross(const vec3 &v1, const vec3 &v2)
{
    return vec3(
        (v1.e[1]*v2.e[2] - v1.e[2]*v2.e[1]),
        (-(v1.e[0]*v2.e[2] - v1.e[2]*v2.e[0])),
        (v1.e[0]*v2.e[1] - v1.e[1]*v2.e[0])
        );
}

inline vec3 reflect(const vec3 &v, const vec3 n)
{
    return v - 2 * dot(v, n) * n;
}

/* -------------------------------------------- iostream -------------------------------------*/
inline std::istream& operator>>(std::istream &is, vec3 &t) {
    is >> t.e[0] >> t.e[1] >> t.e[2];
    return is;
}

inline std::ostream& operator<<(std::ostream &os, const vec3 &t){
    os << "vec3(" << t.e[0] << " " << t.e[1] << " " << t.e[2] << ")";
    return os;
}

/* --------------------------------------------- math utils -----------------------------------*/
inline bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
    vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0 - ni_over_nt * ni_over_nt * (1 - dt*dt);
    if(discriminant > 0)
    {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    }
    else
        return false;
}

float schlick(float cosine, float ref_idx) {
    float r0 = (1-ref_idx) / (1+ref_idx);
    r0 = r0 * r0;
    return r0 + (1-r0)*pow((1-cosine), 5);
}

vec3 random_in_unit_disk() {
    vec3 p;
    do{
        p = 2.0 * vec3(drand48(), drand48(), 0) - vec3(1,1,0);
    } while(dot(p, p) >= 1.0);
    return p;
}

vec3 random_in_unit_sphere() {
    vec3 p;
    do {
        p = 2.0 * vec3 (drand48(), drand48(), drand48()) - vec3(1,1,1);
    } while(p.squared_length() >= 1.0);
    return p;
}

inline float trilinear_interp(float c[2][2][2], float u, float v, float w) {
    float accum = 0;
    for (int i = 0; i < 2; i++){
        for(int j = 0; j < 2; j++) {
            for(int k = 0; k < 2; k++) {
                accum += (i*u + (1-i)*(1-u)) *
                         (j*v + (1-j)*(1-v)) *
                         (k*w + (1-k)*(1-w)) * c[i][j][k];
            }
        }
    }
    return accum;
}

class perlin {
public:
    float noise(const vec3& p) const {
        float u = p.x() - floor(p.x());
        float v = p.y() - floor(p.y());
        float w = p.z() - floor(p.z());
        u = u * u * (3-2*u);
        v = v * v * (3-2*v);
        w = w * w * (3-2*w);
        int i = floor(p.x());
        int j = floor(p.y());
        int k = floor(p.z());
        float c[2][2][2];
        for(int di = 0; di < 2; di++){
            for(int dj = 0; dj < 2; dj++) {
                for(int dk = 0; dk < 2; dk++) {
                    c[di][dj][dk] = ranfloat[perm_x[(i+di) & 255] ^ perm_y[(j+dj) & 255] ^ perm_z[(k + dk) & 255]];
                }
            }
        }
        return trilinear_interp(c, u, v, w);
        // return ranfloat[perm_x[(i) & 255] ^ perm_y[(j) & 255] ^ perm_z[(k) & 255]];
    }

    float turb(const vec3& p, int depth = 7) const {
        float accum = 0;
        vec3 temp_p = p;
        float weight = 1.0;
        for(int i = 0 ; i < depth; i ++) {
            accum += weight * noise(temp_p);
            weight *= 0.5;
            temp_p *= 2;
        }
        return fabs(accum);
    }

    static float *ranfloat;
    static int *perm_x;
    static int *perm_y;
    static int *perm_z;
};

static float* perlin_generate() {
    float *p = new float[256];
    for(int i = 0; i < 256; ++i)
    {
        p[i] = drand48();
    }
    return p;
}

void permute(int *p, int n) {
    for (int i = n -1; i > 0; i--) {
        int target = int(drand48() * (i+1));
        int tmp = p[i];
        p[i] = p[target];
        p[target] = tmp;
    }
}

static int* perlin_generate_perm() {
    int *p = new int[256];
    for(int i = 0; i < 256; i++)
    {
        p[i] = i;
    }
    permute(p, 256);
    return p;
}

float *perlin::ranfloat = perlin_generate();
int *perlin::perm_x = perlin_generate_perm();
int *perlin::perm_y = perlin_generate_perm();
int *perlin::perm_z = perlin_generate_perm();

/* -------------------------------------------- ray -------------------------------------*/
class ray 
{
public:
    ray() {}
    ray(const vec3& a, const vec3 &b) { A = a; B = b; }
    vec3 origin() const { return A; }
    vec3 direction() const { return B; }
    vec3 point_at_parameter(float t) const { return A + t * B; }

    vec3 A;
    vec3 B;
};

class material;

/* -------------------------------------------- hitable -------------------------------------*/
struct hit_record {
    float t;
    float u,v;
    vec3 p;
    vec3 normal;
    material* p_material;
};

class hitable {
public:
    virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
    bool visible;
    material* p_material;
};

class hitable_list: public hitable {
public:
    hitable_list();
    hitable_list(hitable **l, int n) { list = l; list_size = n; }
    virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
        hit_record temp_rec;
        bool hit_anything = false;
        double closest_so_far = t_max;
        for(int i = 0; i < list_size; i++)
        {
            if(list[i]->hit(r, t_min, closest_so_far, temp_rec))
            {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
                rec.p_material = list[i]->p_material;
            }
        }
        return hit_anything;
    }
    hitable **list;
    int list_size;
};

/* -------------------------------------------- geometry -------------------------------------*/
class sphere: public hitable {
public:
    sphere() {}
    sphere(vec3 cen, float r, material* mat) : center(cen), radius(r) { p_material = mat; } 
    virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const
    {
        vec3 oc = r.origin() - center;
        float a = dot(r.direction() , r.direction());
        float b = dot(oc, r.direction());
        float c = dot(oc, oc) - radius * radius;
        float discrimnant = b * b - a * c;
        if(discrimnant > 0)
        {
            float temp = (-b - sqrt(b*b-a*c))/a;
            if(temp < t_max && temp > t_min)
            {
                rec.t = temp;
                rec.p = r.point_at_parameter(rec.t);
                rec.normal = (rec.p - center) / radius;
                return true;
            }
            temp = (-b + sqrt(b*b - a*c))/a;
            if(temp < t_max && temp > t_min) {
                rec.t = temp;
                rec.p = r.point_at_parameter(rec.t);
                rec.normal = (rec.p - center) / radius;
                return true;
            }
        }
        return false;
    }
    vec3 center;
    float radius;
};

class xy_rect : public hitable {
public:
    xy_rect() {}
    xy_rect(float _x0, float _x1, float _y0, float _y1, float _k, material* mat) : x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k) { p_material = mat; }
    virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
        float t = (k-r.origin().z()) / r.direction().z();
        if(t < t_min || t > t_max) return false;
        float x = r.origin().x() + t * r.direction().x();
        float y = r.origin().y() + t * r.direction().y();
        if(x < x0 || x > x1 || y < y0 || y > y1) return false;
        rec.u = (x - x0)/(x1 - x);
        rec.v = (y - y0) / (y1 -y0);
        rec.t = t;
        rec.p_material = p_material;
        rec.p = r.point_at_parameter(t);
        rec.normal = vec3(0, 0, 1);
        return true;
    }
    // virtual bool bounding_box(float t0, float t1, )

    float x0, x1, y0, y1, k;
};

class xz_plane : public hitable {
public:
    xz_plane(material* mat, vec3 _pos, float _w, float _h, const vec3& _up = vec3(0, 1, 0))
    :pos(_pos),w(_w),h(_h),up(_up){ p_material = mat; }
    virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
        float t = (pos.y() - r.origin().y()) / r.direction().y();
        if(t < t_min || t > t_max) return false;
        float x = r.origin().x() + t * r.direction().x();
        float z = r.origin().z() + t * r.direction().z();
        if(x < pos.x() - w / 2 || x > pos.x() + w / 2 || z < pos.z() - h / 2 || z > pos.z() + h / 2) return false;
        rec.u = (x - (pos.x() - w / 2)) / w;
        rec.v = (z - (pos.z() - h / 2)) / h;
        rec.t = t;
        rec.p_material = p_material;
        rec.p = r.point_at_parameter(t);
        rec.normal = up;
        return true;
    }
    float w,h;
    vec3 pos,up;
};

class xy_plane : public hitable {
public:
    xy_plane(material* mat, vec3 _pos, float _w, float _h, const vec3& _up = vec3(0, 0, 1))
    :pos(_pos),w(_w),h(_h),up(_up){ p_material = mat; }
    virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
        float t = (pos.z() - r.origin().z()) / r.direction().z();
        if(t < t_min || t > t_max) return false;
        float x = r.origin().x() + t * r.direction().x();
        float y = r.origin().y() + t * r.direction().y();
        if(x < pos.x() - w / 2 || x > pos.x() + w / 2 || y < pos.y() - h / 2 || y > pos.y() + h / 2) return false;
        rec.u = (x - (pos.x() - w / 2)) / w;
        rec.v = (y - (pos.y() - h / 2)) / h;
        rec.t = t;
        rec.p_material = p_material;
        rec.p = r.point_at_parameter(t);
        rec.normal = up;
        return true;
    }
    float w,h;
    vec3 pos,up;
};

class yz_plane : public hitable {
public:
    yz_plane(material* mat, vec3 _pos, float _w, float _h, const vec3& _up = vec3(-1, 0, 0))
    :pos(_pos),w(_w),h(_h),up(_up){ p_material = mat; }
    virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
        float t = (pos.x() - r.origin().x()) / r.direction().x();
        if(t < t_min || t > t_max) return false;
        float y = r.origin().y() + t * r.direction().y();
        float z = r.origin().z() + t * r.direction().z();
        if(y < pos.y() - w / 2 || y > pos.y() + w / 2 || z < pos.z() - h / 2 || z > pos.z() + h / 2) return false;
        rec.u = (y - (pos.y() - w / 2)) / w;
        rec.v = (z - (pos.z() - h / 2)) / h;
        rec.t = t;
        rec.p_material = p_material;
        rec.p = r.point_at_parameter(t);
        rec.normal = up;
        return true;
    }
    float w,h;
    vec3 pos,up;
};

/* -------------------------------------------- camera -------------------------------------*/
class camera {
public:

    camera(vec3 look_from, vec3 look_at, vec3 up, float fov, float aspect, float aperture = 0.1, float focus_dist = -1)
    {
        eye = look_from;
        at = look_at;
        if(focus_dist <= 0)
            focus_dist = (eye - at).length();
        lens_radius = aperture / 2;
        float theta = fov * M_PI  / 180;
        float haf_h = tan(theta / 2.0);
        float haf_w = aspect * haf_h;
        origin = look_from;
        w = unit_vector(look_from - look_at);
        u = unit_vector(cross(up, w));
        v = cross(w, u);
        lower_left_corner = origin - haf_w * u * focus_dist - haf_h * v * focus_dist - w * focus_dist;
        horizontal = 2 * haf_w * u * focus_dist;
        vertical = 2 * haf_h * v * focus_dist;
    }

    ray get_ray(float s, float t) 
    { 
        vec3 rd = lens_radius * random_in_unit_disk();
        vec3 offset = u * rd.x() + v*rd.y();
        return ray(origin + offset, lower_left_corner + s * horizontal + t*vertical - origin - offset); 
    }

    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 eye;
    vec3 at;
    vec3 u, v, w;
    float lens_radius;
};

/* -------------------------------------------- texture --------------------------------------*/
class texture {
public:
    virtual vec3 value(float u, float v, const vec3& p) const = 0;
    void get_sphere_uv(const vec3&p, float& u, float& v) {
        float phi = atan2(p.z(),p.x());
        float theta = asin(p.y());
        u = 1 - (phi + M_PI) / (2 * M_PI);
        v = (theta + M_PI/ 2) / M_PI;
    }
};

class constant_texture : public texture {
public:
    constant_texture(){}
    constant_texture(vec3 c) : color(c){}
    virtual vec3 value(float u, float v, const vec3& p) const {
        return color;
    }

    vec3 color;
};

class checker_texture : public texture {
public: 
    checker_texture(){}
    checker_texture(texture *t0, texture *t1):even(t0),odd(t1){}
    virtual vec3 value(float u, float v, const vec3& p) const {
        float sines = sin(10*p.x())*sin(10*p.y())*sin(10*p.z());
        if(sines < 0)
            return odd->value(u, v, p);
        else
            return even->value(u, v, p);
    }

    texture *odd;
    texture *even;
};

class pernlin_noise_texture : public texture {
public:
    pernlin_noise_texture(float s = 3) : scale(s) {}
    virtual vec3 value(float u, float v, const vec3& p) const {
        return vec3(0.5,0.3,0.9) * 0.5 * (1 + sin(scale * p.z() + 10 * noise.turb(p)));
    }
    perlin noise;
    float scale;
};

/* -------------------------------------------- material -------------------------------------*/
class material {
public:
    virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered) const = 0;
    virtual vec3 emited(float u, float v, const vec3& p) const { return vec3(0, 0, 0); }
    virtual int emit_strength() { return 0; }
    texture *tex;
};

class lambert : public material {
public:
    lambert(texture* a) { tex = a; } 
    virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered) const {
        // vec3 reflected = reflect(r_in.direction(), rec.normal);
        vec3 target = rec.p + rec.normal + random_in_unit_sphere();
        scattered = ray(rec.p, target - rec.p);
        attenuation = tex->value(0, 0, rec.p);
        return true;
    }

};

class metal : public material {
public:
    metal(vec3 rgb = vec3(1,1,1), float vfuzz = 0.3) : color(rgb), fuzz(vfuzz) {}
    virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered) const {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere());
        attenuation = color;
        return (dot(scattered.direction(), rec.normal) > 0);
    }
    vec3 color;
    float fuzz;
};

class dielectric : public material {
public:
    dielectric(float ri) : ref_idx(ri){}
    virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered) const {
        vec3 outward_normal;
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        float ni_over_nt;
        attenuation = vec3(1, 1, 1);
        vec3 refracted;
        float cosine;
        float reflect_prob;
        if(dot(r_in.direction(), rec.normal) > 0) {
            outward_normal =  -rec.normal;
            ni_over_nt = ref_idx;
            cosine = ref_idx * dot(r_in.direction(), rec.normal) / r_in.direction().length();
        }
        else {
            outward_normal = rec.normal;
            ni_over_nt = 1.0 / ref_idx;
            cosine = -ref_idx * dot(r_in.direction(), rec.normal) / r_in.direction().length();
        }
        if(refract(r_in.direction(), outward_normal, ni_over_nt, refracted)) {
            // scattered = ray(rec.p, refracted);
            reflect_prob = schlick(cosine, ref_idx);
        }
        else {
            scattered = ray(rec.p, reflected);
            reflect_prob = 1.0;
        }
        if(drand48() < reflect_prob) {
            scattered = ray(rec.p, reflected);
        } else {
            scattered = ray(rec.p, refracted);
        }
        return true;
    }
    vec3 color;
    float ref_idx;
};

class bsdf : public material {
public:
    bsdf(vec3 vcolor=vec3(1.0,1.0, 1.0), float fmetal = 0.8, float froughness = 0.2, float fspecular = 0.7):color(vcolor),metal(fmetal),roughness(froughness),specular(fspecular){}
    virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered) const {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + random_in_unit_sphere() * roughness);
        attenuation = color * (1.0 - metal);
        return (dot(scattered.direction(), rec.normal) > 0);
    }

    vec3 color;
    vec3 emission;
    vec3 subsurface_color;
    float emiision_strenth;
    float metal;
    float specular;
    float roughness;
    float ior;
    float alpha;
    float subsurface;
    float subsurface_radius;
};

class diffuse_light : public material {
public:
    diffuse_light(texture* a, const int& s = 2, const bool& slo = false): tex_emit(a) { strength = s;}
    virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered) const { return false; }
    virtual vec3 emited(float u, float v, const vec3& p) const { return tex_emit->value(u, v, p);}
    virtual int emit_strength() { return strength; }
    texture* tex_emit;
    int strength;
};

/* ---------------------------------------- pixel render ---------------------------------*/
vec3 color(const ray& r, hitable * world, int depth)
{
    hit_record rec;
    if(world->hit(r, 0.001, MAXFLOAT, rec))
    {   
        ray scattered;
        vec3 attenuation(0, 0, 0);
        vec3 emit = rec.p_material->emited(rec.u, rec.v, rec.p);
        if(depth < MAX_HIT_DEPTH && rec.p_material->scatter(r, rec, attenuation, scattered))
            return emit + attenuation * color(scattered, world, depth + 1);
        else
            return emit;

    }
    else 
        return vec3(0, 0, 0);
}

#endif // BRAY_H
