#ifdef WIN32
#include <SDL.h>
#undef main
#else
#include <SDL2/SDL.h>
#endif

#include <GL/glew.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#define GLM_FORCE_SWIZZLE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include "obj_parser.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

std::string to_string(std::string_view str) { return std::string(str.begin(), str.end()); }

void sdl2_fail(std::string_view message) { throw std::runtime_error(to_string(message) + SDL_GetError()); }

void glew_fail(std::string_view message, GLenum error) {
    throw std::runtime_error(to_string(message) + reinterpret_cast<const char*>(glewGetErrorString(error)));
}

const char vertex_shader_source[] =
    R"(#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat3 normal_matrix;
uniform mat4 light_vp;

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_texcoord;

out vec3 world_pos;
out vec3 world_normal;
out vec2 texcoord;

void main()
{
    gl_Position = projection * view * model * vec4(in_position, 1.0);
    world_pos = (model * vec4(in_position, 1.0)).xyz;
    world_normal = normalize(normal_matrix * in_normal);
    texcoord = in_texcoord;
}
)";

const char fragment_shader_source[] =
    R"(#version 330 core

uniform sampler2D albedo_map;
uniform sampler2D map_d;
uniform sampler2D map_ka;
uniform sampler2D map_ks;
uniform sampler2D map_bump;

uniform float d_coef;
uniform vec3 camera_pos;
uniform vec3 sun_dir;
uniform vec3 sun_color;
uniform vec3 point_position;
uniform vec3 point_color;

uniform vec3 ka_coef;
uniform vec3 ks_coef;
uniform float ns;
uniform vec3 ambient_light;
uniform samplerCube env_map;
uniform float reflectivity;
uniform int apply_tonemap;
uniform sampler2D shadow_map;
uniform float shadow_bias;
uniform vec3 fog_color;
uniform float fog_density;
uniform int enable_fog;
uniform float volume_density;
uniform float volume_scatter;
uniform int enable_volume;
uniform mat4 light_vp;
uniform vec3 volume_ambient_color;

in vec2 texcoord;
in vec3 world_pos;
in vec3 world_normal;

layout (location = 0) out vec4 out_color;

vec3 tonemap_uncharted2_raw(vec3 x) {
    const float A = 0.15;
    const float B = 0.50;
    const float C = 0.10;
    const float D = 0.20;
    const float E = 0.02;
    const float F = 0.30;
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

vec3 tonemap_uncharted2(vec3 color) {
    return tonemap_uncharted2_raw(color) / tonemap_uncharted2_raw(vec3(11.2));
}

float compute_shadow(vec3 pos)
{
    vec4 lp = light_vp * vec4(pos, 1.0);
    vec3 proj = lp.xyz / lp.w;
    proj = proj * 0.5 + 0.5;
    if (proj.z > 1.0) return 0.0;
    float current = proj.z - shadow_bias;
    float shadow = 0.0;
    vec2 texel = 1.0 / vec2(textureSize(shadow_map, 0));
    for (int x = -1; x <= 1; ++x)
    for (int y = -1; y <= 1; ++y) {
        float pcf_depth = texture(shadow_map, proj.xy + vec2(x, y) * texel).r;
        shadow += current > pcf_depth ? 1.0 : 0.0;
    }
    shadow /= 9.0;
    return shadow;
}

void main()
{
    float alpha = d_coef * texture(map_d, texcoord).r;
    if (alpha < 0.5) discard;

    vec3 kd = texture(albedo_map, texcoord).rgb;
    vec3 n = normalize(world_normal);

    vec3 sigma_s = dFdx(world_pos);
    vec3 sigma_t = dFdy(world_pos);
    vec3 r1 = cross(sigma_t, n);
    vec3 r2 = cross(n, sigma_s);
    float det = dot(sigma_s, r1);
    vec2 tex_dx = dFdx(texcoord);
    vec2 tex_dy = dFdy(texcoord);
    float h_ll = texture(map_bump, texcoord).r;
    float h_lr = texture(map_bump, texcoord + tex_dx).r;
    float h_ul = texture(map_bump, texcoord + tex_dy).r;
    const float bump_scale = 0.015;
    float dBs = (h_lr - h_ll) * bump_scale;
    float dBt = (h_ul - h_ll) * bump_scale;
    vec3 surf_grad = sign(det) * (dBs * r1 + dBt * r2);
    n = normalize(abs(det) * n - surf_grad);

    vec3 v = normalize(camera_pos - world_pos);

    vec3 ka_tex = texture(map_ka, texcoord).rgb;
    vec3 ambient = ka_coef * ka_tex * ambient_light;

    vec3 ks_tex = texture(map_ks, texcoord).rgb;
    vec3 spec_color = ks_coef * ks_tex;

    vec3 color = ambient;

    // sun_dir points as surface -> sun
    vec3 ld = normalize(sun_dir);
    float ndl = max(dot(n, ld), 0.0);
    vec3 h = normalize(v + ld);
    float spec = 0.0;
    if (ndl > 0.0)
        spec = pow(max(dot(n, h), 0.0), ns);
    float shadow = compute_shadow(world_pos);
    color += (1.0 - shadow) * sun_color * (ndl * kd + spec_color * spec);

    vec3 lvec = point_position - world_pos;
    float dist = length(lvec);
    vec3 l = dist > 1e-4 ? lvec / dist : vec3(0.0, 1.0, 0.0);
    float atten = 1.0 / (1.0 + 2.0 * dist + 2.0 * dist * dist);
    float ndlp = max(dot(n, l), 0.0);
    vec3 hp = normalize(v + l);
    float specp = 0.0;
    if (ndlp > 0.0)
        specp = pow(max(dot(n, hp), 0.0), ns);
    color += point_color * atten * (ndlp * kd + spec_color * specp);

    if (reflectivity > 0.0) {
        vec3 refl = texture(env_map, reflect(-v, n)).rgb;
        color = mix(color, refl, clamp(reflectivity, 0.0, 1.0));
    }

    if (enable_volume != 0) {
        float dist = length(camera_pos - world_pos);
        int steps = 64;
        float step_len = dist / float(steps);
        vec3 ray_dir = (dist > 1e-6) ? (world_pos - camera_pos) / dist : vec3(0.0, 0.0, 1.0);
        vec3 sample_pos = camera_pos + ray_dir * (0.5 * step_len);
        float transmittance = 1.0;
        vec3 vol_light = vec3(0.0);
        float scatter = volume_density * step_len;
        for (int i = 0; i < steps; ++i) {
            float shadow_vol = compute_shadow(sample_pos);
            vol_light += transmittance * scatter * ((1.0 - shadow_vol) * sun_color * volume_scatter + volume_ambient_color);
            transmittance *= exp(-scatter);
            sample_pos += ray_dir * step_len;
        }
        color = color * transmittance + vol_light;
    }

    if (enable_fog != 0) {
        float dist = length(camera_pos - world_pos);
        float fog = exp(-fog_density * dist);
        color = mix(fog_color, color, clamp(fog, 0.0, 1.0));
    }

    if (apply_tonemap != 0) {
        color = tonemap_uncharted2(color);
    }

    out_color = vec4(color, alpha);
}
)";

const char depth_vertex_shader_source[] =
    R"(#version 330 core
layout (location = 0) in vec3 in_position;
uniform mat4 model;
uniform mat4 light_vp;
void main()
{
    gl_Position = light_vp * model * vec4(in_position, 1.0);
}
)";

const char depth_fragment_shader_source[] =
    R"(#version 330 core
void main() {}
)";

GLuint create_shader(GLenum type, const char* source) {
    GLuint result = glCreateShader(type);
    glShaderSource(result, 1, &source, nullptr);
    glCompileShader(result);
    GLint status;
    glGetShaderiv(result, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE) {
        GLint info_log_length;
        glGetShaderiv(result, GL_INFO_LOG_LENGTH, &info_log_length);
        std::string info_log(info_log_length, '\0');
        glGetShaderInfoLog(result, info_log.size(), nullptr, info_log.data());
        throw std::runtime_error("Shader compilation failed: " + info_log);
    }
    return result;
}

GLuint create_program(GLuint vertex_shader, GLuint fragment_shader) {
    GLuint result = glCreateProgram();
    glAttachShader(result, vertex_shader);
    glAttachShader(result, fragment_shader);
    glLinkProgram(result);

    GLint status;
    glGetProgramiv(result, GL_LINK_STATUS, &status);
    if (status != GL_TRUE) {
        GLint info_log_length;
        glGetProgramiv(result, GL_INFO_LOG_LENGTH, &info_log_length);
        std::string info_log(info_log_length, '\0');
        glGetProgramInfoLog(result, info_log.size(), nullptr, info_log.data());
        throw std::runtime_error("Program linkage failed: " + info_log);
    }

    return result;
}

void tolower(std::string& s) {
    for (char& ch : s) ch = std::tolower((unsigned char)ch);
}

struct MtlEntry {
    std::filesystem::path map_kd, map_ka, map_ks, map_d, map_bump;
    glm::vec3 ka_coef = glm::vec3(1.f);
    glm::vec3 ks_coef = glm::vec3(0.f);
    float ns = 32.f;
    float d_coef = 1.f;
};

static std::unordered_map<std::string, MtlEntry> load_mtl_maps(const std::filesystem::path& mtl_path) {
    std::unordered_map<std::string, MtlEntry> map;

    std::ifstream in(mtl_path);
    if (!in) return map;

    std::string line, cur;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string head;
        if (!(iss >> head)) continue;
        tolower(head);

        if (head == "newmtl") {
            iss >> cur;
            if (!map.count(cur)) map[cur] = MtlEntry{};
            continue;
        }

        if (cur.empty()) continue;

        if (head == "ka") {
            float r, g, b;
            if (iss >> r >> g >> b) map[cur].ka_coef = glm::vec3(r, g, b);
        } else if (head == "ks") {
            float r, g, b;
            if (iss >> r >> g >> b) map[cur].ks_coef = glm::vec3(r, g, b);
        } else if (head == "ns") {
            float v;
            if (iss >> v) map[cur].ns = v;
        } else if (head == "d") {
            float v;
            if (iss >> v) map[cur].d_coef = v;
        } else if (head == "map_kd" || head == "map_ka" || head == "map_ks" || head == "map_d" || head == "map_bump") {
            std::string rest;
            std::getline(iss, rest);
            while (!rest.empty() && (rest.front() == ' ' || rest.front() == '\t' || rest.front() == '\r'))
                rest.erase(rest.begin());
            while (!rest.empty() && (rest.back() == ' ' || rest.back() == '\t' || rest.back() == '\r')) rest.pop_back();
            if (rest.empty()) continue;

            std::istringstream args(rest);
            std::string token, last;
            while (args >> token) last = token;
            if (last.empty()) continue;

            for (char& ch : last) {
                if (ch == '\\') ch = '/';
            }
            if (head == "map_kd")
                map[cur].map_kd = std::filesystem::path{last};
            else if (head == "map_ka")
                map[cur].map_ka = std::filesystem::path{last};
            else if (head == "map_ks")
                map[cur].map_ks = std::filesystem::path{last};
            else if (head == "map_d")
                map[cur].map_d = std::filesystem::path{last};
            else
                map[cur].map_bump = std::filesystem::path{last};
        }
    }

    return map;
}

int main(int argc, char** argv) try {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) sdl2_fail("SDL_Init: ");

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_FRAMEBUFFER_SRGB_CAPABLE, 1);

    SDL_Window* window = SDL_CreateWindow("HW 3", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 800, 600,
                                          SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_MAXIMIZED);

    if (!window) sdl2_fail("SDL_CreateWindow: ");

    int width, height;
    SDL_GetWindowSize(window, &width, &height);

    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    if (!gl_context) sdl2_fail("SDL_GL_CreateContext: ");

    if (auto result = glewInit(); result != GLEW_NO_ERROR) glew_fail("glewInit: ", result);

    if (!GLEW_VERSION_3_3) throw std::runtime_error("OpenGL 3.3 is not supported");
    glEnable(GL_FRAMEBUFFER_SRGB);

    auto vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_shader_source);
    auto fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_shader_source);
    auto program = create_program(vertex_shader, fragment_shader);
    auto depth_vertex_shader = create_shader(GL_VERTEX_SHADER, depth_vertex_shader_source);
    auto depth_fragment_shader = create_shader(GL_FRAGMENT_SHADER, depth_fragment_shader_source);
    auto depth_program = create_program(depth_vertex_shader, depth_fragment_shader);

    GLuint model_location = glGetUniformLocation(program, "model");
    GLuint view_location = glGetUniformLocation(program, "view");
    GLuint projection_location = glGetUniformLocation(program, "projection");
    GLuint normal_matrix_location = glGetUniformLocation(program, "normal_matrix");
    GLuint light_vp_location = glGetUniformLocation(program, "light_vp");
    GLuint albedo_map_location = glGetUniformLocation(program, "albedo_map");
    GLuint map_d_location = glGetUniformLocation(program, "map_d");
    GLuint map_ka_location = glGetUniformLocation(program, "map_ka");
    GLuint map_ks_location = glGetUniformLocation(program, "map_ks");
    GLuint map_bump_location = glGetUniformLocation(program, "map_bump");
    GLuint d_coef_location = glGetUniformLocation(program, "d_coef");
    GLuint camera_pos_location = glGetUniformLocation(program, "camera_pos");
    GLuint sun_dir_location = glGetUniformLocation(program, "sun_dir");
    GLuint sun_color_location = glGetUniformLocation(program, "sun_color");
    GLuint point_position_location = glGetUniformLocation(program, "point_position");
    GLuint point_color_location = glGetUniformLocation(program, "point_color");
    GLuint ka_coef_location = glGetUniformLocation(program, "ka_coef");
    GLuint ks_coef_location = glGetUniformLocation(program, "ks_coef");
    GLuint ns_location = glGetUniformLocation(program, "ns");
    GLuint ambient_light_location = glGetUniformLocation(program, "ambient_light");
    GLuint env_map_location = glGetUniformLocation(program, "env_map");
    GLuint reflectivity_location = glGetUniformLocation(program, "reflectivity");
    GLuint apply_tonemap_location = glGetUniformLocation(program, "apply_tonemap");
    GLuint shadow_map_location = glGetUniformLocation(program, "shadow_map");
    GLuint shadow_bias_location = glGetUniformLocation(program, "shadow_bias");
    GLuint depth_light_vp_location = glGetUniformLocation(depth_program, "light_vp");
    GLuint depth_model_location = glGetUniformLocation(depth_program, "model");
    GLuint fog_color_location = glGetUniformLocation(program, "fog_color");
    GLuint fog_density_location = glGetUniformLocation(program, "fog_density");
    GLuint enable_fog_location = glGetUniformLocation(program, "enable_fog");
    GLuint volume_density_location = glGetUniformLocation(program, "volume_density");
    GLuint volume_scatter_location = glGetUniformLocation(program, "volume_scatter");
    GLuint enable_volume_location = glGetUniformLocation(program, "enable_volume");
    GLuint volume_ambient_color_location = glGetUniformLocation(program, "volume_ambient_color");

    glUseProgram(program);
    glUniform1i(albedo_map_location, 1);
    glUniform1i(map_ka_location, 2);
    glUniform1i(map_ks_location, 3);
    glUniform1i(map_bump_location, 4);
    glUniform1i(map_d_location, 5);
    glUniform1i(env_map_location, 6);
    glUniform1i(shadow_map_location, 7);
    glUniform1i(apply_tonemap_location, 1);

    glm::vec3 ambient_light(0.03f);
    glUniform3fv(ambient_light_location, 1, reinterpret_cast<float*>(&ambient_light));
    glm::vec3 fog_color(0.28f, 0.30f, 0.33f);
    float fog_density = 0.75f;
    float volume_density = 0.25f;
    float volume_scatter = 4.f;
    glUniform3fv(fog_color_location, 1, reinterpret_cast<float*>(&fog_color));
    glUniform1f(fog_density_location, fog_density);
    glUniform1f(volume_density_location, volume_density);
    glUniform1f(volume_scatter_location, volume_scatter);
    glm::vec3 volume_ambient_color = ambient_light * 0.2f;
    glUniform3fv(volume_ambient_color_location, 1, reinterpret_cast<float*>(&volume_ambient_color));

    if (argc <= 1) throw std::out_of_range("no scene path were passed as arg");
    std::filesystem::path scene_dir = std::filesystem::path(argv[1]);
    if (!std::filesystem::exists(scene_dir) || !std::filesystem::is_directory(scene_dir))
        throw std::runtime_error("Scene path must be an existing directory: " + scene_dir.string());

    std::filesystem::path obj_path = scene_dir / (scene_dir.filename().string() + ".obj");
    if (!std::filesystem::exists(obj_path)) {
        for (const auto& entry : std::filesystem::directory_iterator(scene_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".obj") {
                obj_path = entry.path();
                break;
            }
        }
    }
    if (!std::filesystem::exists(obj_path))
        throw std::runtime_error("No .obj file found in directory: " + scene_dir.string());

    obj_data scene = parse_obj(obj_path);

    std::filesystem::path project_root = std::filesystem::path(PROJECT_ROOT);
    std::filesystem::path bunny_path = project_root / "bunny.obj";
    if (!std::filesystem::exists(bunny_path)) throw std::runtime_error("Bunny OBJ is missing: " + bunny_path.string());
    obj_data bunny = parse_obj(bunny_path);

    auto compute_bbox = [](const obj_data& mesh) {
        glm::vec3 bb_min(std::numeric_limits<float>::infinity());
        glm::vec3 bb_max(-std::numeric_limits<float>::infinity());
        for (const auto& v : mesh.vertices) {
            glm::vec3 p(v.position[0], v.position[1], v.position[2]);
            bb_min = glm::min(bb_min, p);
            bb_max = glm::max(bb_max, p);
        }
        return std::pair{bb_min, bb_max};
    };

    auto [bb_min, bb_max] = compute_bbox(scene);
    auto [bunny_bb_min, bunny_bb_max] = compute_bbox(bunny);

    glm::vec3 scene_center = (bb_min + bb_max) * 0.5f;
    float scene_radius = glm::length(bb_max - bb_min) * 0.5f;
    float scene_scale = scene_radius > 0.f ? 1.0f / scene_radius : 1.0f;
    float sponza_floor_y = (bb_min.y - scene_center.y) * scene_scale;

    glm::vec3 bunny_center = (bunny_bb_min + bunny_bb_max) * 0.5f;
    float bunny_height = bunny_bb_max.y - bunny_bb_min.y;
    float normalized_scene_height = (bb_max.y - bb_min.y) * scene_scale;
    float bunny_target_height = normalized_scene_height * 0.12f;
    float bunny_scale = bunny_height > 0.f ? bunny_target_height / bunny_height : 0.05f;
    glm::vec3 bunny_base_position(0.05f, sponza_floor_y + 0.055f, 0.f);
    auto build_bunny_model = [&](const glm::vec3& pos) {
        glm::mat4 m = glm::translate(glm::mat4(1.f), pos);
        m = glm::scale(m, glm::vec3(bunny_scale));
        m = glm::translate(m, -glm::vec3(bunny_center.x, bunny_bb_min.y, bunny_center.z));
        return m;
    };

    glm::mat4 scene_model = glm::scale(glm::mat4(1.f), glm::vec3(scene_scale));
    scene_model = glm::translate(scene_model, -scene_center);
    glm::mat3 scene_normal_matrix = glm::transpose(glm::inverse(glm::mat3(scene_model)));

    glm::vec3 sun_dir = glm::normalize(glm::vec3(-0.3f, 1.f, -0.2f));  // surface -> sun (light comes from -sun_dir)
    glm::vec3 sun_color(1.0f, 0.95f, 0.9f);
    glm::vec3 point_color(2.5f, 2.2f, 2.1f);

    const float light_radius = 3.0f;
    auto compute_light_matrices = [&](const glm::vec3& dir) {
        glm::vec3 light_dir = -glm::normalize(dir);
        glm::vec3 center(0.f);
        glm::vec3 light_pos = center - light_dir * 4.0f;
        glm::vec3 up = (std::abs(light_dir.y) > 0.99f) ? glm::vec3(0.f, 0.f, 1.f) : glm::vec3(0.f, 1.f, 0.f);
        glm::mat4 light_view = glm::lookAt(light_pos, center, up);
        glm::mat4 light_proj = glm::ortho(-light_radius, light_radius, -light_radius, light_radius, 0.1f, 15.0f);
        return light_proj * light_view;
    };

    const int cubemap_size = 256;
    GLuint env_cubemap = 0;
    glGenTextures(1, &env_cubemap);
    glBindTexture(GL_TEXTURE_CUBE_MAP, env_cubemap);
    for (int face = 0; face < 6; ++face) {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, 0, GL_RGBA16F, cubemap_size, cubemap_size, 0, GL_RGBA,
                     GL_FLOAT, nullptr);
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP);

    GLuint env_fbo = 0, env_rbo = 0;
    glGenFramebuffers(1, &env_fbo);
    glGenRenderbuffers(1, &env_rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, env_rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, cubemap_size, cubemap_size);
    glBindFramebuffer(GL_FRAMEBUFFER, env_fbo);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, env_rbo);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    const int shadow_size = 2048;
    GLuint shadow_map = 0, shadow_fbo = 0;
    glGenTextures(1, &shadow_map);
    glBindTexture(GL_TEXTURE_2D, shadow_map);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, shadow_size, shadow_size, 0, GL_DEPTH_COMPONENT,
                 GL_UNSIGNED_INT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    float border_color[] = {1.f, 1.f, 1.f, 1.f};
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border_color);

    glGenFramebuffers(1, &shadow_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, shadow_fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, shadow_map, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    GLuint scene_vao, scene_vbo, scene_ebo;
    glGenVertexArrays(1, &scene_vao);
    glBindVertexArray(scene_vao);

    glGenBuffers(1, &scene_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, scene_vbo);
    glBufferData(GL_ARRAY_BUFFER, scene.vertices.size() * sizeof(scene.vertices[0]), scene.vertices.data(),
                 GL_STATIC_DRAW);

    glGenBuffers(1, &scene_ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, scene_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, scene.indices.size() * sizeof(scene.indices[0]), scene.indices.data(),
                 GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(obj_data::vertex), (void*)(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(obj_data::vertex), (void*)(12));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(obj_data::vertex), (void*)(24));

    GLuint bunny_vao, bunny_vbo, bunny_ebo;
    glGenVertexArrays(1, &bunny_vao);
    glBindVertexArray(bunny_vao);

    glGenBuffers(1, &bunny_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, bunny_vbo);
    glBufferData(GL_ARRAY_BUFFER, bunny.vertices.size() * sizeof(bunny.vertices[0]), bunny.vertices.data(),
                 GL_STATIC_DRAW);

    glGenBuffers(1, &bunny_ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bunny_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, bunny.indices.size() * sizeof(bunny.indices[0]), bunny.indices.data(),
                 GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(obj_data::vertex), (void*)(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(obj_data::vertex), (void*)(12));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(obj_data::vertex), (void*)(24));

    stbi_set_flip_vertically_on_load(true);
    std::ifstream in(obj_path);
    std::vector<std::string> mtllibs;
    struct Range {
        uint32_t offset = 0, cnt = 0;
        std::string mtl;
    };
    std::vector<Range> ranges;
    std::string line;
    std::string cur_mtl;
    uint32_t ind_cursor = 0;

    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string token;
        if (!(iss >> token)) continue;
        if (token == "mtllib") {
            std::string f;
            while (iss >> f) mtllibs.push_back(f);
        } else if (token == "usemtl") {
            if (!ranges.empty() || !cur_mtl.empty()) {
                uint32_t start = ranges.empty() ? 0u : ranges.back().offset + ranges.back().cnt;
                uint32_t cnt = ind_cursor - start;
                if (!ranges.empty())
                    ranges.back().cnt = cnt;
                else if (cnt)
                    ranges.push_back(Range{0, cnt, cur_mtl});
            }
            iss >> cur_mtl;
            ranges.push_back(Range{ind_cursor, 0, cur_mtl});
        } else if (token == "f") {
            int k = 0;
            std::string v;
            while (iss >> v) ++k;
            if (k >= 3) ind_cursor += uint32_t((k - 2) * 3);
        }
    }
    if (!ranges.empty())
        ranges.back().cnt = ind_cursor - ranges.back().offset;
    else if (ind_cursor)
        ranges.push_back(Range{0, ind_cursor, ""});

    std::unordered_map<std::string, MtlEntry> maps;
    for (const auto& f : mtllibs) {
        auto map = load_mtl_maps(scene_dir / f);
        maps.insert(map.begin(), map.end());
    }

    struct DrawCall {
        uint32_t ind_offset = 0, ind_cnt = 0;
        GLuint kd = 0, ka = 0, ks = 0, bump = 0, d = 0;
        glm::vec3 ka_coef = glm::vec3(1.f);
        glm::vec3 ks_coef = glm::vec3(0.f);
        float ns = 32.f;
        float d_coef = 1.f;
        float reflectivity = 0.f;
    };

    std::vector<DrawCall> drawcalls;
    drawcalls.reserve(ranges.size());
    GLuint white_fallback_srgb = 0, white_fallback_linear = 0, black_fallback = 0, flat_height_fallback = 0;
    glGenTextures(1, &white_fallback_srgb);
    glBindTexture(GL_TEXTURE_2D, white_fallback_srgb);
    unsigned char px[4] = {255, 255, 255, 255};
    glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB8_ALPHA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glGenTextures(1, &white_fallback_linear);
    glBindTexture(GL_TEXTURE_2D, white_fallback_linear);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glGenTextures(1, &black_fallback);
    glBindTexture(GL_TEXTURE_2D, black_fallback);
    unsigned char bx[4] = {0, 0, 0, 255};
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, bx);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glGenTextures(1, &flat_height_fallback);
    glBindTexture(GL_TEXTURE_2D, flat_height_fallback);
    unsigned char hx[4] = {128, 128, 128, 255};
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, hx);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    std::unordered_map<std::string, GLuint> texture_cache;
    for (const auto& r : ranges) {
        DrawCall call{};
        call.kd = white_fallback_srgb;
        call.ka = white_fallback_linear;
        call.ks = black_fallback;
        call.bump = flat_height_fallback;
        call.d = white_fallback_linear;
        call.ind_offset = r.offset;
        call.ind_cnt = r.cnt;
        auto it = maps.find(r.mtl);
        if (it != maps.end()) {
            std::filesystem::path entry_path{};
            auto try_load = [&](const std::filesystem::path& raw, GLuint& out_texture, const bool& srgb) {
                if (raw.empty()) return;
                std::filesystem::path cand = raw;
                if (!(cand.is_absolute() && std::filesystem::exists(cand))) {
                    std::filesystem::path other_cand = scene_dir / raw;
                    if (std::filesystem::exists(other_cand))
                        cand = other_cand;
                    else
                        return;
                }
                std::error_code ec;
                std::string key = std::filesystem::absolute(cand, ec).lexically_normal().string();
                key += srgb ? "|srgb" : "|linear";
                if (auto it_tex = texture_cache.find(key); it_tex != texture_cache.end()) {
                    out_texture = it_tex->second;
                    return;
                }
                int w = 0, h = 0, n = 0;
                unsigned char* data = stbi_load(cand.string().c_str(), &w, &h, &n, 4);
                if (!data) return;

                glGenTextures(1, &out_texture);
                glBindTexture(GL_TEXTURE_2D, out_texture);
                glTexImage2D(GL_TEXTURE_2D, 0, srgb ? GL_SRGB8_ALPHA8 : GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                             data);
                glGenerateMipmap(GL_TEXTURE_2D);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
                stbi_image_free(data);
                texture_cache[key] = out_texture;
            };
            try_load(it->second.map_kd, call.kd, true);
            try_load(it->second.map_ka, call.ka, false);
            try_load(it->second.map_ks, call.ks, false);
            try_load(it->second.map_bump, call.bump, false);
            try_load(it->second.map_d, call.d, false);
            call.ka_coef = it->second.ka_coef;
            call.ks_coef = it->second.ks_coef;
            call.ns = it->second.ns;
            call.d_coef = it->second.d_coef;
        }
        drawcalls.push_back(call);
    }
    if (drawcalls.empty()) {
        DrawCall call{};
        call.ind_offset = 0;
        call.ind_cnt = scene.indices.size();
        call.kd = white_fallback_srgb;
        call.ka = white_fallback_linear;
        call.ks = black_fallback;
        call.bump = flat_height_fallback;
        call.d = white_fallback_linear;
        drawcalls.push_back(call);
    }

    DrawCall bunny_call{};
    bunny_call.ind_offset = 0;
    bunny_call.ind_cnt = static_cast<uint32_t>(bunny.indices.size());
    bunny_call.kd = white_fallback_srgb;
    bunny_call.ka = white_fallback_linear;
    bunny_call.ks = black_fallback;
    bunny_call.bump = flat_height_fallback;
    bunny_call.d = white_fallback_linear;
    bunny_call.d_coef = 1.f;
    bunny_call.reflectivity = 0.8f;

    auto last_frame_start = std::chrono::high_resolution_clock::now();
    SDL_SetRelativeMouseMode(SDL_TRUE);
    std::map<SDL_Keycode, bool> button_down;
    float time = 0.f;
    bool paused = false;

    glm::vec3 cam_pos(0.f, -0.15f, -0.05f);
    glm::vec3 init_forward = glm::normalize(-cam_pos);
    float yaw = std::atan2(init_forward.x, init_forward.z);
    float pitch = std::asin(init_forward.y);
    bool running = true;
    while (running) {
        for (SDL_Event event; SDL_PollEvent(&event);) switch (event.type) {
                case SDL_QUIT:
                    running = false;
                    break;
                case SDL_WINDOWEVENT:
                    switch (event.window.event) {
                        case SDL_WINDOWEVENT_RESIZED:
                            width = event.window.data1;
                            height = event.window.data2;
                            glViewport(0, 0, width, height);
                            break;
                    }
                    break;
                case SDL_KEYDOWN:
                    button_down[event.key.keysym.sym] = true;
                    if (event.key.keysym.sym == SDLK_SPACE && event.key.repeat == 0) paused = !paused;
                    break;
                case SDL_KEYUP:
                    button_down[event.key.keysym.sym] = false;
                    break;
                case SDL_MOUSEMOTION:
                    const float sens = 0.0025f;
                    yaw -= sens * event.motion.xrel;
                    pitch -= sens * event.motion.yrel;
                    const float lim = glm::radians(89.0f);
                    pitch = std::clamp(pitch, -lim, lim);
                    break;
            }

        if (!running) break;

        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration_cast<std::chrono::duration<float>>(now - last_frame_start).count();
        last_frame_start = now;
        if (!paused) time += dt;

        glm::vec3 forward = glm::normalize(
            glm::vec3(std::cos(pitch) * std::sin(yaw), std::sin(pitch), std::cos(pitch) * std::cos(yaw)));
        glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0.f, 1.f, 0.f)));
        glm::vec3 up = glm::normalize(glm::cross(right, forward));

        float speed = button_down[SDLK_LSHIFT] ? 0.6f : 0.17f;
        float spd = speed * dt;
        if (button_down[SDLK_w]) cam_pos += forward * spd;
        if (button_down[SDLK_s]) cam_pos -= forward * spd;
        if (button_down[SDLK_d]) cam_pos += right * spd;
        if (button_down[SDLK_a]) cam_pos -= right * spd;
        if (button_down[SDLK_z]) cam_pos += up * spd;
        if (button_down[SDLK_x]) cam_pos -= up * spd;

        glm::vec3 bunny_offset(0.16f * std::sin(time * 0.7f), 0.035f * std::sin(time * 1.8f),
                               0.16f * std::cos(time * 0.5f));
        glm::vec3 bunny_position = bunny_base_position + bunny_offset;
        bunny_position.y = std::max(bunny_position.y, bunny_base_position.y);
        glm::mat4 bunny_model = build_bunny_model(bunny_position);
        glm::mat3 bunny_normal_matrix = glm::transpose(glm::inverse(glm::mat3(bunny_model)));
        glm::vec3 bunny_world_center = glm::vec3(bunny_model * glm::vec4(bunny_center, 1.f));
        glm::vec3 point_position =
            glm::vec3(bunny_model * glm::vec4(bunny_center.x, bunny_bb_max.y, bunny_center.z, 1.f));

        const float near = 0.01f;
        const float far = 10.f;
        glm::mat4 light_vp = compute_light_matrices(sun_dir);

        // shadow pass
        glBindFramebuffer(GL_FRAMEBUFFER, shadow_fbo);
        glViewport(0, 0, shadow_size, shadow_size);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_FRONT);
        glClear(GL_DEPTH_BUFFER_BIT);
        glUseProgram(depth_program);
        glUniformMatrix4fv(depth_light_vp_location, 1, GL_FALSE, reinterpret_cast<float*>(&light_vp));

        glUniformMatrix4fv(depth_model_location, 1, GL_FALSE, reinterpret_cast<float*>(&scene_model));
        glBindVertexArray(scene_vao);
        glDrawElements(GL_TRIANGLES, scene.indices.size(), GL_UNSIGNED_INT, 0);

        glUniformMatrix4fv(depth_model_location, 1, GL_FALSE, reinterpret_cast<float*>(&bunny_model));
        glBindVertexArray(bunny_vao);
        glDrawElements(GL_TRIANGLES, bunny.indices.size(), GL_UNSIGNED_INT, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glCullFace(GL_BACK);

        glm::mat4 capture_projection = glm::perspective(glm::pi<float>() / 2.f, 1.f, near, far);
        std::array<glm::mat4, 6> capture_views = {
            glm::lookAt(bunny_world_center, bunny_world_center + glm::vec3(1.f, 0.f, 0.f), glm::vec3(0.f, -1.f, 0.f)),
            glm::lookAt(bunny_world_center, bunny_world_center + glm::vec3(-1.f, 0.f, 0.f), glm::vec3(0.f, -1.f, 0.f)),
            glm::lookAt(bunny_world_center, bunny_world_center + glm::vec3(0.f, 1.f, 0.f), glm::vec3(0.f, 0.f, 1.f)),
            glm::lookAt(bunny_world_center, bunny_world_center + glm::vec3(0.f, -1.f, 0.f), glm::vec3(0.f, 0.f, -1.f)),
            glm::lookAt(bunny_world_center, bunny_world_center + glm::vec3(0.f, 0.f, 1.f), glm::vec3(0.f, -1.f, 0.f)),
            glm::lookAt(bunny_world_center, bunny_world_center + glm::vec3(0.f, 0.f, -1.f), glm::vec3(0.f, -1.f, 0.f))};

        glBindFramebuffer(GL_FRAMEBUFFER, env_fbo);
        glViewport(0, 0, cubemap_size, cubemap_size);
        glDisable(GL_FRAMEBUFFER_SRGB);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        glUseProgram(program);
        glUniform3fv(camera_pos_location, 1, reinterpret_cast<float*>(&bunny_world_center));
        glUniform3fv(sun_dir_location, 1, reinterpret_cast<float*>(&sun_dir));
        glUniform3fv(sun_color_location, 1, reinterpret_cast<float*>(&sun_color));
        glUniform3fv(point_position_location, 1, reinterpret_cast<float*>(&point_position));
        glUniform3fv(point_color_location, 1, reinterpret_cast<float*>(&point_color));
        glUniformMatrix3fv(normal_matrix_location, 1, GL_FALSE, reinterpret_cast<float*>(&scene_normal_matrix));
        glUniformMatrix4fv(model_location, 1, GL_FALSE, reinterpret_cast<float*>(&scene_model));
        glUniformMatrix4fv(light_vp_location, 1, GL_FALSE, reinterpret_cast<float*>(&light_vp));
        glActiveTexture(GL_TEXTURE6);
        glBindTexture(GL_TEXTURE_CUBE_MAP, env_cubemap);
        glUniform1i(apply_tonemap_location, 0);
        glActiveTexture(GL_TEXTURE7);
        glBindTexture(GL_TEXTURE_2D, shadow_map);
        glUniform1f(shadow_bias_location, 0.0015f);
        glUniform1i(enable_fog_location, 0);
        glUniform1i(enable_volume_location, 0);

        for (int i = 0; i < 6; ++i) {
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                                   env_cubemap, 0);
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            glUniformMatrix4fv(view_location, 1, GL_FALSE, reinterpret_cast<float*>(&capture_views[i]));
            glUniformMatrix4fv(projection_location, 1, GL_FALSE, reinterpret_cast<float*>(&capture_projection));

            glBindVertexArray(scene_vao);
            for (auto& call : drawcalls) {
                glActiveTexture(GL_TEXTURE1);
                glBindTexture(GL_TEXTURE_2D, call.kd);
                glActiveTexture(GL_TEXTURE2);
                glBindTexture(GL_TEXTURE_2D, call.ka);
                glActiveTexture(GL_TEXTURE3);
                glBindTexture(GL_TEXTURE_2D, call.ks);
                glActiveTexture(GL_TEXTURE4);
                glBindTexture(GL_TEXTURE_2D, call.bump);
                glActiveTexture(GL_TEXTURE5);
                glBindTexture(GL_TEXTURE_2D, call.d);

                glUniform1f(d_coef_location, call.d_coef);
                glUniform3fv(ka_coef_location, 1, reinterpret_cast<float*>(&call.ka_coef));
                glUniform3fv(ks_coef_location, 1, reinterpret_cast<float*>(&call.ks_coef));
                glUniform1f(ns_location, call.ns);
                glUniform1f(reflectivity_location, 0.0f);

                glDrawElements(GL_TRIANGLES, call.ind_cnt, GL_UNSIGNED_INT,
                               (void*)(size_t(call.ind_offset) * sizeof(uint32_t)));
            }
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glEnable(GL_FRAMEBUFFER_SRGB);
        glViewport(0, 0, width, height);
        glBindTexture(GL_TEXTURE_CUBE_MAP, env_cubemap);
        glGenerateMipmap(GL_TEXTURE_CUBE_MAP);

        glm::mat4 view = glm::lookAt(cam_pos, cam_pos + forward, glm::vec3(0.f, 1.f, 0.f));
        glm::mat4 projection = glm::perspective(glm::pi<float>() / 2.f, float(width) / float(height), near, far);

        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glViewport(0, 0, width, height);

        glClearColor(0.8f, 0.8f, 0.9f, 0.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);

        glUseProgram(program);
        glUniform3fv(camera_pos_location, 1, reinterpret_cast<float*>(&cam_pos));
        glUniform3fv(sun_dir_location, 1, reinterpret_cast<float*>(&sun_dir));
        glUniform3fv(sun_color_location, 1, reinterpret_cast<float*>(&sun_color));
        glUniform3fv(point_position_location, 1, reinterpret_cast<float*>(&point_position));
        glUniform3fv(point_color_location, 1, reinterpret_cast<float*>(&point_color));
        glUniformMatrix3fv(normal_matrix_location, 1, GL_FALSE, reinterpret_cast<float*>(&scene_normal_matrix));
        glUniformMatrix4fv(model_location, 1, GL_FALSE, reinterpret_cast<float*>(&scene_model));
        glUniformMatrix4fv(view_location, 1, GL_FALSE, reinterpret_cast<float*>(&view));
        glUniformMatrix4fv(projection_location, 1, GL_FALSE, reinterpret_cast<float*>(&projection));
        glUniform1i(apply_tonemap_location, 1);
        glUniformMatrix4fv(light_vp_location, 1, GL_FALSE, reinterpret_cast<float*>(&light_vp));
        glActiveTexture(GL_TEXTURE7);
        glBindTexture(GL_TEXTURE_2D, shadow_map);
        glUniform1f(shadow_bias_location, 0.0015f);
        glUniform1i(enable_fog_location, 1);
        glUniform1i(enable_volume_location, 1);

        glActiveTexture(GL_TEXTURE6);
        glBindTexture(GL_TEXTURE_CUBE_MAP, env_cubemap);
        glBindVertexArray(scene_vao);
        for (auto& call : drawcalls) {
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, call.kd);
            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D, call.ka);
            glActiveTexture(GL_TEXTURE3);
            glBindTexture(GL_TEXTURE_2D, call.ks);
            glActiveTexture(GL_TEXTURE4);
            glBindTexture(GL_TEXTURE_2D, call.bump);
            glActiveTexture(GL_TEXTURE5);
            glBindTexture(GL_TEXTURE_2D, call.d);

            glUniform1f(d_coef_location, call.d_coef);
            glUniform3fv(ka_coef_location, 1, reinterpret_cast<float*>(&call.ka_coef));
            glUniform3fv(ks_coef_location, 1, reinterpret_cast<float*>(&call.ks_coef));
            glUniform1f(ns_location, call.ns);
            glUniform1f(reflectivity_location, call.reflectivity);

            glDrawElements(GL_TRIANGLES, call.ind_cnt, GL_UNSIGNED_INT,
                           (void*)(size_t(call.ind_offset) * sizeof(uint32_t)));
        }

        glUniformMatrix3fv(normal_matrix_location, 1, GL_FALSE, reinterpret_cast<float*>(&bunny_normal_matrix));
        glUniformMatrix4fv(model_location, 1, GL_FALSE, reinterpret_cast<float*>(&bunny_model));
        glBindVertexArray(bunny_vao);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, bunny_call.kd);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, bunny_call.ka);
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, bunny_call.ks);
        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_2D, bunny_call.bump);
        glActiveTexture(GL_TEXTURE5);
        glBindTexture(GL_TEXTURE_2D, bunny_call.d);
        glActiveTexture(GL_TEXTURE6);
        glBindTexture(GL_TEXTURE_CUBE_MAP, env_cubemap);

        glUniform1f(d_coef_location, bunny_call.d_coef);
        glUniform3fv(ka_coef_location, 1, reinterpret_cast<float*>(&bunny_call.ka_coef));
        glUniform3fv(ks_coef_location, 1, reinterpret_cast<float*>(&bunny_call.ks_coef));
        glUniform1f(ns_location, bunny_call.ns);
        glUniform1f(reflectivity_location, bunny_call.reflectivity);

        glDrawElements(GL_TRIANGLES, bunny_call.ind_cnt, GL_UNSIGNED_INT,
                       (void*)(size_t(bunny_call.ind_offset) * sizeof(uint32_t)));

        SDL_GL_SwapWindow(window);
    }

    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
} catch (std::exception const& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
