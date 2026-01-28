# Real-time 3D Scene Viewer

Interactive 3D scene viewer written in **C++20** and **OpenGL** (SDL2 + GLEW).  
Designed as a small, self-contained rendering playground: load a textured OBJ scene, fly around with the camera, and explore common real-time rendering effects.

## Features

- Textured OBJ scene loading (materials + textures)
- Real-time lighting (sun + point light)
- Shadows
- Reflections on a moving object
- Fog and volumetric lighting/shadowing
- HDR-ish color pipeline (tonemapping + gamma-correct rendering)

## Build

**Dependencies**

- CMake 3.x+
- A C++20 compiler
- SDL2
- GLEW
- OpenGL (3.3+ recommended)

**Commands**

```bash
cmake -S . -B build
cmake --build build -j
```

## Run

The program expects a **directory** with an `.obj` file (and its `.mtl` + textures).

```bash
./build/opengl-scene-viewer /path/to/scene_dir
```

If `/path/to/scene_dir/scene_dir.obj` does not exist, it will try to pick **any** `.obj` inside the directory.

## Controls

- Mouse: look around
- `W/A/S/D`: move
- `Z/X`: up/down
- `Left Shift`: faster movement
- `Space`: pause animation
- Close window: exit
