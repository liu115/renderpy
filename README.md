# Simple python rasterizer implemented by OpenGL and C++

Key features:
- Support both PINHOLE and camera models with distortion parameters like (OPENCV, OPENCV_FISHEYE)
- Headless OpenGL rendering
- Output numpy.ndarray directly

## Requirements
- cmake >= 3.16


## Install
```bash
git clone --recursive https://github.com/liu115/renderpy    # clone with submodules
conda create -n renderpy python=3.9
conda activate renderpy
pip install .
```


## Usage

```python
import renderpy

render_engine = renderpy.Render()
render_engine.setupMesh(MESH_PATH)

camera_model = "OPENCV_FISHEYE"
render_engine.setupCamera(
    height, width,
    fx, fy, cx, cy,
    camera_model,
    params,      # Distortion parameters np.array([k1, k2, k3, k4]) or np.array([k1, k2, p1, p2])
)
near = 0.05
far = 20.0
rgb, depth, vert_indices = render_engine.renderAll(world_to_camera, near, far)
```
### Explanation

`render_engine.renderAll` render RGB, depth, and corresponding vertex indices of the mesh

* rgb: np.ndarray (height, width, 3) uint8
* depth: np.ndarray (height, width) float32. Pixels with value 0.0 means no depth
* vertex_indices: np.ndarray (height, width) int32. Each pixel represent the three vertex indices in the mesh during rendering. Useful for 2D-3D back projection. Pixels with value -1 means no vertex

## Common issues

### Missing Dependencies
If some dependencies are missing during build, you can install them by:
```bash
# Build tools
apt-get install build-essential cmake git
# OpenGL related
apt-get install libgl1-mesa-dev libglu1-mesa-dev libxrandr-dev libxext-dev
# OpenCV
apt-get install libopencv-dev
# If missing gflags
apt-get install libgflags-dev
apt-get install libboost-all-dev    # for mLib
```

### Version `GLIBCXX_3.4.30' not found
If you encounter missing libstdc++.so.6 error in conda environment, try this:

Add this line to your .bashrc or .zshrc
```
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

### fatal error: filesystem: No such file or directory

This is because before gcc/g++ 9, it should be `#include <experimental/filesystem>` instead (see [reference](https://stackoverflow.com/questions/33149878/experimentalfilesystem-linker-error?fbclid=IwAR0LKMp-P6RQTJ-amZ8fQypdyKcK5UGHXPjfnjEjO10gqaCrFNLek490xHo)).
You can fixed it by replace all the `#include <filesystem>` with `#include <experimental/filesystem>` and add `link_libraries(stdc++fs)` in **CMakeLists.txt**


## Other notes

Currently, the `thirdparty/eigen-3.4.0/Eigen` has been modified:
*  MACRO Success is replaced with SuccessfulComputation to avoid conflict with X.h (X11)
* See also https://eigen.tuxfamily.org/bz/show_bug.cgi?id=253
