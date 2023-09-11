# Simple python rasterizer implemented by OpenGL and C++

Key features:
- Support both PINHOLE and camera models with distortion parameters like (OPENCV, OPENCV_FISHEYE)
- Headless OpenGL rendering
- Output numpy.ndarray directly

## Requirements
- cmake >= 3.22


## Install
```
conda create -n renderpy python=3.9
conda activate renderpy
pip install .
```


### Common issues

If you encounter missing libstdc++.so.6 error in conda environment, try this:

Add this line to your .bashrc or .zshrc
```
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
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
    params      # Distortion parameters [k1, k2, k3, k4] or [k1, k2, p1, p2]
)
near = 0.05
far = 20.0
rgb, depth, vert_indices = render_engine.renderAll(world_to_camera, near, far)
```
