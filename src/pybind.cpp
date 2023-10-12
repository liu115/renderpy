#include <pybind11/pybind11.h>
// #include "class.h"
#include "render.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


namespace py = pybind11;

PYBIND11_MODULE(renderpy, m) {
    py::class_<RenderEngine>(m, "Render")
        .def(py::init<>())
        .def("loadMesh",
             &RenderEngine::loadMesh,
             "Load PLY mesh file"
            )
        .def("copyMesh",
             &RenderEngine::copyMesh,
             "Copy mesh data to GPU (OpenGL buffer)")
        .def("setupMesh",
             &RenderEngine::setupMesh,
             "Setup mesh data for rendering from PLY file. This function will call loadMesh and copyMesh"
            )
        .def("setupCamera",
             &RenderEngine::setupCamera,
             "Setup camera parameters for rendering"
            )
        .def("renderAll",
             &RenderEngine::renderAll,
             R"(
                Function that render RGB, depth, and corresponding vertex indices
                :param world2cam: 4x4 matrix, world to camera transformation
                :param near: float number, near plane
                :param far: float number, far plane

                :return rgb: np.ndarray (height, width, 3) uint8
                :return depth: np.ndarray (height, width) float32. Pixels with value 0.0 means no depth
                :return vertex_indices: np.ndarray (height, width) int32. Each pixel represent the three vertex indices in the mesh during rendering.
                    Useful for 2D-3D back projection. Pixels with value -1 means no vertex
              )"
            );

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
