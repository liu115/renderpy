#include <pybind11/pybind11.h>
// #include "class.h"
#include "render.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


namespace py = pybind11;

PYBIND11_MODULE(renderpy, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: renderpy

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";\
    py::class_<RenderEngine>(m, "Render")
        .def(py::init<>())
        .def("setupMesh", &RenderEngine::setupMesh)
        .def("setupCamera", &RenderEngine::setupCamera)
        .def("renderAll", &RenderEngine::renderAll);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
