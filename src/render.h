#pragma once
#include <mLibCore.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "opengl/opengl.h"
#include "opengl/opengl_semantic.h"
#include "opt/camera.h"

namespace py = pybind11;


class RenderEngine {
public:
    RenderEngine() {
        egl_ctx = new opengl::ContextManager();
        if (!opengl::init()) {
            throw std::runtime_error("Failed to initialize OpenGL");
        }   // Try initializing opengl
    }
    ~RenderEngine() {
        // Free all resources
        delete meshdata;
        delete trimesh;
        delete gl_mesh;
        delete gl_renderer;
        delete camera;
        delete egl_ctx;
    }
    void loadMesh(std::string mesh_path) {
        meshdata = new ml::MeshDataf();
        ml::MeshIOf::loadFromPLY(mesh_path, *meshdata);
        std::cout << "Loaded mesh:" << *meshdata << std::endl;
    }

    // Setup mesh
    void copyMesh() {
        gl_mesh = new opengl::GLSemanticMesh();
        trimesh = new ml::TriMeshf(*meshdata);
        auto& verts = trimesh->getVertices();
        for (int i = 0; i < verts.size(); i++) {
            gl_mesh->addVertex(verts[i].position.x, verts[i].position.y, verts[i].position.z);
            gl_mesh->addVertexColor(verts[i].color.r, verts[i].color.g, verts[i].color.b, verts[i].color.a);
            gl_mesh->addVertexSemantic(0);
        }
        for (auto& f: trimesh->getIndices()) {
            gl_mesh->addFaceIndice(f.x, f.y, f.z);
        }
        gl_mesh->setupGLBuffer();
        std::cout << "Copy mesh to GPU: " << verts.size() << " vertices, " << trimesh->getIndices().size() << " faces" << std::endl;
    }

    void setupMesh(std::string mesh_path) {
        loadMesh(mesh_path);
        copyMesh();
    }

    void setupCamera(
        int height,
        int width,
        float fx,
        float fy,
        float cx,
        float cy,
        std::string camera_type,
        py::array_t<float>& distortion_params
        // float distortion_params
    ) {
        gl_renderer = new opengl::GLSemanticRenderer(height, width);
        camera = new opt::CameraWrapper(
            // height, width, fx, fy, cx, cy, camera_type, distortion_params
            height, width, fx, fy, cx, cy, camera_type, distortion_params.data()
        );
    }

    // Setup camera parameter
    // typedef std::tuple<py::array_t<float> rgb, py::array_t<float> depth, py::array_t<float> semantics, py::array_t<float> vertex_indices> RenderOut;
    typedef std::tuple<py::array_t<uint8_t>, py::array_t<float>, py::array_t<int>> RenderOut;
    RenderOut renderAll(py::array_t<float>& world_to_camera, float near, float far) {
        assert (world_to_camera.ndim() == 2);
        assert (world_to_camera.shape(0) == 4);
        assert (world_to_camera.shape(1) == 4);
        cv::Mat render_rgb, render_depth, render_semantic, vertex_indices, triangle_weights;
        ml::mat4f intrinsic = camera->getIntrinsicMatrix();
        auto params = camera->getDistortionParams();
        gl_renderer->render(
            *gl_mesh, camera->type(),
            intrinsic.getData(),
            params,
            world_to_camera.data(),
            near, far,
            render_rgb,
            render_depth, render_semantic,
            vertex_indices, triangle_weights,
            camera->radius_cutoff_squared()
        );

        py::array_t<uint8_t> rgb_array(
            py::buffer_info(
                render_rgb.data,
                sizeof(uint8_t), //itemsize
                py::format_descriptor<uint8_t>::format(),
                3, // ndim
                std::vector<size_t> {render_rgb.rows, render_rgb.cols , 3}, // shape
                std::vector<size_t> {sizeof(uint8_t) * render_rgb.cols * 3, sizeof(uint8_t) * 3, sizeof(uint8_t)} // strides
                // std::vector<size_t> {cols * sizeof(uint8_t), sizeof(uint8_t), 3} // strides
            )
        );

        py::array_t<float> depth_array(
            py::buffer_info(
                render_depth.data,
                sizeof(float), //itemsize
                py::format_descriptor<float>::format(),
                2, // ndim
                std::vector<size_t> {render_depth.rows, render_depth.cols}, // shape
                std::vector<size_t> {sizeof(float) * render_depth.cols, sizeof(float)} // strides
            )
        );

        py::array_t<int> vert_indices_array(
            py::buffer_info(
                vertex_indices.data,
                sizeof(int), //itemsize
                py::format_descriptor<int>::format(),
                3, // ndim
                std::vector<size_t>{vertex_indices.rows, vertex_indices.cols, 3}, // shape
                std::vector<size_t>{sizeof(int) * vertex_indices.cols * 3, sizeof(int) * 3, sizeof(int)} // strides
            )
        );
        return std::make_tuple(
            std::move(rgb_array),
            std::move(depth_array),
            std::move(vert_indices_array)
        );
    }

    // TODO: Render Semantic

private:
    opengl::ContextManager* egl_ctx = nullptr;
    ml::MeshDataf* meshdata = nullptr;
    ml::TriMeshf* trimesh = nullptr;
    opengl::GLSemanticMesh* gl_mesh = nullptr;
    opengl::GLSemanticRenderer* gl_renderer = nullptr;
    opt::CameraWrapper* camera = nullptr;

    void writeMat3(cv::Mat& mat, std::string path) {
        // Write 3 channel matrix to file. Binary format. Pixel by pixel.
        assert (mat.channels() == 3);
        // assert (mat.type() == CV_32FC3);
        std::ofstream fout(path, std::ios::binary);
        // Write height and width and channel
        fout.write((char*)&mat.rows, sizeof(int));
        fout.write((char*)&mat.cols, sizeof(int));
        const int channel = 3;
        fout.write((char*)&channel, sizeof(int));
        fout.write((char*)mat.data, mat.total() * mat.elemSize());
        fout.close();
    }
};
