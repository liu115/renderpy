#pragma once
#include <mLibCore.h>
#include "opengl/opengl.h"

namespace opt {

class MultiScaleMesh {
public:
    MultiScaleMesh() {}

    void addMesh(ml::TriMeshf& mesh) {
        meshes_.push_back(std::shared_ptr<ml::TriMeshf>(new ml::TriMeshf(mesh)));
    }

    int levels() const {
        return meshes_.size();
    }

    ml::TriMeshf& getMesh(int level) const {
        assert (level < meshes_.size());
        return *meshes_[level];
    }

    int getVertexCount(int level) const {
        assert (level < meshes_.size());
        return meshes_[level]->getVertices().size();
    }

    void copyVertexToVector(int level, std::vector<ml::vec3f>& vertex_xyzs, std::vector<ml::vec3f>& vertex_colors) {
        assert (level < meshes_.size());
        auto& vertices = getMesh(level).getVertices();
        vertex_xyzs.resize(vertices.size());
        vertex_colors.resize(vertices.size());
        for (int vertex_id = 0; vertex_id < vertices.size(); vertex_id++) {
            vertex_xyzs[vertex_id] = vertices[vertex_id].position;
            // Ignore alpha channel for the vertex color
            vertex_colors[vertex_id].x = vertices[vertex_id].color.x;
            vertex_colors[vertex_id].y = vertices[vertex_id].color.y;
            vertex_colors[vertex_id].z = vertices[vertex_id].color.z;
        }
    }

private:
    std::vector<std::shared_ptr<ml::TriMeshf>> meshes_;
};
} // namespace opt
