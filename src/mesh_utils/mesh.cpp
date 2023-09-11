#include <mLibCore.h>
#include "mesh_utils/mesh.h"

using namespace std;


void drawLineOnMesh(ml::MeshDataf& mesh, ml::vec3f& start, ml::vec3f& end) {
    ml::vec4f color(1, 0, 1, 0);    // purple
    float thinkness = 0.01;
    ml::TriMeshf line = ml::Shapesf::line(start, end, color, thinkness);
    mesh.merge(line.computeMeshData());
}


ml::vec3f elementwiseMult(ml::vec3f a, ml::vec3f b) {
    return ml::vec3f(a[0] * b[0], a[1] * b[1], a[2] * b[2]);
}


void drawPoseOnMesh(ml::MeshDataf& mesh, ml::mat4f& camera_to_world, float radius, float length) {
    ml::vec4f origin_color(0, 0, 0, 1);
    ml::vec4f x_color(1, 0, 0, 1);
    ml::vec4f y_color(0, 1, 0, 1);
    ml::vec4f z_color(0, 0, 1, 1);

    ml::vec3f origin = camera_to_world.getTranslation();
    // Draw the camera origin circle
    float r = radius;
    ml::TriMeshf sphere = ml::Shapesf::sphere(r, origin, 10, 10, origin_color);
    mesh.merge(sphere.computeMeshData());

    // Draw the xyz axis
    float thinkness = radius / 2;

    ml::vec3f x_axis(1, 0, 0);
    x_axis = camera_to_world.getRotation() * x_axis;
    x_axis = x_axis / x_axis.length() * length;
    x_axis += origin;
    ml::TriMeshf line = ml::Shapesf::line(origin, x_axis, x_color, thinkness);
    mesh.merge(line.computeMeshData());

    ml::vec3f y_axis(0, 1, 0);
    y_axis = camera_to_world.getRotation() * y_axis;
    y_axis = y_axis / y_axis.length() * length;
    y_axis += origin;
    line = ml::Shapesf::line(origin, y_axis, y_color, thinkness);
    mesh.merge(line.computeMeshData());

    ml::vec3f z_axis(0, 0, 1);
    z_axis = camera_to_world.getRotation() * z_axis;
    z_axis = z_axis / z_axis.length() * length;
    z_axis += origin;
    line = ml::Shapesf::line(origin, z_axis, z_color, thinkness);
    mesh.merge(line.computeMeshData());
}
