#pragma once

void drawLineOnMesh(ml::MeshDataf& mesh, ml::vec3f& start, ml::vec3f& end);

void drawPoseOnMesh(ml::MeshDataf& mesh, ml::mat4f& camera_to_world,  float radius = 0.5, float length = 0.5);
