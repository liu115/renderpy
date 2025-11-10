#pragma once
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "opengl/egl.h"
#include "opengl/opengl.h"


namespace opengl {
class GLSemanticMesh{
public:
    GLSemanticMesh() {}
    ~GLSemanticMesh() {
        glDeleteVertexArrays(1, &vertex_array_);
        glDeleteBuffers(1, &vertex_buffer_);
        glDeleteBuffers(1, &semantic_buffer_);
        glDeleteBuffers(1, &color_buffer_);
        glDeleteBuffers(1, &index_buffer_);
        glDeleteBuffers(1, &element_buffer_);
    }

    void setupGLBuffer() {
        // Run this after setting up all the vertices and colors data
        glGenVertexArrays(1, &vertex_array_);
    	glBindVertexArray(vertex_array_);

        glGenBuffers(1, &vertex_buffer_);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), &vertices[0], GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

        glGenBuffers(1, &color_buffer_);
        glBindBuffer(GL_ARRAY_BUFFER, color_buffer_);
        glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(glm::vec4), &colors[0], GL_STATIC_DRAW);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);

    	glGenBuffers(1, &semantic_buffer_);
        glBindBuffer(GL_ARRAY_BUFFER, semantic_buffer_);
        glBufferData(GL_ARRAY_BUFFER, semantics.size() * sizeof(int), &semantics[0], GL_STATIC_DRAW);
        glEnableVertexAttribArray(2);
        glVertexAttribIPointer(2, 1, GL_INT, 0, (void*)0);

        std::vector<int> vertex_indices(vertices.size());
        std::iota(std::begin(vertex_indices), std::end(vertex_indices), 0);
    	glGenBuffers(1, &index_buffer_);
        glBindBuffer(GL_ARRAY_BUFFER, index_buffer_);
        glBufferData(GL_ARRAY_BUFFER, vertex_indices.size() * sizeof(int), &vertex_indices[0], GL_STATIC_DRAW);
        glEnableVertexAttribArray(3);
        glVertexAttribIPointer(3, 1, GL_INT, 0, (void*)0);

    	glGenBuffers(1, &element_buffer_);
	    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer_);
	    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

        glBindVertexArray(0);
    }

    void addVertex(float x, float y, float z) {
        vertices.push_back(glm::vec3(x, y, z));
    }

    void addFaceIndice(int x, int y, int z) {
        indices.push_back(x);
        indices.push_back(y);
        indices.push_back(z);
    }

    void addVertexSemantic(int s) {
        semantics.push_back(s);
    }

    void addVertexColor(float r, float g, float b, float a) {
        colors.push_back(glm::vec4(r, g, b, a));
    }

    void draw() {
        // std::cerr << "Vertex size: " << vertices.size() << std::endl;
        // std::cerr << "Index size: " << indices.size() << std::endl;
        glBindVertexArray(vertex_array_);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer_);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }

private:
    glm::mat4 model_to_world_;
    std::vector<glm::vec3> vertices;
    std::vector<unsigned int> indices;
    std::vector<int> semantics;
    std::vector<glm::vec4> colors;
    unsigned int vertex_array_;
    unsigned int vertex_buffer_;
    unsigned int semantic_buffer_;
    unsigned int color_buffer_;
    unsigned int index_buffer_;
    unsigned int element_buffer_;
};


class GLSemanticProgram {
public:
    GLSemanticProgram() {}
    GLSemanticProgram(CameraModelType type) : camera_model(type) {
        // std::cout << "Compile vertex shader:" << std::endl << vertex_shader_program << std::endl;
        vertex_shader_ = glCreateShader(GL_VERTEX_SHADER);
        std::string vertex_shader_program;
        if (type == CameraModelType::CAMERA_PINHOLE) {
            vertex_shader_program = getPinholeVertexShader();
        }
        else if (type == CameraModelType::CAMERA_FISHEYE) {
            vertex_shader_program = getFisheyeVertexShader();
        }
        else if (type == CameraModelType::CAMERA_OPENCV) {
            vertex_shader_program = getOpenCVVertexShader();
        }
        else {
            throw std::runtime_error("Unknown camera model type for OpenGL render");
        }
        const char *code_ptr = vertex_shader_program.c_str();
        glShaderSource(vertex_shader_, 1, &code_ptr, NULL);
        glCompileShader(vertex_shader_);
        checkShaderCompile(vertex_shader_);

        // std::cout << "Compile fragment shader:" << std::endl << fragment_shader_program << std::endl;
        std::string geometry_shader_program_ = getGeometryShader();
        const char *geo_code_ptr = geometry_shader_program_.c_str();
        geometry_shader_ = glCreateShader(GL_GEOMETRY_SHADER);
        glShaderSource(geometry_shader_, 1, &geo_code_ptr, NULL);
        glCompileShader(geometry_shader_);
        checkShaderCompile(geometry_shader_);

        std::string fragment_shader_program_ = getFragmentShader();
        const char *frag_code_ptr = fragment_shader_program_.c_str();
        fragment_shader_ = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragment_shader_, 1, &frag_code_ptr, NULL);
        glCompileShader(fragment_shader_);
        checkShaderCompile(fragment_shader_);

        program_id = glCreateProgram();
	    glAttachShader(program_id, vertex_shader_);
	    glAttachShader(program_id, fragment_shader_);
        glAttachShader(program_id, geometry_shader_);
	    glLinkProgram(program_id);

        GLint program_linked;
        glGetProgramiv(program_id, GL_LINK_STATUS, &program_linked);
        if (program_linked != GL_TRUE)
        {
            GLsizei log_length = 0;
            GLchar message[1024];
            glGetProgramInfoLog(program_id, 1024, &log_length, message);
            std::cout << message << std::endl;
            // Write the error to a log
        }

        glDeleteShader(vertex_shader_);
        glDeleteShader(fragment_shader_);
        glDeleteShader(geometry_shader_);
    }

    ~GLSemanticProgram() {
    	glDeleteProgram(program_id);
    }

    bool checkShaderCompile(unsigned int shader) {
        int success;
        char infoLog[512];
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 512, NULL, infoLog);
            std::cout << "Error when compiling shader" << std::endl;
            std::cout << infoLog << std::endl;
            return false;
        }
        return true;
    }

    void use() {
		glUseProgram(program_id);
    }

    const std::string getGeometryShader() {
        return "#version 330 core\n"
            "layout (triangles) in;\n"
            "layout (triangle_strip, max_vertices = 3) out;\n"
            "in vec3 fragmentColor[];\n"
            "in vec3 fragmentCoord[];\n"
            "in int fragmentSemantic[];\n"
            "in int fragmentIndex[];\n"
            "flat out vec3 vertexColor[3];\n"
            "flat out int vertexSemantic[3];\n"
            "flat out int vertexIndex[3];\n"
            "out vec3 fragmentCoordOut;\n"
            "out vec3 weightCoord;\n"
            "void main() {\n"
            "    for (int i = 0; i < 3; ++i) {\n"
            "        vertexColor[i] = fragmentColor[i];\n"
            "        vertexSemantic[i] = fragmentSemantic[i];\n"
            "        vertexIndex[i] = fragmentIndex[i];\n"
            "    }\n"
            "    for (int i = 0; i < 3; ++i) {\n"
            "        weightCoord = vec3(0.0);\n"
            "        weightCoord[i] = 1.0;\n"
            "        fragmentCoordOut = fragmentCoord[i];\n"
            "        gl_Position = gl_in[i].gl_Position;\n"
            "        EmitVertex();\n"
            "    }\n"
            "    EndPrimitive();\n"
            "}\n";
    }

    const std::string getFragmentShader() {
        return "#version 330 core\n"
            "in vec3 fragmentCoordOut;\n"
            "in vec3 weightCoord;\n"
            "flat in vec3 vertexColor[3];\n"
            "flat in int vertexSemantic[3];\n"
            "flat in int vertexIndex[3];\n"
            "layout (location = 0) out vec3 color;\n"
            "layout (location = 1) out highp float depth;\n"
            "layout (location = 2) out int semantic;\n"
            "layout (location = 3) out ivec3 indices;\n"
            "layout (location = 4) out vec3 weights;\n"
            "void main() {\n"
            "    int i = (weightCoord.x > weightCoord.y && weightCoord.x > weightCoord.z) ? 0 :\n"
            "    ((weightCoord.y > weightCoord.z) ? 1 : 2);\n"
            "    color = vertexColor[i];\n"
            "    semantic = vertexSemantic[i];\n"
            "    depth = fragmentCoordOut.z;\n"
            "    indices = ivec3(vertexIndex[0], vertexIndex[1], vertexIndex[2]);\n"
            "    weights = weightCoord;\n"
            "}\n";
    }

    const std::string getFisheyeVertexShader () {
        return
            "#version 330 core\n"
            "layout(location = 0) in highp vec3 vertex;\n"
            "layout(location = 1) in vec4 vertexColor;\n"
            "layout(location = 2) in int vertexSemantic;\n"
            "layout(location = 3) in int vertexIndex;\n"
            "out vec3 fragmentColor;\n"
            "out vec3 fragmentCoord;\n"
            "out int fragmentSemantic;\n"
            "out int fragmentIndex;\n"
            "uniform float radius_cutoff_squared;\n"
            "uniform float k1;\n"
            "uniform float k2;\n"
            "uniform float k3;\n"
            "uniform float k4;\n"
            "uniform mat4 model_view_matrix;\n"
            "uniform mat4 projection_matrix;\n"
            "void main() {\n"
            "    vec4 cameraCoord = model_view_matrix * vec4(vertex, 1);\n"
            "    cameraCoord.xyz /= cameraCoord.w;\n"
            "    float nx = cameraCoord.x / cameraCoord.z;\n"
            "    float ny = cameraCoord.y / cameraCoord.z;\n"
            "    float r2 = nx * nx + ny * ny;\n"
            "    if (r2 < radius_cutoff_squared) {\n"
            "        float r = sqrt(r2);\n"
            "        if (r > 1e-6) {\n"
            "            float theta_by_r = atan(r, 1.0) / r;\n"
            "            nx = theta_by_r * nx;\n"
            "            ny = theta_by_r * ny;\n"
            "            r2 = theta_by_r * theta_by_r * r2;\n"
            "        }\n"
            "        r2 = 1.0 + r2 * (k1 + r2 * (k2 + r2 * (k3 + r2 * k4)));\n"
            "    } else {\n"
            "        r2 = 99.0;\n"
            "    }\n"
            "    cameraCoord.x = cameraCoord.z * r2 * nx;\n"
            "    cameraCoord.y = cameraCoord.z * r2 * ny;\n"
            "    cameraCoord.w = 1.0;\n"
            "    gl_Position = projection_matrix * cameraCoord;\n"
            "    fragmentColor = vertexColor.rgb;\n"
            "    fragmentCoord = cameraCoord.xyz;\n"
            "    fragmentSemantic = vertexSemantic;\n"
            "    fragmentIndex = vertexIndex;\n"
            "}\n";
    }

    const std::string getPinholeVertexShader() {
        return
            "#version 330 core\n"
            "layout(location = 0) in highp vec3 vertex;\n"
            "layout(location = 1) in vec4 vertexColor;\n"
            "layout(location = 2) in int vertexSemantic;\n"
            "layout(location = 3) in int vertexIndex;\n"
            "out vec3 fragmentColor;\n"
            "out vec3 fragmentCoord;\n"
            "out int fragmentSemantic;\n"
            "out int fragmentIndex;\n"
            "uniform mat4 model_view_matrix;\n"
            "uniform mat4 projection_matrix;\n"
            "void main() {\n"
            "    vec4 cameraCoord = model_view_matrix * vec4(vertex, 1);\n"
            "    gl_Position = projection_matrix * cameraCoord;\n"
            "    fragmentColor = vertexColor.rgb;\n"
            "    fragmentCoord = cameraCoord.xyz / cameraCoord.w;\n"
            "    fragmentSemantic = vertexSemantic;\n"
            "    fragmentIndex = vertexIndex;\n"
            "}\n";
    }

    const std::string getOpenCVVertexShader() {
        return
            "#version 330 core\n"
            "layout(location = 0) in highp vec3 vertex;\n"
            "layout(location = 1) in vec4 vertexColor;\n"
            "layout(location = 2) in int vertexSemantic;\n"
            "layout(location = 3) in int vertexIndex;\n"
            "out vec3 fragmentColor;\n"
            "out vec3 fragmentCoord;\n"
            "out int fragmentSemantic;\n"
            "out int fragmentIndex;\n"
            "uniform float radius_cutoff_squared;\n"
            "uniform float k1;\n"
            "uniform float k2;\n"
            "uniform float p1;\n"
            "uniform float p2;\n"
            "uniform mat4 model_view_matrix;\n"
            "uniform mat4 projection_matrix;\n"
            "void main() {\n"
            "    vec4 cameraCoord = model_view_matrix * vec4(vertex, 1);\n"
            "    cameraCoord.xyz /= cameraCoord.w;\n"
            "    float nx = cameraCoord.x / cameraCoord.z;\n"
            "    float ny = cameraCoord.y / cameraCoord.z;\n"
            "    float x2 = nx * nx;\n"
            "    float xy = nx * ny;\n"
            "    float y2 = ny * ny;\n"
            "    float r2 = x2 + y2;\n"
            "    if (r2 < radius_cutoff_squared) {\n"             // need clipping https://github.com/ETH3D/dataset-pipeline/blob/dc4a1069e3351bb07e26221f97ef011611af9803/src/camera/camera_base_impl.h#L413
            "        float radial = 1.0 + r2 * (k1 + r2 * k2);\n"
            "        cameraCoord.x = cameraCoord.z * (radial * nx + 2.0 * p1 * xy + p2 * (r2 + 2.0 * x2));\n"
            "        cameraCoord.y = cameraCoord.z * (radial * ny + 2.0 * p2 * xy + p1 * (r2 + 2.0 * y2));\n"
            "    } else {\n"
            "        cameraCoord.x = cameraCoord.x * 99.0;\n"
            "        cameraCoord.y = cameraCoord.y * 99.0;\n"
            "    }\n"
            "    cameraCoord.w = 1.0;\n"
            "    gl_Position = projection_matrix * cameraCoord;\n"
            "    fragmentColor = vertexColor.rgb;\n"
            "    fragmentCoord = cameraCoord.xyz;\n"
            "    fragmentSemantic = vertexSemantic;\n"
            "    fragmentIndex = vertexIndex;\n"
            "}\n";
    }

    unsigned int getProjectionMatrixID() {
	    return glGetUniformLocation(program_id, "projection_matrix");
    }

    // unsigned int getModelViewMatrixID() {
    void setProjectionMatrix(glm::mat4 mat) {
	    GLint mid = glGetUniformLocation(program_id, "projection_matrix");
        if (mid == -1) {
            std::cout << "Error when finding projection matrix" << std::endl;
            throw;
        }
        glUniformMatrix4fv(mid, 1, GL_FALSE, &mat[0][0]);
    }

    void setModelViewMatrix(glm::mat4 mat) {
	    GLint mid = glGetUniformLocation(program_id, "model_view_matrix");
        if (mid == -1) {
            std::cout << "Error when finding model-view matrix" << std::endl;
            throw;
        }
        glUniformMatrix4fv(mid, 1, GL_FALSE, &mat[0][0]);
    }

    void setDistortionParams(DistortParams params, float radius_cutoff) {
        GLint k1_id = glGetUniformLocation(program_id, "k1");
        glUniform1f(k1_id, params.k1);
        GLint k2_id = glGetUniformLocation(program_id, "k2");
        glUniform1f(k2_id, params.k2);
        if (radius_cutoff > 1e3f) radius_cutoff = 1e3f;
        GLint cutoff_id = glGetUniformLocation(program_id, "radius_cutoff_squared");
        glUniform1f(cutoff_id, radius_cutoff);
        if (camera_model == CameraModelType::CAMERA_OPENCV) {
            GLint p1_id = glGetUniformLocation(program_id, "p1");
            glUniform1f(p1_id, params.p1);
            GLint p2_id = glGetUniformLocation(program_id, "p2");
            glUniform1f(p2_id, params.p2);
        }
        if (camera_model == CameraModelType::CAMERA_FISHEYE) {
            GLint k3_id = glGetUniformLocation(program_id, "k3");
            glUniform1f(k3_id, params.k3);
            GLint k4_id = glGetUniformLocation(program_id, "k4");
            glUniform1f(k4_id, params.k4);
        }
    }

private:
    unsigned int program_id;
    unsigned int vertex_shader_;
    unsigned int fragment_shader_;
    unsigned int geometry_shader_;
    CameraModelType camera_model;
};


class GLSemanticRenderer {
public:
    GLSemanticRenderer() {}
    GLSemanticRenderer(int height, int width): height(height), width(width) {
        pinhole_shader_program_ = new GLSemanticProgram(CameraModelType::CAMERA_PINHOLE);
        opencv_shader_program_ = new GLSemanticProgram(CameraModelType::CAMERA_OPENCV);
        fisheye_shader_program_ = new GLSemanticProgram(CameraModelType::CAMERA_FISHEYE);
        glEnable(GL_DEPTH_TEST);        // Enable depth test
        glDepthFunc(GL_LESS);           // Accept fragment if it closer to the camera than the former one

        std::cout << "Setup the frame and render buffer" << std::endl;
        // Setup frame buffer
        glGenFramebuffers(1, &frame_buffer_);
        glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer_);

        // Setup RGB texture buffer
        glGenTextures(1, &rgb_texture_);
        glBindTexture(GL_TEXTURE_2D, rgb_texture_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rgb_texture_, 0);
        glBindTexture(GL_TEXTURE_2D, 0);

        // Setup depth render buffer. This is only for OpenGL rendering.
        glGenRenderbuffers(1, &depth_buffer_);
        glBindRenderbuffer(GL_RENDERBUFFER, depth_buffer_);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_buffer_);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);

        // Setup depth texture buffer for outputing depth map
        glGenTextures(1, &depth_texture_);
        glBindTexture(GL_TEXTURE_2D, depth_texture_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, depth_texture_, 0);
        glBindTexture(GL_TEXTURE_2D, 0);

        glGenTextures(1, &semantic_texture_);
        glBindTexture(GL_TEXTURE_2D, semantic_texture_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32I, width, height, 0, GL_RED_INTEGER, GL_INT, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, semantic_texture_, 0);
        glBindTexture(GL_TEXTURE_2D, 0);

        glGenTextures(1, &indices_texture_);
        glBindTexture(GL_TEXTURE_2D, indices_texture_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32I, width, height, 0, GL_RGB_INTEGER, GL_INT, 0);
        // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, indices_texture_, 0);
        glBindTexture(GL_TEXTURE_2D, 0);

        glGenTextures(1, &weights_texture_);
        glBindTexture(GL_TEXTURE_2D, weights_texture_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, GL_RGB, GL_FLOAT, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, GL_TEXTURE_2D, weights_texture_, 0);
        // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glBindTexture(GL_TEXTURE_2D, 0);

        if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        	std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    ~GLSemanticRenderer() {
        glDeleteFramebuffers(1, &frame_buffer_);
        glDeleteTextures(1, &rgb_texture_);
        glDeleteTextures(1, &depth_texture_);
        glDeleteTextures(1, &semantic_texture_);
        glDeleteTextures(1, &indices_texture_);
        glDeleteTextures(1, &weights_texture_);
	    glDeleteRenderbuffers(1, &depth_buffer_);
	    delete pinhole_shader_program_;
        delete opencv_shader_program_;
        delete fisheye_shader_program_;
    }

    void render(GLSemanticMesh& glmesh,
                CameraModelType model_type,
                const float* intrinsic,
                DistortParams& distort_params,
                const float* world_to_camera,
                const float min_depth,
                const float max_depth,
                cv::Mat& rgb_image,
                cv::Mat& depth_image,
                cv::Mat& semantic_image,
                cv::Mat& index_image,
                cv::Mat& weight_image,
                const float radius_cutoff = std::numeric_limits<float>::max()) {
        glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer_);
        GLenum buffers[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3, GL_COLOR_ATTACHMENT4};
        glDrawBuffers(5, buffers);

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        float white[] = {1.0, 1.0f, 1.0f, 1.0f};
        glClearBufferfv(GL_COLOR, 0, white);
        float black[] = {0, 0, 0, 0};
        glClearBufferfv(GL_COLOR, 1, black);
        int no_label[] = {0};
        glClearBufferiv(GL_COLOR, 2, no_label);
        int no_indices[] = {-1, -1, -1};
        glClearBufferiv(GL_COLOR, 3, no_indices);
        float no_weights[] = {0.0f, 0.0f, 0.0f};
        glClearBufferfv(GL_COLOR, 4, no_weights);


        if (model_type == CameraModelType::CAMERA_OPENCV) {
            opencv_shader_program_->use();
            opencv_shader_program_->setDistortionParams(distort_params, radius_cutoff);
            setupCamera(opencv_shader_program_, intrinsic, world_to_camera, min_depth, max_depth);
        } else if (model_type == CameraModelType::CAMERA_FISHEYE) {
            fisheye_shader_program_->use();
            fisheye_shader_program_->setDistortionParams(distort_params, radius_cutoff);
            setupCamera(fisheye_shader_program_, intrinsic, world_to_camera, min_depth, max_depth);
        } else {
            pinhole_shader_program_->use();
            setupCamera(pinhole_shader_program_, intrinsic, world_to_camera, min_depth, max_depth);
        }
        glViewport(0, 0, width, height);

        glmesh.draw();

        // Read the render data out
        rgb_image = cv::Mat(height, width, CV_8UC3);
        depth_image = cv::Mat(height, width, CV_32F);
        semantic_image = cv::Mat(height, width, CV_32S);
        index_image = cv::Mat(height, width, CV_32SC3);
        weight_image = cv::Mat(height, width, CV_32FC3);

        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, rgb_image.data);
        // cv::cvtColor(rgb_image, rgb_image, cv::COLOR_RGB2BGR);

        glReadBuffer(GL_COLOR_ATTACHMENT1);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, width, height, GL_RED, GL_FLOAT, depth_image.data);

        glReadBuffer(GL_COLOR_ATTACHMENT2);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, width, height, GL_RED_INTEGER, GL_INT, semantic_image.data);

        glReadBuffer(GL_COLOR_ATTACHMENT3);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, width, height, GL_RGB_INTEGER, GL_INT, index_image.data);

        glReadBuffer(GL_COLOR_ATTACHMENT4);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, width, height, GL_RGB, GL_FLOAT, weight_image.data);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    glm::mat4 convertIntrinsicToProjection(const float* intrinsic, float min_depth, float max_depth) {
        // intrinsic is row-major
        float fx = intrinsic[0*4 + 0];
        float fy = intrinsic[1*4 + 1];
        float cx = intrinsic[0*4 + 2];
        float cy = intrinsic[1*4 + 2];
        glm::mat4 mat(0);       // mat is column major
        mat[0][0] = (2*fx) / width;
        mat[1][0] = 0;
        mat[2][0] = 2 * (0.5f + cx) / width - 1.0f;
        mat[3][0] = 0;

        mat[0][1] = 0;
        mat[1][1] = (2*fy) / height;
        mat[2][1] = 2 * (0.5f + cy) / height - 1.0f;
        mat[3][1] = 0;

        mat[0][2] = 0;
        mat[1][2] = 0;
        mat[2][2] = (max_depth + min_depth) / (max_depth - min_depth);
        mat[3][2] = -(2 * max_depth * min_depth) / (max_depth - min_depth);

        mat[0][3] = 0;
        mat[1][3] = 0;
        mat[2][3] = 1;
        mat[3][3] = 0;
        return mat;
    }

    glm::mat4 convertW2CtoView(const float* world_to_camera) {
        // world_to_camera is row-major
        glm::mat4 mat(0);       // mat is column major
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                mat[j][i] = world_to_camera[i*4+j];
            }
        }
        mat[3][0] = world_to_camera[0*4+3];
        mat[3][1] = world_to_camera[1*4+3];
        mat[3][2] = world_to_camera[2*4+3];
        mat[3][3] = 1.0;
        return mat;
    }

    void setupCamera(GLSemanticProgram* program, const float* intrinsic, const float* world_to_camera, float min_depth, float max_depth) {
        glm::mat4 proj_mat = convertIntrinsicToProjection(intrinsic, min_depth, max_depth);
		// glUniformMatrix4fv(shader_program_->getProjectionMatrixID(), 1, GL_FALSE, &mat1[0][0]);
        program->setProjectionMatrix(proj_mat);

        glm::mat4 mv_mat = convertW2CtoView(world_to_camera);
        // glUniformMatrix4fv(shader_program_->getModelViewMatrixID(), 1, GL_FALSE, &mat2[0][0]);
        program->setModelViewMatrix(mv_mat);
    }

private:
    GLSemanticProgram* pinhole_shader_program_;
    GLSemanticProgram* opencv_shader_program_;
    GLSemanticProgram* fisheye_shader_program_;
    unsigned int vertex_;
    unsigned int frame_buffer_;
    unsigned int rgb_texture_;
    unsigned int semantic_texture_;
    unsigned int indices_texture_;
    unsigned int weights_texture_;
    unsigned int depth_texture_;
    unsigned int depth_buffer_;
    glm::mat4 world_to_camera_;
    int height, width;
};

} // namespace opengl