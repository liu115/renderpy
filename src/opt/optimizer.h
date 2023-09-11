#pragma once
#include <mLibCore.h>

#include "camera/camera_models.h"
#include "opengl/opengl.h"
#include "opt/image.h"
#include "opt/camera.h"
#include "opt/evaluate.h"
#include "opt/multi_scale_mesh.h"

namespace opt {

typedef std::vector<std::vector<unsigned int>> IndexMapping;

static const float MIN_RENDER_DEPTH = 0.02;  // 2cm
static const float MAX_RENDER_DEPTH = 20.0;  // 20m

class OptimizerConfig {
public:
    enum class Type { OPT_GN, OPT_LM };
    OptimizerConfig() {}
    OptimizerConfig(std::string debug_path,
                    int debug_iter,
                    bool use_debug,
                    Type opt_type,
                    int g_ksize,
                    float g_sigma)
        : debug_path(debug_path), debug_iter(debug_iter), use_debug(use_debug), opt_type(opt_type), g_ksize(g_ksize), g_sigma(g_sigma) {}


    void addPhase(int num_iters, float scale_factor, std::string mesh_path) {
        num_iters_.push_back(num_iters);
        scale_factor_.push_back(scale_factor);
        mesh_paths_.push_back(mesh_path);
    }

    int num_threads;    // TODO: add this to config
    std::string debug_path;
    int debug_iter;
    bool use_debug;
    Type opt_type;
    int g_ksize;
    float g_sigma;

    std::vector<int> num_iters_;
    std::vector<float> scale_factor_;
    std::vector<std::string> mesh_paths_;
};



class ColorMapOptimizer {
public:
    ColorMapOptimizer() {}
    ColorMapOptimizer(OptimizerConfig config, opt::CameraWrapper* camera)
        : config_(config), camera_(camera) {
            ms_mesh = new MultiScaleMesh();
            assert (config.num_iters_.size() == config.scale_factor_.size());
            assert (config.num_iters_.size() == config.mesh_paths_.size());
            for (int i = 0; i < config.mesh_paths_.size(); i++) {
                ml::MeshDataf meshdata;
                std::cout << "Loading mesh from " << config.mesh_paths_[i] << std::endl;
                ml::MeshIOf::loadFromPLY(config.mesh_paths_[i], meshdata);
                std::cout << "Loaded mesh with " << meshdata.m_Vertices.size() << " vertices and " << meshdata.m_FaceIndicesVertices.size() << " triangles" << std::endl;
                ml::TriMeshf trimesh(meshdata);
                trimesh.computeNormals();
                ms_mesh->addMesh(trimesh);
            }
            assert (ms_mesh->levels() == config.mesh_paths_.size());
        }
    ~ColorMapOptimizer() {
        delete ms_mesh;
    }

    void addFrame(opt::Image& image, ml::mat4f& world_to_camera) {
        images_.push_back(image);
        poses_.push_back(world_to_camera);
        valid_.push_back(true);
    }

    void scaleCamerasAndImages(
        const float factor,
        opt::CameraWrapper* camera,
        std::vector<opt::Image>& images);

    void prepareGrayAndGradientImages(
        std::vector<opt::Image>& images,
        std::vector<opt::Image>& gray_images,
        std::vector<opt::Image>& grad_x_images,
        std::vector<opt::Image>& grad_y_images);

    float computeResidual(
        IndexMapping& image_to_vertex,
        std::vector<ml::vec3f>& vertex_xyzs,
        std::vector<float>& vertex_intensities,
        opt::CameraWrapper* camera,
        std::vector<Image>& gray_norm_images,
        std::vector<ml::mat4f>& poses,
        std::vector<bool>& images_valid);

    float computeResidualAndJacobian(
        IndexMapping& image_to_vertex,
        std::vector<ml::vec3f>& vertex_xyzs,
        std::vector<float>& vertex_intensities,
        opt::CameraWrapper* camera,
        std::vector<Image>& gray_norm_images,
        std::vector<Image>& dx_images,
        std::vector<Image>& dy_images,
        std::vector<ml::mat4f>& poses,
        std::vector<bool>& images_valid,
        std::vector<Eigen::Matrix<float, 6, 6>>& JTJs,    // H of Hx=b
        std::vector<Eigen::Vector<float, 6>>& JTrs        // b of Hx=b
    );

    std::tuple<IndexMapping,IndexMapping> computeImageVertexVisibility(
        std::vector<ml::vec3f>& vertex_xyzs,
        opt::CameraWrapper* camera,
        std::vector<opt::Image>& depth_images,
        std::vector<ml::mat4f>& poses,
        std::vector<bool>& images_valid);

    void updateVertexIntensity(
        IndexMapping& vertex_to_image,
        std::vector<float>& intensities,
        std::vector<ml::vec3f>& vertex_xyz,
        opt::CameraWrapper* camera,
        std::vector<Image>& images,
        std::vector<ml::mat4f>& poses,
        std::vector<bool>& images_valid);

    void updateVertexColor(
        IndexMapping& vertex_to_image,
        std::vector<ml::vec3f>& vertex_colors,
        std::vector<ml::vec3f>& vertex_xyz,
        opt::CameraWrapper* camera,
        std::vector<Image>& images,
        std::vector<ml::mat4f>& poses,
        std::vector<bool>& images_valid);

    void visualizeAndSaveColorMesh(
        std::string save_path,
        ml::TriMeshf& mesh,
        IndexMapping& vertex_to_image,
        opt::CameraWrapper* camera,
        std::vector<Image>& images,
        std::vector<ml::mat4f>& poses,
        std::vector<bool>& images_valid,
        bool draw_pose);


    void optimizeGN(
        int max_iter,
        int mesh_level,
        opt::CameraWrapper* camera,
        std::vector<Image>& gray_norm_images,
        std::vector<Image>& dx_images,
        std::vector<Image>& dy_images,
        std::vector<ml::mat4f>& poses,
        std::vector<bool>& valid
    );

    void optimizeLM(
        int max_iter,
        int mesh_level,
        opt::CameraWrapper* camera,
        std::vector<Image>& gray_norm_images,
        std::vector<Image>& dx_images,
        std::vector<Image>& dy_images,
        std::vector<ml::mat4f>& poses,
        std::vector<bool>& valid
    );

    void optimize();

    static void setupGLMesh(ml::TriMeshf& mesh, opengl::GLMesh& gl_mesh) {
        // if (!opengl::init()) return;
        for (auto& v: mesh.getVertices()) {
            gl_mesh.addVertex(v.position.x, v.position.y, v.position.z);
            gl_mesh.addVertexColor(v.color.r, v.color.g, v.color.b, v.color.a);
        }
        for (auto& f: mesh.getIndices()) {
            gl_mesh.addFaceIndice(f.x, f.y, f.z);
        }
        gl_mesh.setupGLBuffer();
        // copyTriMeshToGLMesh(mesh, gl_mesh);
    }

    static void renderRGBDepth(
        opengl::GLMesh& gl_mesh,
        opengl::GLRenderer& gl_renderer,
        opt::CameraWrapper* camera,
        ml::mat4f& world_to_camera,
        int image_id,
        opt::Image& render_rgb,
        opt::Image& render_depth)
    {
        cv::Mat gl_rgb_image, gl_depth_image;
        ml::mat4f intrinsic = camera->getIntrinsicMatrix();
        auto params = camera->getDistortionParams();

        gl_renderer.render(gl_mesh, camera->type(),
                            intrinsic.getData(),
                            params,
                            world_to_camera.getData(),
                            MIN_RENDER_DEPTH, MAX_RENDER_DEPTH,
                            gl_rgb_image, gl_depth_image,
                            camera->radius_cutoff_squared());

        render_rgb = Image(image_id, gl_rgb_image);
        render_depth = Image(image_id, gl_depth_image);
    }

    static void render(
        opengl::GLMesh& gl_mesh,
        opengl::GLRenderer& gl_renderer,
        opt::CameraWrapper* camera,
        std::vector<ml::mat4f>& poses,
        std::vector<opt::Image>& render_rgb_images,
        std::vector<opt::Image>& render_depth_images
    ) {
        // assert (cameras.size() == poses.size());
        render_rgb_images.clear();
        render_depth_images.clear();
        for (int image_id = 0; image_id < poses.size(); image_id++) {
            opt::Image render_rgb, render_depth;
            renderRGBDepth(gl_mesh, gl_renderer, camera, poses[image_id],
                           image_id, render_rgb, render_depth);
            render_rgb_images.push_back(render_rgb);
            render_depth_images.push_back(render_depth);
        }
    }

    void computeValidPixelRatio(
        std::vector<opt::Image>& rgb_images,
        opt::CameraWrapper* camera,
        std::vector<ml::mat4f>& poses,
        std::vector<float>& valid_pixel_ratios
    ) {
        std::vector<ml::vec3f> vertex_xyzs;
        std::vector<ml::vec3f> vertex_colors;
        std::vector<opt::Image> render_rgb_images;
        std::vector<opt::Image> render_depth_images;
        // Use the highest resolution mesh for visualization and evaluation
        int mesh_level = ms_mesh->levels() - 1;
        std::cout << "Debug on mesh level: " << mesh_level << std::endl;
        auto& mesh = ms_mesh->getMesh(mesh_level);

        opengl::GLMesh gl_mesh;
        setupGLMesh(mesh, gl_mesh);
        opengl::GLRenderer gl_renderer(camera->height(), camera->width());
        render(gl_mesh, gl_renderer, camera, poses, render_rgb_images, render_depth_images);

        valid_pixel_ratios.resize(rgb_images.size());
        for (int image_id = 0; image_id < rgb_images.size(); image_id++) {
            auto& render_depth = render_depth_images[image_id];
            cv::Mat depth_threshold, valid_mask;
            cv::threshold(render_depth.getCVImage(), depth_threshold, 1e-6, 1.0, cv::THRESH_BINARY);
            depth_threshold.convertTo(valid_mask, CV_8U);
            float num_valid_pixels = cv::countNonZero(valid_mask);
            valid_pixel_ratios[image_id] = num_valid_pixels / (camera->height() * camera->width());
        }
    }

    void computePSNRAll(
        std::vector<opt::Image>& rgb_images,
        opt::CameraWrapper* camera,
        std::vector<ml::mat4f>& poses,
        std::vector<float>& psnrs
    ) {
        std::vector<ml::vec3f> vertex_xyzs;
        std::vector<ml::vec3f> vertex_colors;
        std::vector<opt::Image> render_rgb_images;
        std::vector<opt::Image> render_depth_images;
        // Use the highest resolution mesh for visualization and evaluation
        int mesh_level = ms_mesh->levels() - 1;
        std::cout << "Debug on mesh level: " << mesh_level << std::endl;
        auto& mesh = ms_mesh->getMesh(mesh_level);

        opengl::GLMesh gl_mesh;
        setupGLMesh(mesh, gl_mesh);
        opengl::GLRenderer gl_renderer(camera->height(), camera->width());
        render(gl_mesh, gl_renderer, camera, poses, render_rgb_images, render_depth_images);

        IndexMapping image_to_vertex, vertex_to_image;
        ms_mesh->copyVertexToVector(mesh_level, vertex_xyzs, vertex_colors);
        std::tie(image_to_vertex, vertex_to_image) = computeImageVertexVisibility(
            vertex_xyzs, camera, render_depth_images, poses, valid_);
        updateVertexColor(vertex_to_image, vertex_colors, vertex_xyzs, camera, rgb_images, poses, valid_);

        // Assign the color to the real mesh vertices
        mesh.setColor(ml::vec4f(0, 0, 0, 1.0f));        // Fill all black
        for (int vertex_id = 0; vertex_id < vertex_colors.size(); vertex_id++) {
            if (vertex_to_image[vertex_id].size() > 0) {
                ml::vec4f color(vertex_colors[vertex_id], 1.0f);
                mesh.setColor(vertex_id, color);
            }
        }
        ml::MeshDataf meshdata = mesh.computeMeshData();
        opengl::GLMesh gl_mesh_recolor;
        setupGLMesh(mesh, gl_mesh_recolor);
        render(gl_mesh_recolor, gl_renderer, camera, poses, render_rgb_images, render_depth_images);

        psnrs.resize(rgb_images.size());
        for (int image_id = 0; image_id < rgb_images.size(); image_id++) {
            auto& image = rgb_images[image_id];
            auto& render_rgb = render_rgb_images[image_id];
            auto& render_depth = render_depth_images[image_id];

            cv::Mat depth_threshold, valid_mask;
            cv::threshold(render_depth.getCVImage(), depth_threshold, 1e-6, 1.0, cv::THRESH_BINARY);
            depth_threshold.convertTo(valid_mask, CV_8U);
            float psnr = computePSNR(render_rgb.getCVImage(), image.getCVImage(), valid_mask);
            psnrs[image_id] = psnr;
        }
    }

    float debugEvaluation(
        std::string save_dir,
        int iter,
        std::vector<opt::Image>& rgb_images,
        opt::CameraWrapper* camera,
        std::vector<ml::mat4f>& poses) {

        // Render the images and compute PSNR
        // Save the debug visualization
        std::vector<ml::vec3f> vertex_xyzs;
        std::vector<ml::vec3f> vertex_colors;
        std::vector<opt::Image> render_rgb_images;
        std::vector<opt::Image> render_depth_images;
        // Use the highest resolution mesh for visualization and evaluation
        int mesh_level = ms_mesh->levels() - 1;
        std::cout << "Debug on mesh level: " << mesh_level << std::endl;
        auto& mesh = ms_mesh->getMesh(mesh_level);

        opengl::GLMesh gl_mesh;
        std::cout << "[DEBUG] setup mesh for render pass 1" << std::endl;
        setupGLMesh(mesh, gl_mesh);
        std::cout << "[DEBUG] setup renderer for render pass 1" << std::endl;
        opengl::GLRenderer gl_renderer(camera->height(), camera->width());
        std::cout << "[DEBUG] render pass 1" << std::endl;
        render(gl_mesh, gl_renderer, camera, poses, render_rgb_images, render_depth_images);

        IndexMapping image_to_vertex, vertex_to_image;
        ms_mesh->copyVertexToVector(mesh_level, vertex_xyzs, vertex_colors);
        std::tie(image_to_vertex, vertex_to_image) = computeImageVertexVisibility(
            vertex_xyzs, camera, render_depth_images, poses, valid_);
        updateVertexColor(vertex_to_image, vertex_colors, vertex_xyzs, camera, rgb_images, poses, valid_);

        // Assign the color to the real mesh vertices
        mesh.setColor(ml::vec4f(0, 0, 0, 1.0f));        // Fill all black
        for (int vertex_id = 0; vertex_id < vertex_colors.size(); vertex_id++) {
            if (vertex_to_image[vertex_id].size() > 0) {
                ml::vec4f color(vertex_colors[vertex_id], 1.0f);
                mesh.setColor(vertex_id, color);
            }
        }
        std::string iter_str = std::to_string(iter);
        size_t n_zero = 5;
        iter_str = std::string(n_zero - std::min(n_zero, iter_str.length()), '0') + iter_str;

        ml::MeshDataf meshdata = mesh.computeMeshData();
        std::filesystem::create_directories(save_dir + "/mesh/");
        ml::MeshIOf::saveToPLY(save_dir + "/mesh/recolor_mesh_" + iter_str + ".ply", meshdata);

        opengl::GLMesh gl_mesh_recolor;
        std::cout << "[DEBUG] setup mesh for render pass 2" << std::endl;
        setupGLMesh(mesh, gl_mesh_recolor);
        std::cout << "[DEBUG] render pass 2" << std::endl;
        render(gl_mesh_recolor, gl_renderer, camera, poses, render_rgb_images, render_depth_images);
        int cnt = 0;
        float sum_psnr = 0.0;
        std::vector<std::pair<std::string, float>> all_psnr;
        std::filesystem::create_directories(save_dir + "/edge/");
        std::filesystem::create_directories(save_dir + "/render/");
        for (int image_id = 0; image_id < rgb_images.size(); image_id++) {
            if (!valid_[image_id]) continue;
            auto& image = rgb_images[image_id];
            auto& render_rgb = render_rgb_images[image_id];
            auto& render_depth = render_depth_images[image_id];

            cv::Mat depth_threshold, valid_mask;
            cv::threshold(render_depth.getCVImage(), depth_threshold, 1e-6, 1.0, cv::THRESH_BINARY);
            depth_threshold.convertTo(valid_mask, CV_8U);
            float psnr = computePSNR(render_rgb.getCVImage(), image.getCVImage(), valid_mask);
            std::cout << "Image " << image_id << " PSNR: " << psnr << std::endl;
            sum_psnr += psnr;
            cnt++;
            all_psnr.push_back(std::make_pair(image.getImageName(), psnr));

            auto image_with_edge = visualizeDepthEdge(render_depth.getCVImage(), image.getCVImage());
            cv::putText(image_with_edge,
                        "PSNR: " + std::to_string(psnr),    //text
                        cv::Point(50, image_with_edge.rows / 2),    //top-left position
                        cv::FONT_HERSHEY_DUPLEX,
                        3.0,
                        CV_RGB(118, 185, 0), //font color
                        2);
            // Clone the images in render_rgb
            cv::Mat out_rgb = render_rgb.getCVImage().clone();
            cv::putText(out_rgb,
                        "PSNR: " + std::to_string(psnr),    //text
                        cv::Point(50, out_rgb.rows / 2),    //top-left position
                        cv::FONT_HERSHEY_DUPLEX,
                        3.0,
                        CV_RGB(118, 185, 0), //font color
                        2);

            cv::imwrite(save_dir + "/edge/" + image.getImageName() + "_iter-" + iter_str + ".jpg", image_with_edge);
            cv::imwrite(save_dir + "/render/" + image.getImageName() + "_iter-" + iter_str + ".jpg", out_rgb);
        }
        assert (cnt > 0);
        std::cout << "Average PSNR: " << sum_psnr / cnt << std::endl;

        std::ofstream fs(save_dir + "/psnr_iter-" + iter_str + ".txt");
        for (auto p: all_psnr) {
            fs << p.first << " " << p.second << std::endl;
        }
        fs.close();

        return sum_psnr / cnt;
    }

    void dumpResultCOLMAP(const std::string save_path) {
        std::filesystem::create_directories(save_path);
        std::cout << "Dumping result to " << save_path << " in colmap format" << std::endl;

        // Write cameras.txt
        std::ofstream cam_fs(save_path + "/cameras.txt");
        cam_fs << "# Camera list with one line of data per camera:" << std::endl;
        cam_fs << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]" << std::endl;
        cam_fs << "# Number of cameras: " << 1 << std::endl;
        cam_fs << "1 " << camera_->getString() << std::endl;
        cam_fs.close();


        // Write images.txt
        int cnt = 0;
        for (int image_id = 0; image_id < images_.size(); image_id++) {
            if (valid_[image_id]) cnt++;
        }

        std::ofstream img_fs(save_path + "/images.txt");
        img_fs << "# Image list with two lines of data per image:" << std::endl;
        img_fs << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME" << std::endl;
        img_fs << "#   POINTS2D[] as (X, Y, POINT3D_ID)" << std::endl;
        img_fs << "# Number of images: " << cnt << ", mean observations per image: " << 0 << std::endl;
        for (int image_id = 0; image_id < images_.size(); image_id++) {
            if (!valid_[image_id]) {
                std::cout << "Ignore invalid image: " << images_[image_id].getImageName() << std::endl;
                continue;
            }
            auto filename = std::filesystem::path(images_[image_id].image_path).filename().generic_string();
            img_fs << image_id << " ";
            ml::quatf qvec(poses_[image_id]);
            img_fs << qvec << " " << poses_[image_id].getTranslation() << " 1 " << filename << std::endl;
            img_fs << std::endl;    // Empty line for feature points
        }
        img_fs.close();
        std::ofstream pts_fs(save_path + "/points3D.txt");
        pts_fs << "# 3D point list with one line of data per point:" << std::endl;
        pts_fs << "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)" << std::endl;
        pts_fs << "# Number of points: " << 0 << ", mean track length: " << 0 << std::endl;
        pts_fs.close();
    }

    void saveMatrixToFile(const std::string& path, const ml::mat4f& mat) {
        std::ofstream out(path.c_str());
        for (unsigned int i = 0; i < 4; i++) {
            for (unsigned int j = 0; j < 4; j++) {
                out << mat(i, j);
                if (j < 3) out << " ";
            }
            out << "\n";
        }
    }


    void dumpResult(const std::string save_path) {
        std::cout << "Dumping result to " << save_path << std::endl;
        std::filesystem::create_directories(std::filesystem::path(save_path) / "depth");
        std::filesystem::create_directories(std::filesystem::path(save_path) / "poses");

        std::filesystem::path output_path = std::filesystem::path(save_path) / "camera_params.txt";
        std::ofstream fs(output_path);
        fs << camera_->getString(true) << std::endl;
        fs.close();

        std::vector<opt::Image> render_rgb_images;
        std::vector<opt::Image> render_depth_images;
        // Use the highest resolution mesh
        int mesh_level = ms_mesh->levels() - 1;
        auto& mesh = ms_mesh->getMesh(mesh_level);

        opengl::GLMesh gl_mesh;
        setupGLMesh(mesh, gl_mesh);
        opengl::GLRenderer gl_renderer(camera_->height(), camera_->width());
        render(gl_mesh, gl_renderer, camera_, poses_, render_rgb_images, render_depth_images);
        for (int image_id = 0; image_id < images_.size(); image_id++) {
            if (!valid_[image_id]) continue;
            auto filename = std::filesystem::path(images_[image_id].image_path).stem().generic_string();

            output_path = std::filesystem::path(save_path) / "depth" / filename;
            cv::imwrite(output_path.generic_string() + ".tiff", render_depth_images[image_id].getCVImage());

            output_path = std::filesystem::path(save_path) / "poses" / filename;
            saveMatrixToFile(output_path.generic_string() + ".txt", poses_[image_id]);
        }
    }
private:
    // Hyper-parameters

    const float MAX_OPT_DEPTH = 20.0;     // 20m
    const float DEPTH_THRESHOLD = 0.02;   // 2cm
    const int MIN_OPT_OBSERVATIONS = 50;
    const float MIN_DEPTH_VALID_RATIO = 0.3;
    const float MIN_VALID_PSNR = 12.0;
    const int LM_MAX_ITER = 10;
    const int NO_NEW_OPTIMAL_ITER_THRESHOLD = 10;
    const float INIT_LM_LAMBDA = 64.0f;

    OptimizerConfig config_;

    // Mesh and image data
    MultiScaleMesh* ms_mesh;
    std::vector<opt::Image> images_;
    // std::vector<opt::CameraWrapper*> cameras_;
    opt::CameraWrapper* camera_;
    std::vector<ml::mat4f> poses_;   // world_to_camera
    std::vector<bool> valid_;

    // Logging
    std::vector<float> all_psnr;
    std::vector<float> all_residuals;
};

}; // namespace opt