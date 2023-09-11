#include <mLibCore.h>
#include "image.h"
#include "optimizer.h"
#include "mesh_utils/mesh.h"


namespace opt {


std::tuple<IndexMapping,IndexMapping> ColorMapOptimizer::computeImageVertexVisibility(
    std::vector<ml::vec3f>& vertex_xyzs,
    opt::CameraWrapper* camera,
    std::vector<opt::Image>& depth_images,
    std::vector<ml::mat4f>& poses,
    std::vector<bool>& images_valid) {

    IndexMapping image_to_vertex;
    image_to_vertex.resize(depth_images.size());
    IndexMapping vertex_to_image;
    vertex_to_image.resize(vertex_xyzs.size());

#pragma omp parallel for
    for (int image_id = 0; image_id < depth_images.size(); image_id++) {
        if (!images_valid[image_id]) continue;
        auto &depth = depth_images[image_id];
        // auto &camera = cameras[image_id];
        auto &world_to_camera = poses[image_id];
        for (int vertex_id = 0; vertex_id < vertex_xyzs.size(); vertex_id++) {
            ml::vec3f p = vertex_xyzs[vertex_id];
            p = world_to_camera * p;
            if (p.z <= 0 || p.z > MAX_OPT_DEPTH) continue;
            auto uvz = camera->projectPoint(p);
            float u = uvz.x;
            float v = uvz.y;
            if (!depth.inRange(u, v)) continue;
            // TODO: check boundary mask implementation
            // if (mask.getNearestPixelValue<uchar>(u, v) == 0) continue;

            float pixel_depth = depth.getNearestPixelValue<float>(u, v);
            if (std::fabs(pixel_depth - p.z) > DEPTH_THRESHOLD) continue;
#pragma omp critical(computeImageVertexVisibility)
            {
                image_to_vertex[image_id].push_back(vertex_id);
                vertex_to_image[vertex_id].push_back(image_id);
            }
        }
    }
    // for (int image_id = 0; image_id < images.size(); image_id++) {
    //     printf("[%d] Visible ratio: %f\n", image_id, 1.0f * image_to_vertex[image_id].size() / vertices.size());
    // }
    return std::make_tuple(image_to_vertex, vertex_to_image);
}

void ColorMapOptimizer::scaleCamerasAndImages(
    const float factor,
    opt::CameraWrapper* camera,
    std::vector<opt::Image>& images) {
    camera->scale(factor);
    for (int image_id = 0; image_id < images.size(); image_id++) {
        // auto &camera = cameras[image_id];
        auto &image = images[image_id];
        int scaled_width = static_cast<int>(factor * image.width() + 0.5f);
        int scaled_height = static_cast<int>(factor * image.height() + 0.5f);
        image.resize(scaled_height, scaled_width);
    }
}

void ColorMapOptimizer::prepareGrayAndGradientImages(
    std::vector<opt::Image>& images,
    std::vector<opt::Image>& gray_images,
    std::vector<opt::Image>& grad_x_images,
    std::vector<opt::Image>& grad_y_images) {
    // Input RGB [0-255] images
    // Output gray images [0-1], gradient images [0-1] in X and Y direction

    assert (gray_images.size() == 0);
    assert (grad_x_images.size() == 0);
    assert (grad_y_images.size() == 0);
    for (int image_id = 0; image_id < images.size(); image_id++) {
        auto gray = opt::Image(images[image_id]);
        gray.convertToGrayscale();      // From RGB to gray
        gray.normalize();               // Normalize to [0, 1]
        gray.applyGaussian(config_.g_ksize, config_.g_sigma);       // Smoothing
        gray_images.push_back(gray);

        auto grad_x = opt::Image(gray);     // Copy
        auto grad_y = opt::Image(gray);     // Copy
        grad_x.computeGradientX();
        grad_y.computeGradientY();
        grad_x_images.push_back(grad_x);
        grad_y_images.push_back(grad_y);
    }


}

void ColorMapOptimizer::updateVertexIntensity(
        IndexMapping& vertex_to_image,
        std::vector<float>& intensities,
        std::vector<ml::vec3f>& vertex_xyz,
        opt::CameraWrapper* camera,
        std::vector<Image>& images,         // Gray scale normalized images
        std::vector<ml::mat4f>& poses,
        std::vector<bool>& images_valid) {

    assert (images[0].channels() == 1);
    if (intensities.size() != vertex_to_image.size()) {
        intensities.resize(vertex_to_image.size());
    }
    std::fill(intensities.begin(), intensities.end(), 0.0f);
#pragma omp parallel for
    for (int vertex_id = 0; vertex_id < vertex_to_image.size(); vertex_id++) {
        int cnt = 0;
        float sum = 0.0f;

        for (int image_id: vertex_to_image[vertex_id]) {
            if (!images_valid[image_id]) continue;
            ml::vec3f p = vertex_xyz[vertex_id];
            auto& world_to_camera = poses[image_id];
            // auto& camera = cameras[image_id];
            auto& image = images[image_id];
            p = world_to_camera * p;
            if (p.z <= 0 || p.z > MAX_OPT_DEPTH) continue;
            auto uvz = camera->projectPoint(p);
            float u = uvz.x;
            float v = uvz.y;
            if (!image.inRange(u, v)) continue;
            sum += image.getPixelValue<float>(u, v);
            cnt++;
        }
        if (cnt > 0) {
            intensities[vertex_id] = sum / cnt;
        } else {
            intensities[vertex_id] = 0.0f;
        }
    }
}


void ColorMapOptimizer::updateVertexColor(
        IndexMapping& vertex_to_image,
        std::vector<ml::vec3f>& vertex_colors,
        std::vector<ml::vec3f>& vertex_xyz,
        opt::CameraWrapper* camera,
        std::vector<Image>& images,         // RGB [0-255] images
        std::vector<ml::mat4f>& poses,
        std::vector<bool>& images_valid) {
    std::cout << "Channels: " << images[0].channels() << std::endl;
    assert (images[0].channels() == 3);
    if (vertex_colors.size() != vertex_to_image.size()) {
        vertex_colors.resize(vertex_to_image.size());
    }

#pragma omp parallel for
    for (int vertex_id = 0; vertex_id < vertex_to_image.size(); vertex_id++) {
        int cnt = 0;
        ml::vec3f sum = ml::vec3f(0.0f, 0.0f, 0.0f);

        for (int image_id: vertex_to_image[vertex_id]) {
            if (!images_valid[image_id]) continue;
            ml::vec3f p = vertex_xyz[vertex_id];
            auto& world_to_camera = poses[image_id];
            // auto& camera = cameras[image_id];
            auto& image = images[image_id];
            p = world_to_camera * p;
            if (p.z <= 0 || p.z > MAX_OPT_DEPTH) continue;
            auto uvz = camera->projectPoint(p);
            float u = uvz.x;
            float v = uvz.y;
            if (!image.inRange(u, v)) continue;
            sum += image.getRGBPixelValue(u, v);
            cnt++;

        }
        // TODO: Could apply some outlier removal here
        if (cnt > 0) {
            vertex_colors[vertex_id] = sum / cnt;
        } else {
            vertex_colors[vertex_id] = 0;
        }
    }
}

void ColorMapOptimizer::visualizeAndSaveColorMesh(std::string save_path,
                     ml::TriMeshf& mesh,
                     IndexMapping& vertex_to_image,
                     opt::CameraWrapper* camera,
                     std::vector<Image>& images,
                     std::vector<ml::mat4f>& poses,
                     std::vector<bool>& images_valid,
                     bool draw_pose) {

    auto& vertices = mesh.getVertices();
    std::vector<ml::vec3f> vertex_xyz(vertices.size());
    std::vector<ml::vec3f> vertex_colors(vertices.size());
    std::fill(vertex_colors.begin(), vertex_colors.end(), ml::vec3f(0, 0, 0));
    for (int vertex_id = 0; vertex_id < vertices.size(); vertex_id++) {
        vertex_xyz[vertex_id] = vertices[vertex_id].position;
    }
    updateVertexColor(vertex_to_image, vertex_colors, vertex_xyz, camera, images, poses, images_valid);

    // Visualize vertex color
    // updateVerticeRGBColor(vertices, dump_vertices_color, rgb_images, vertex_to_image);
    mesh.setColor(ml::vec4f(0, 0, 0, 1.0f));
    for (int vertex_id = 0; vertex_id < vertex_colors.size(); vertex_id++) {
        if (vertex_to_image[vertex_id].size() > 0) {
            ml::vec4f color(vertex_colors[vertex_id], 1.0f);
            mesh.setColor(vertex_id, color);
        }
    }

    ml::MeshDataf mesh_data = mesh.computeMeshData();
    // Visualize the camera poses
    if (draw_pose) {
        for (int image_id = 0; image_id < images.size(); image_id++) {
            ml::mat4f camera_to_world = poses[image_id].getInverse();
            drawPoseOnMesh(mesh_data, camera_to_world, 0.02, 0.2);
        }
    }
    ml::MeshIOf::saveToPLY(save_path, mesh_data);
}

ml::mat4f getLocalTransformPose(const Eigen::Vector<float, 6> x) {
    Eigen::Matrix<float, 4, 4> pose_delta_eigen = Eigen::Matrix<float, 4, 4>::Identity();
    pose_delta_eigen.block<3, 3>(0, 0) =
            (Eigen::AngleAxisf(x(2), Eigen::Vector3f::UnitZ()) *
             Eigen::AngleAxisf(x(1), Eigen::Vector3f::UnitY()) *
             Eigen::AngleAxisf(x(0), Eigen::Vector3f::UnitX())).matrix();
    pose_delta_eigen.block<3, 1>(0, 3) = x.block<3, 1>(3, 0);

    ml::mat4f pose_delta;
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) pose_delta.at(i, j) = pose_delta_eigen(i, j);
    return pose_delta;
}


float ColorMapOptimizer::computeResidual(
    IndexMapping& image_to_vertex,
    std::vector<ml::vec3f>& vertex_xyzs,
    std::vector<float>& vertex_intensities,
    opt::CameraWrapper* camera,
    std::vector<Image>& gray_norm_images,
    std::vector<ml::mat4f>& poses,
    std::vector<bool>& images_valid) {

    assert (vertex_xyzs.size() == vertex_intensities.size());
    // assert (gray_norm_images.size() == cameras.size());
    assert (gray_norm_images.size() == poses.size());
    float total_r2 = 0.0f;
#pragma omp parallel for reduction(+:total_r2)
    for (int image_id = 0; image_id < gray_norm_images.size(); image_id++) {
        if (!images_valid[image_id]) continue;

        auto& world_to_camera = poses[image_id];
        // auto& camera = cameras[image_id];
        auto& gray_norm = gray_norm_images[image_id];

        double r2 = 0;      // residual

        int cnt = 0;
        for (int vertex_id: image_to_vertex[image_id]) {
            auto p = vertex_xyzs[vertex_id];
            p = world_to_camera * p;
            if (p.z <= 0 || p.z > MAX_OPT_DEPTH) continue;
            auto uvz = camera->projectPoint(p);
            float u = uvz.x;
            float v = uvz.y;
            if (!gray_norm.inRange(u, v)) continue;

            float c = gray_norm.getPixelValue<float>(u, v);
            float r = c - vertex_intensities[vertex_id];
            r2 += r * r;
            cnt++;
        }
        total_r2 += r2;
    }
    return total_r2;
}


float ColorMapOptimizer::computeResidualAndJacobian(
    IndexMapping& image_to_vertex,
    std::vector<ml::vec3f>& vertex_xyzs,
    std::vector<float>& vertex_intensities,
    opt::CameraWrapper* camera,
    std::vector<Image>& gray_norm_images,
    std::vector<Image>& grad_x_images,
    std::vector<Image>& grad_y_images,
    std::vector<ml::mat4f>& poses,
    std::vector<bool>& images_valid,
    std::vector<Eigen::Matrix<float, 6, 6>>& JTJs,    // H of Hx=b
    std::vector<Eigen::Vector<float, 6>>& JTrs       // b of Hx=b
) {
    assert (vertex_xyzs.size() == vertex_intensities.size());
    // assert (gray_norm_images.size() == cameras.size());
    assert (gray_norm_images.size() == poses.size());
    assert (gray_norm_images.size() == grad_x_images.size());
    assert (gray_norm_images.size() == grad_y_images.size());
    float total_r2 = 0.0f;
    JTJs.resize(gray_norm_images.size());
    JTrs.resize(gray_norm_images.size());
#pragma omp parallel for reduction(+:total_r2)
    for (int image_id = 0; image_id < gray_norm_images.size(); image_id++) {
        if (!images_valid[image_id]) continue;

        Eigen::Vector<float, 6> jacobian;
        double r2 = 0;      // residual
        int cnt = 0;
        auto& world_to_camera = poses[image_id];
        // auto& camera = cameras[image_id];
        auto& gray_norm = gray_norm_images[image_id];

        JTJs[image_id].setZero();
        JTrs[image_id].setZero();

        for (int vertex_id: image_to_vertex[image_id]) {
            jacobian.setZero();
            auto p = vertex_xyzs[vertex_id];
            p = world_to_camera * p;
            if (p.z <= 0 || p.z > MAX_OPT_DEPTH) continue;
            auto uvz = camera->projectPoint(p);
            float u = uvz.x;
            float v = uvz.y;
            if (!gray_norm.inRange(u, v)) continue;

            float c = gray_norm.getPixelValue<float>(u, v);
            float r = c - vertex_intensities[vertex_id];
            float dx = grad_x_images[image_id].getPixelValue<float>(u, v);
            float dy = grad_y_images[image_id].getPixelValue<float>(u, v);
            Eigen::Vector<float, 2> J_image(dx, dy);
            auto J_xyz = camera->computeJacobianByWorld(p);     // 2, 3
            Eigen::Matrix<float, 3, 6> J_world;
            J_world <<
                0, p.z, -p.y, 1, 0, 0,
                -p.z, 0, p.x, 0, 1, 0,
                p.y, -p.x, 0, 0, 0, 1;

            jacobian = (J_image.transpose() * J_xyz * J_world).row(0);     // 1, 6
            JTJs[image_id] += jacobian * jacobian.transpose();
            JTrs[image_id] += jacobian * r;
            r2 += r * r;
            cnt++;
        }
        total_r2 += r2;
    }
    return total_r2;
}


void ColorMapOptimizer::optimizeGN(
    int max_iter,
    int mesh_level,
    opt::CameraWrapper* camera,
    std::vector<Image>& gray_norm_images,
    std::vector<Image>& grad_x_images,
    std::vector<Image>& grad_y_images,
    std::vector<ml::mat4f>& poses,
    std::vector<bool>& valid
) {

    std::vector<ml::vec3f> vertex_xyzs;
    std::vector<ml::vec3f> vertex_colors;
    std::vector<float> vertex_intensities;
    ms_mesh->copyVertexToVector(mesh_level, vertex_xyzs, vertex_colors);

    float optimal_residual = std::numeric_limits<float>::infinity();
    int iter_without_new_optimal = 0;

    opengl::GLMesh gl_mesh;
    setupGLMesh(ms_mesh->getMesh(mesh_level), gl_mesh);
    opengl::GLRenderer gl_renderer(camera->height(), camera->width());
    for (int iter = 0; iter < max_iter; iter++) {

        std::vector<Eigen::Matrix<float, 6, 6>> JTJs;     // H of Hx=b
        std::vector<Eigen::Vector<float, 6>> JTrs;        // b of Hx=b

        IndexMapping image_to_vertex, vertex_to_image;
        std::vector<opt::Image> render_rgb_images;
        std::vector<opt::Image> render_depth_images;

        std::cout << "Render RGB and depth" << std::endl;
        render(gl_mesh, gl_renderer, camera, poses, render_rgb_images, render_depth_images);

        std::cout << "Compute image-to-vertex visibility" << std::endl;
        std::tie(image_to_vertex, vertex_to_image) = computeImageVertexVisibility(
            vertex_xyzs, camera, render_depth_images, poses, valid);

        // Filter out images with too few observations
        for (int image_id = 0; image_id < gray_norm_images.size(); image_id++) {
            if (!valid[image_id]) continue;
            if (image_to_vertex[image_id].size() <= MIN_OPT_OBSERVATIONS) {
                std::cout << gray_norm_images[image_id].getImageName() << " has too few observations" << std::endl;
                valid[image_id] = false;
            }
        }

        updateVertexIntensity(vertex_to_image, vertex_intensities, vertex_xyzs, camera, gray_norm_images, poses, valid);

        if (config_.use_debug && iter % config_.debug_iter == 0) {
            // visualizeAndSaveColorMesh(
            //     debug_dir + "/recolor_mesh_" + std::to_string(iter) + ".ply",
            //     ms_mesh->getLargestMesh(), vertex_to_image, cameras, images, poses, true
            // );
            // Use the original images and cameras for evaluation and visualization
            const float psnr = debugEvaluation(config_.debug_path, iter, images_, camera_, poses);
            all_psnr.push_back(psnr);
        }
        float residual = computeResidualAndJacobian(
            image_to_vertex,
            vertex_xyzs,
            vertex_intensities,
            camera,
            gray_norm_images,
            grad_x_images,
            grad_y_images,
            poses,
            valid,
            JTJs, JTrs);
        all_residuals.push_back(residual);
        // printf("#iter: %d, image (%d / %ld) #obs: %d, r2=%lf\n", iter, image_id, image_to_vertex.size(), cnt, r2);
        printf("### iter=%d, total r2=%lf\n", iter, residual);

        for (int image_id = 0; image_id < poses.size(); image_id++) {
            if (!valid[image_id]) continue;
            Eigen::Vector<float, 6> x = JTJs[image_id].ldlt().solve(-JTrs[image_id]);
            ml::mat4f pose_delta = getLocalTransformPose(x);
            poses[image_id] = pose_delta * poses[image_id];
        }

        if (residual < optimal_residual) {
            optimal_residual = residual;
            iter_without_new_optimal = 0;
        } else {
            iter_without_new_optimal++;
        }

        if (iter_without_new_optimal >= NO_NEW_OPTIMAL_ITER_THRESHOLD) {
            std::cout << "No new optimal. Early exit." << std::endl;
            break;
        }
    }
}


void ColorMapOptimizer::optimizeLM(
    int max_iter,
    int mesh_level,
    opt::CameraWrapper* camera,
    std::vector<Image>& gray_norm_images,
    std::vector<Image>& grad_x_images,
    std::vector<Image>& grad_y_images,
    std::vector<ml::mat4f>& poses,
    std::vector<bool>& valid
) {

    std::vector<ml::vec3f> vertex_xyzs;
    std::vector<ml::vec3f> vertex_colors;
    std::vector<float> vertex_intensities;
    ms_mesh->copyVertexToVector(mesh_level, vertex_xyzs, vertex_colors);

    float lambda = INIT_LM_LAMBDA;
    float optimal_residual = std::numeric_limits<float>::infinity();
    int iter_without_new_optimal = 0;

    opengl::GLMesh gl_mesh;
    setupGLMesh(ms_mesh->getMesh(mesh_level), gl_mesh);
    opengl::GLRenderer gl_renderer(camera->height(), camera->width());
    for (int iter = 0; iter < max_iter; iter++) {

        std::vector<Eigen::Matrix<float, 6, 6>> JTJs;     // H of Hx=b
        std::vector<Eigen::Vector<float, 6>> JTrs;        // b of Hx=b

        IndexMapping image_to_vertex, vertex_to_image;
        std::vector<opt::Image> render_rgb_images;
        std::vector<opt::Image> render_depth_images;

        std::cout << "Render RGB and depth" << std::endl;
        render(gl_mesh, gl_renderer, camera, poses, render_rgb_images, render_depth_images);

        std::cout << "Compute image-to-vertex visibility" << std::endl;
        std::tie(image_to_vertex, vertex_to_image) = computeImageVertexVisibility(
            vertex_xyzs, camera, render_depth_images, poses, valid);

        // Filter out images with too few observations
        for (int image_id = 0; image_id < gray_norm_images.size(); image_id++) {
            if (!valid[image_id]) continue;
            if (image_to_vertex[image_id].size() <= MIN_OPT_OBSERVATIONS) {
                std::cout << gray_norm_images[image_id].getImageName() << " has too few observations" << std::endl;
                valid[image_id] = false;
            }
        }

        updateVertexIntensity(vertex_to_image, vertex_intensities, vertex_xyzs, camera, gray_norm_images, poses, valid);

        if (config_.use_debug && iter % config_.debug_iter == 0) {
            // This will change the mesh color in the largest ms_mesh
            // visualizeAndSaveColorMesh(
            //     debug_dir + "/recolor_mesh_" + std::to_string(iter) + ".ply",
            //     ms_mesh->getLargestMesh(), vertex_to_image, cameras, images, poses, true
            // );
            // Use the original images and cameras for evaluation and visualization
            const float psnr = debugEvaluation(config_.debug_path, iter, images_, camera_, poses);
            all_psnr.push_back(psnr);
        }

        float initial_residual = computeResidualAndJacobian(
            image_to_vertex,
            vertex_xyzs,
            vertex_intensities,
            camera,
            gray_norm_images,
            grad_x_images,
            grad_y_images,
            poses,
            valid,
            JTJs, JTrs);

        all_residuals.push_back(initial_residual);
        printf("### iter=%d, total r2=%lf\n", iter, initial_residual);
        // Compute jacobian and residual
        for (int lm_iter = 0; lm_iter < LM_MAX_ITER; lm_iter++) {
            // LM inner loop (do neither update the vertex intensity nor recompute the depth)
            std::vector<ml::mat4f> new_poses;
            for (int image_id = 0; image_id < poses.size(); image_id++) {
                if (!valid[image_id]) continue;
                Eigen::Matrix<float, 6, 6> JTJ_LM = JTJs[image_id] + lambda * Eigen::Matrix<float, 6, 6>::Identity();
                Eigen::Vector<float, 6> x = JTJ_LM.ldlt().solve(-JTrs[image_id]);

                ml::mat4f pose_delta = getLocalTransformPose(x);
                new_poses.push_back(pose_delta * poses[image_id]);
            }
            // Evaluate the new poses
            float residual = computeResidual(
                    image_to_vertex,
                    vertex_xyzs,
                    vertex_intensities,
                    camera,
                    gray_norm_images,
                    new_poses,
                    valid);
            printf("#LM iter: %d, total r2=%lf, lambda=%lf\n", lm_iter, residual, lambda);

            if (residual < initial_residual || lm_iter == LM_MAX_ITER - 1) {
                lambda /= 2.0f;
                // Update the poses
                for (int image_id = 0; image_id < poses.size(); image_id++)
                    poses[image_id] = new_poses[image_id];
                break;
            } else {
                lambda *= 2.0f;
            }
        }

        if (initial_residual < optimal_residual) {
            optimal_residual = initial_residual;
            iter_without_new_optimal = 0;
        } else {
            iter_without_new_optimal++;
        }

        if (iter_without_new_optimal >= NO_NEW_OPTIMAL_ITER_THRESHOLD) {
            std::cout << "No new optimal. Early exit." << std::endl;
            break;
        }
    }
}

void ColorMapOptimizer::optimize() {
    if (config_.use_debug) {
        std::filesystem::create_directories(std::filesystem::path(config_.debug_path));
    }
    all_psnr.clear();
    all_residuals.clear();

    for (int mesh_level = 0; mesh_level < ms_mesh->levels(); mesh_level++) {
        std::cout << "Optimize mesh level " << mesh_level << std::endl;
        std::cout << "Scale factor: " << config_.scale_factor_[mesh_level] << std::endl;
        std::cout << "Iteration: " << config_.num_iters_[mesh_level] << std::endl;
        std::cout << config_.mesh_paths_[mesh_level] << std::endl;
        if (config_.num_iters_[mesh_level] <= 0) {
            continue;
        }

        // Filter out the images that have large invalid depth areas
        std::vector<float> image_valid_ratios;
        computeValidPixelRatio(images_, camera_, poses_, image_valid_ratios);
        assert (image_valid_ratios.size() == images_.size());
        for (int i = 0; i < image_valid_ratios.size(); i++) {
            if (image_valid_ratios[i] < MIN_DEPTH_VALID_RATIO) {
                std::cout << images_[i].getImageName() << " has too few valid depth pixels. Ratio: " << image_valid_ratios[i] << std::endl;
                valid_[i] = false;
            }
        }
        // Filter out the images that too low render PSNR
        std::vector<float> image_psnrs;
        computePSNRAll(images_, camera_, poses_, image_psnrs);
        assert (image_psnrs.size() == images_.size());
        for (int i = 0; i < image_psnrs.size(); i++) {
            if (image_psnrs[i] < MIN_VALID_PSNR) {
                std::cout << images_[i].getImageName() << " has too low PSNR: " << image_psnrs[i] << std::endl;
                valid_[i] = false;
            }
        }

        // Copy the images and cameras
        std::vector<opt::Image> scaled_images(images_);
        opt::CameraWrapper* scaled_camera = new opt::CameraWrapper(std::move(*camera_));
        std::vector<ml::mat4f> cur_poses(poses_);   // world_to_camera
        std::vector<bool> cur_valid(valid_);
        scaleCamerasAndImages(config_.scale_factor_[mesh_level], scaled_camera, scaled_images);

        std::vector<opt::Image> gray_norm_images;
        std::vector<opt::Image> grad_x_images;
        std::vector<opt::Image> grad_y_images;
        prepareGrayAndGradientImages(scaled_images, gray_norm_images, grad_x_images, grad_y_images);

        if (config_.opt_type == OptimizerConfig::Type::OPT_GN) {
            std::cout << "Optimize with Gauss-Newton" << std::endl;
            optimizeGN(config_.num_iters_[mesh_level],
                        mesh_level,
                        scaled_camera,
                        gray_norm_images,
                        grad_x_images,
                        grad_y_images,
                        cur_poses,
                        cur_valid);
        } else if (config_.opt_type == OptimizerConfig::Type::OPT_LM) {
            std::cout << "Optimize with Levenberg–Marquardt" << std::endl;
            optimizeLM(config_.num_iters_[mesh_level],
                        mesh_level,
                        scaled_camera,
                        gray_norm_images,
                        grad_x_images,
                        grad_y_images,
                        cur_poses,
                        cur_valid);
        } else {
            throw std::runtime_error("Unknown optimization type");
        }

        // Copy the poses back
        poses_ = cur_poses;
        valid_ = cur_valid;
    }

    // Run evaluation in the end
    const float psnr = debugEvaluation(config_.debug_path, 9999, images_, camera_, poses_);
    all_psnr.push_back(psnr);

    std::cout << "Residuals:" << std::endl;
    for (int i = 0; i < all_residuals.size(); i++) {
        if (i == 0)
            std::cout << all_residuals[i];
        else
            std::cout << ", " << all_residuals[i];
    }
    std::cout << std::endl;
    std::cout << "PSNR:" << std::endl;
    for (int i = 0; i < all_psnr.size(); i++) {
        if (i == 0)
            std::cout << all_psnr[i];
        else
            std::cout << ", " << all_psnr[i];
    }
    std::cout << std::endl;

    if (config_.use_debug) {
        std::cout << "Saving debug results into: " << config_.debug_path << std::endl;
        // Write the residual and PSNR log during optimization into files
        std::ofstream residual_ofs(config_.debug_path + "/residuals.txt");
        for (int i = 0; i < all_residuals.size(); i++) {
            if (i == 0)
                residual_ofs << all_residuals[i];
            else
                residual_ofs << ", " << all_residuals[i];
        }
        std::ofstream psnr_ofs(config_.debug_path + "/psnr.txt");
        for (int i = 0; i < all_psnr.size(); i++) {
            if (i == 0)
                psnr_ofs << all_psnr[i];
            else
                psnr_ofs << ", " << all_psnr[i];
        }
    }
}

} // namespace opt;
