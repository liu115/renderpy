// This is the wrapper class of eth3d-pipeline camera class


#pragma once
#include <mLibCore.h>
#include "camera/camera_models.h"
#include "opengl/opengl.h"
// #include "colmap.h"

namespace opt {
class CameraWrapper {
public:
    CameraWrapper() {}
    CameraWrapper(
        int height,
        int width,
        float fx,
        float fy,
        float cx,
        float cy,
        std::string camera_type,
        const float* distortion_params
    ) {
        if (camera_type == "PINHOLE") {
            type_ = opengl::CameraModelType::CAMERA_PINHOLE;
            camera_.reset(new camera::PinholeCamera(width, height, fx, fy, cx, cy));
        } else if (camera_type == "OPENCV") {
            type_ = opengl::CameraModelType::CAMERA_OPENCV;
            // k1, k2, p1, p2
            camera_.reset(new camera::PolynomialTangentialCamera(width, height, fx, fy, cx, cy, distortion_params[0], distortion_params[1], distortion_params[2], distortion_params[3]));
        } else if (camera_type == "OPENCV_FISHEYE") {
            type_ = opengl::CameraModelType::CAMERA_FISHEYE;
            // k1, k2, k3, k4
            camera_.reset(new camera::FisheyePolynomial4Camera(width, height, fx, fy, cx, cy, distortion_params[0], distortion_params[1], distortion_params[2], distortion_params[3]));
        } else {
            LOG(ERROR) << "ERROR: Unknown camera model " << camera_type;
            return;
        }
    }
    // CameraWrapper(const colmap::COLMAPCamera& camera) {
    //     if (camera.model_name == "PINHOLE") {
    //         type_ = opengl::CameraModelType::CAMERA_PINHOLE;
    //         camera_.reset(new camera::PinholeCamera(camera.width, camera.height, camera.parameters.data()));
    //     } else if (camera.model_name == "OPENCV") {
    //         type_ = opengl::CameraModelType::CAMERA_OPENCV;
    //         camera_.reset(new camera::PolynomialTangentialCamera(camera.width, camera.height, camera.parameters.data()));
    //     } else if (camera.model_name == "OPENCV_FISHEYE") {
    //         type_ = opengl::CameraModelType::CAMERA_FISHEYE;
    //         camera_.reset(new camera::FisheyePolynomial4Camera(camera.width, camera.height, camera.parameters.data()));
    //     } else {
    //         LOG(ERROR) << "ERROR: Unknown camera model " << camera.model_name;
    //         return;
    //     }
    // }
    CameraWrapper(opengl::CameraModelType type, camera::CameraBase* c): type_(type) {
        camera_ = std::shared_ptr<camera::CameraBase>(c);
    }

    CameraWrapper(CameraWrapper&& other, bool deep_copy = true) {
        type_ = other.type_;
        if (deep_copy) {
            // Deep copy by creating a new camera instance
            float params[100];  // allocate a buffer to copy the parameters (e.g., fx, fy, cx, cy, distortion parameters)
            if (type_ == opengl::CameraModelType::CAMERA_PINHOLE) {
                // camera_.reset(new camera::PinholeCamera(camera->width(), camera->height(), camera.parameters.data()));
                camera::PinholeCamera* camera = (camera::PinholeCamera*) other.camera_.get();
                int width = camera->width();
                int height = camera->height();
                camera->GetParameters(params);
                camera_.reset(new camera::PinholeCamera(width, height, params));
            } else if (type_ == opengl::CameraModelType::CAMERA_OPENCV) {
                camera::PolynomialTangentialCamera* camera = (camera::PolynomialTangentialCamera*) other.camera_.get();
                int width = camera->width();
                int height = camera->height();
                camera->GetParameters(params);
                camera_.reset(new camera::PolynomialTangentialCamera(width, height, params));
            } else if (type_ == opengl::CameraModelType::CAMERA_FISHEYE) {
                camera::FisheyePolynomial4Camera* camera = (camera::FisheyePolynomial4Camera*) other.camera_.get();
                int width = camera->width();
                int height = camera->height();
                camera->GetParameters(params);
                camera_.reset(new camera::FisheyePolynomial4Camera(width, height, params));
            }
        } else {
            // Shallow copy by sharing the same camera instance
            // Careful to use shallow copy if you are scaling the camera size (which is a inplace operation)
            camera_ = other.camera_;
        }
    }

    opengl::CameraModelType type() {
        return type_;
    }

    int width() {
        return camera_->width();
    }

    int height() {
        return camera_->height();
    }

    float radius_cutoff_squared() {
        if (type_ == opengl::CameraModelType::CAMERA_FISHEYE) {
            auto fisheye_camera = getFisheyeCamera();
            return fisheye_camera->radius_cutoff_squared();
        } else if (type_ == opengl::CameraModelType::CAMERA_OPENCV) {
            auto opencv_camera = getOpenCVCamera();
            return opencv_camera->radius_cutoff_squared();
        } else {
            return std::numeric_limits<float>::infinity();
        }
    }

    void scale(const float factor) {
        camera_.reset(camera_->ScaledBy(factor));
    }

    opengl::DistortParams getDistortionParams() {
        if (type_ == opengl::CameraModelType::CAMERA_FISHEYE) {
            auto fisheye_camera = getFisheyeCamera();
            float k1 = fisheye_camera->distortion_parameters()[0];
            float k2 = fisheye_camera->distortion_parameters()[1];
            float k3 = fisheye_camera->distortion_parameters()[2];
            float k4 = fisheye_camera->distortion_parameters()[3];
            opengl::DistortParams params(k1, k2, k3, k4);
            return params;
        } else if (type_ == opengl::CameraModelType::CAMERA_OPENCV) {
            auto opencv_camera = getOpenCVCamera();
            float k1 = opencv_camera->distortion_parameters()(0);
            float k2 = opencv_camera->distortion_parameters()(1);
            float p1 = opencv_camera->distortion_parameters()(2);
            float p2 = opencv_camera->distortion_parameters()(3);
            opengl::DistortParams params(k1, k2, 0, 0, 0, 0, p1, p2);
        }
        // if not fisheye or opencv, return all zeros for pinhole
        return opengl::DistortParams();
    }

    ml::mat4f getIntrinsicMatrix() {
        ml::mat4f intrinsic = ml::mat4f::identity();
        intrinsic(0, 0) = camera_->fx();
        intrinsic(1, 1) = camera_->fy();
        intrinsic(0, 2) = camera_->cx();
        intrinsic(1, 2) = camera_->cy();
        return intrinsic;
    }

    std::shared_ptr<camera::FisheyePolynomial4Camera> getFisheyeCamera() {
        return std::dynamic_pointer_cast<camera::FisheyePolynomial4Camera>(camera_);
    }

    std::shared_ptr<camera::PolynomialTangentialCamera> getOpenCVCamera() {
        return std::dynamic_pointer_cast<camera::PolynomialTangentialCamera>(camera_);
    }

    ml::vec3f projectPoint(const ml::vec3f p) {
        // p: 3D point in camera coordinate
        // return: u, v, z in float
        Eigen::Vector2f p_norm(p.x / p.z, p.y / p.z);
        if (type_ == opengl::CameraModelType::CAMERA_PINHOLE) {
            // Do nothing
        }
        else if (type_ == opengl::CameraModelType::CAMERA_FISHEYE) {
            auto fisheye_camera = getFisheyeCamera();
            p_norm = fisheye_camera->Distort(p_norm);

        }
        else if (type_ == opengl::CameraModelType::CAMERA_OPENCV) {
            auto opencv_camera = getOpenCVCamera();
            p_norm = opencv_camera->Distort(p_norm);
        }
        return ml::vec3f(camera_->fx() * p_norm.x() + camera_->cx(), camera_->fy() * p_norm.y() + camera_->cy(), p.z);
    }

    Eigen::Matrix<float, 2, 3> computeJacobianByWorld(ml::vec3f& p) {
        // p: 3D point in camera coordinate
        // return: Jacobian matrix of f_uv(xyz)
        Eigen::Matrix<float, 2, 3> J;
        Eigen::Vector3f pp(p.x, p.y, p.z);
        if (type_ == opengl::CameraModelType::CAMERA_PINHOLE) {
            const float z_inv = 1.0f / p.z;
            J << z_inv, 0, -p.x * z_inv * z_inv,
              0, z_inv, -p.y * z_inv * z_inv;
            Eigen::Matrix2f J_proj;
            J_proj << camera_->fx(), 0,
                      0, camera_->fy();
            J = J_proj * J;
        } else if (type_ == opengl::CameraModelType::CAMERA_FISHEYE) {
            auto fisheye_camera = getFisheyeCamera();
            fisheye_camera->ImageDerivativeByWorld(pp, J);
        } else if (type_ == opengl::CameraModelType::CAMERA_OPENCV) {
            auto opencv_camera = getOpenCVCamera();
            opencv_camera->ImageDerivativeByWorld(pp, J);
        }
        return J;
    }

    std::string getString(bool with_cutoff = false) {
        std::stringstream ss;
        if (type_ == opengl::CameraModelType::CAMERA_FISHEYE) {
            auto fisheye_camera = getFisheyeCamera();
            auto params = fisheye_camera->distortion_parameters();
            ss << "OPENCV_FISHEYE " << camera_->width() << " " << camera_->height() << " " << fisheye_camera->fx() << " " << fisheye_camera->fy() << " " << fisheye_camera->cx() << " " << fisheye_camera->cy();
            ss << " " << params[0] << " " << params[1] << " " << params[2] << " " << params[3];
            if (with_cutoff) {
                ss << " " << fisheye_camera->radius_cutoff_squared();
            }
        } else if (type_ == opengl::CameraModelType::CAMERA_OPENCV) {
            auto opencv_camera = getOpenCVCamera();
            auto params = opencv_camera->distortion_parameters();
            ss << "OPENCV " << camera_->width() << " " << camera_->height() << " " << opencv_camera->fx() << " " << opencv_camera->fy() << " " << opencv_camera->cx() << " " << opencv_camera->cy();
            ss << " " << params(0) << " " << params(1) << " " << params(2) << " " << params(3);
            if (with_cutoff) {
                ss << " " << opencv_camera->radius_cutoff_squared();
            }
        } else if (type_ == opengl::CameraModelType::CAMERA_PINHOLE) {
            ss << "PINHOLE " << camera_->width() << " " << camera_->height() << " " << camera_->fx() << " " << camera_->fy() << " " << camera_->cx() << " " << camera_->cy();
        }
        return ss.str();
    }

private:
    opengl::CameraModelType type_;
    std::shared_ptr<camera::CameraBase> camera_;
};
} // namespace camera
