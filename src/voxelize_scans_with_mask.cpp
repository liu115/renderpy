#include <iostream>
#include <mLibCore.h>
#include "E57SimpleReader.h"
#include"cnpy.h"
#include <omp.h>


#define VOXEL_SIZE 0.02
#define NEAR_CLIP 0.01
#define DIST_THRESH (VOXEL_SIZE * 1.0)
#define FAR_CLIP 20.0
#define NUM_RAYS_PER_SCAN 1000000
#define CHUNK_SIZE 250
#define CHUNK_MIN_POINTS 5000


ml::mat4f read_pose_txt(std::string filename) {
    // The pose file is a text files with four lines, each lines contains 4 numbers separated by comma
    // Read the file and parse the numbers, make it a 4x4 matrix
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file " + filename);
    }
    ml::mat4f pose;
    for (int i = 0; i < 4; i++) {
        std::string line;
        std::getline(file, line);
        std::stringstream ss(line);
        for (int j = 0; j < 4; j++) {
            float val;
            char comma;
            ss >> val >> comma;
            pose(i, j) = val;
        }
    }
    return pose;
}


ml::mat4f get_pose(e57::Data3D& scanHeader) {
    ml::vec3f trans(scanHeader.pose.translation.x, scanHeader.pose.translation.y, scanHeader.pose.translation.z);
    ml::quatf rot(scanHeader.pose.rotation.w, scanHeader.pose.rotation.x, scanHeader.pose.rotation.y, scanHeader.pose.rotation.z);
    ml::mat4f pose = rot.matrix4x4();
    // Set translation
    pose.matrix2[0][3] = trans.x;
    pose.matrix2[1][3] = trans.y;
    pose.matrix2[2][3] = trans.z;
    return std::move(pose);
}


ml::vec3i get_voxel_indices(ml::vec3f xyz, ml::vec3f min_bound) {
    ml::vec3i voxel;
    voxel.x = (int)floor((xyz.x - min_bound.x) / VOXEL_SIZE);
    voxel.y = (int)floor((xyz.y - min_bound.y) / VOXEL_SIZE);
    voxel.z = (int)floor((xyz.z - min_bound.z) / VOXEL_SIZE);
    return voxel;
}


// A function that random samples N indices from 0 to M
void random_sample(std::vector<int>& indices, int N, int M) {
    std::vector<int> all_indices(M);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    std::random_shuffle(all_indices.begin(), all_indices.end());
    indices.resize(N);
    for (int i = 0; i < N; i++) {
        indices[i] = all_indices[i];
    }
}



bool voxel_in_grid(ml::vec3i voxel, ml::vec3i grid_dim) {
    return voxel.x >= 0 && voxel.x < grid_dim.x &&
           voxel.y >= 0 && voxel.y < grid_dim.y &&
           voxel.z >= 0 && voxel.z < grid_dim.z;
}


int voxel_to_points(std::vector<ml::vec3f>& points, ml::Grid3f& voxel_grid, ml::vec3i grid_dim, ml::vec3f min_bound) {
    // Return the number of points
    int num_points = 0;
    for (int x = 0; x < grid_dim.x; x++) {
        for (int y = 0; y < grid_dim.y; y++) {
            for (int z = 0; z < grid_dim.z; z++) {
                if (voxel_grid(x, y, z) > 0.5f) {
                    points.push_back(ml::vec3f(x * VOXEL_SIZE + min_bound.x, y * VOXEL_SIZE + min_bound.y, z * VOXEL_SIZE + min_bound.z));
                    num_points++;
                }
            }
        }
    }
    return num_points;
}

void save_dense_npy(
    ml::Grid3f& voxel_grid_input,
    ml::Grid3f& voxel_grid_occ,
    ml::Grid3f& voxel_grid_known,
    ml::vec3i grid_dim,
    std::string filename
) {
    // Boolean np array format (0 or 1)
    // py::array_t<bool> np_array({grid_dim.x, grid_dim.y, grid_dim.z});
    // auto np_array_ptr = np_array.mutable_unchecked<3>();
    // for (int x = 0; x < grid_dim.x; x++) {
    //     for (int y = 0; y < grid_dim.y; y++) {
    //         for (int z = 0; z < grid_dim.z; z++) {
    //             np_array_ptr(x, y, z) = voxel_grid(x, y, z) > 0.5f;
    //         }
    //     }
    // }
    std::vector<char> grid(grid_dim.x * grid_dim.y * grid_dim.z);
    for (int x = 0; x < grid_dim.x; x++) {
        for (int y = 0; y < grid_dim.y; y++) {
            for (int z = 0; z < grid_dim.z; z++) {
                char val = voxel_grid_input(x, y, z) > 0.5f ? 1 : 0;
                val = (val << 1) + (voxel_grid_occ(x, y, z) > 0.5f ? 1 : 0);
                val = (val << 1) + (voxel_grid_known(x, y, z) > 0.5f ? 1 : 0);
                grid[x * grid_dim.y * grid_dim.z + y * grid_dim.z + z] = val;
            }
        }
    }
    const char* data_ptr = grid.data();
    const unsigned int size_x = static_cast<unsigned int>(grid_dim.x);
    const unsigned int size_y = static_cast<unsigned int>(grid_dim.y);
    const unsigned int size_z = static_cast<unsigned int>(grid_dim.z);
    cnpy::npy_save(filename, data_ptr, {size_x, size_y, size_z}, "w");
}


int ray_voxel_interesection(
    ml::Grid3f& occ_grid,
    ml::Grid3f& known_grid,
    ml::vec3f ray_origin,
    ml::vec3f ray_target,
    ml::vec3i grid_dim,
    ml::vec3f min_bound
) {
    // Compute the ray direction
    ml::vec3f ray_dir = ray_target - ray_origin;
    float ray_length = ray_dir.length();
    if (ray_length < NEAR_CLIP) {
        return 0;
    }
    ray_dir = ray_dir / ray_length;
    ml::vec3f inv_ray_dir = ml::vec3f(1.0f / ray_dir.x, 1.0f / ray_dir.y, 1.0f / ray_dir.z);
    ml::vec3i ray_origin_voxel = ml::vec3i(
        (int)floor((ray_origin.x - min_bound.x) / VOXEL_SIZE),
        (int)floor((ray_origin.y - min_bound.y) / VOXEL_SIZE),
        (int)floor((ray_origin.z - min_bound.z) / VOXEL_SIZE));

    // Compute the step size in the voxel grid
    const ml::vec3i step(
        ray_dir.x > 0 ? 1 : -1,
        ray_dir.y > 0 ? 1 : -1,
        ray_dir.z > 0 ? 1 : -1);

    const ml::vec3f t_delta(
        std::abs(VOXEL_SIZE * inv_ray_dir.x),
        std::abs(VOXEL_SIZE * inv_ray_dir.y),
        std::abs(VOXEL_SIZE * inv_ray_dir.z));

    ml::vec3f t_max(
        (ray_origin_voxel.x + (ray_dir.x > 0 ? 1 : 0)) * VOXEL_SIZE + min_bound.x - ray_origin.x,
        (ray_origin_voxel.y + (ray_dir.y > 0 ? 1 : 0)) * VOXEL_SIZE + min_bound.y - ray_origin.y,
        (ray_origin_voxel.z + (ray_dir.z > 0 ? 1 : 0)) * VOXEL_SIZE + min_bound.z - ray_origin.z);
    t_max = ml::vec3f(t_max.x * inv_ray_dir.x, t_max.y * inv_ray_dir.y, t_max.z * inv_ray_dir.z);

    // std::cout << "ray_dir: " << ray_dir << std::endl;
    // std::cout << "inv_ray_dir: " << inv_ray_dir << std::endl;
    // std::cout << "ray_origin_voxel: " << ray_origin_voxel << std::endl;
    // std::cout << "step: " << step << std::endl;
    // std::cout << "t_delta: " << t_delta << std::endl;
    // std::cout << "t_max: " << t_max << std::endl;

    // Start the DDA algorithm
    int num_voxels = 0;
    float t = 0.0f;
    ml::vec3i current_voxel = ray_origin_voxel;
    while (t < ray_length + DIST_THRESH && t < FAR_CLIP) {
        // std::cout << "t: " << t << " voxel: " << current_voxel << std::endl;
        if (voxel_in_grid(current_voxel, grid_dim)) {
            known_grid(current_voxel.x, current_voxel.y, current_voxel.z) = 1.0f;
            if (t >= ray_length && t <= ray_length + DIST_THRESH) {
                occ_grid(current_voxel.x, current_voxel.y, current_voxel.z) = 1.0f;
            }
            num_voxels++;
        }

        if (t_max.x < t_max.y && t_max.x < t_max.z) {
            current_voxel.x += step.x;
            t = t_max.x;
            t_max.x += t_delta.x;
        } else if (t_max.y < t_max.z) {
            current_voxel.y += step.y;
            t = t_max.y;
            t_max.y += t_delta.y;
        } else {
            current_voxel.z += step.z;
            t = t_max.z;
            t_max.z += t_delta.z;
        }
    }
    return num_voxels;
}


int main(int argc, char **argv) {

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " scene_id" << std::endl;
        return 1;
    }

    // TODO: Read data_root from arguments
    std::string data_root = "/menegroth/scannetpp/data/";

    // Read scene_id (string) from arguments
    std::string scene_id = argv[1];
    std::cout << "scene_id: " << scene_id << std::endl;

    std::string out_dir = argv[2];
    std::cout << "out_dir: " << out_dir << std::endl;

    // Print omp_get_thread_num()
    #pragma omp parallel
    {
        #pragma omp master
        {
            std::cout << "Number of threads: " << omp_get_num_threads() << std::endl;
        }
    }

    std::string e57_file = data_root + "/" + scene_id + "/scans/original/" + scene_id + ".e57";
    // std::string mesh_file = data_root + "/" + scene_id + "/scans/1mm/merged.ply";
    std::string mesh_file = data_root + "/" + scene_id + "/scans/1mm/chunks_depth9/mesh/simplified_0.0156_mesh_aligned_colorfixed.ply";
    std::string transform_file = data_root + "/" + scene_id + "/scans/1mm/transform.txt";
    std::string colmap_ply_file = "/cluster/ramdal/yliu/scannetpp-chunk/colmap_point_filtered/" + scene_id + ".ply";

    // Read the transformation matrix
    ml::mat4f transform = read_pose_txt(transform_file);

    // Read the mesh file
    ml::MeshDataf mesh;
    ml::MeshIOf::loadFromPLY(mesh_file, mesh);
    // mesh.applyTransform(transform);
    ml::vec3f min_bound = mesh.computeBoundingBox().getMin();
    ml::vec3f max_bound = mesh.computeBoundingBox().getMax();
    // auto mesh_center = (min_bound + max_bound) / 2.0f;
    auto mesh_size = max_bound - min_bound;

    std::cout << "min_bound: " << min_bound << std::endl;
    std::cout << "max_bound: " << max_bound << std::endl;

    min_bound -= mesh_size * 0.25f;
    max_bound += mesh_size * 0.25f;

    std::cout << "min_bound * 1.5: " << min_bound << std::endl;
    std::cout << "max_bound * 1.5: " << max_bound << std::endl;

    // Define the voxel grid based on the mesh bounds
    const int grid_dim_x = (int) (std::ceil((max_bound.x - min_bound.x) / VOXEL_SIZE));
    const int grid_dim_y = (int) (std::ceil((max_bound.y - min_bound.y) / VOXEL_SIZE));
    const int grid_dim_z = (int) (std::ceil((max_bound.z - min_bound.z) / VOXEL_SIZE));
    const ml::vec3i grid_dim(grid_dim_x, grid_dim_y, grid_dim_z);

    std::cout << "grid_dim_x: " << grid_dim_x << std::endl;
    std::cout << "grid_dim_y: " << grid_dim_y << std::endl;
    std::cout << "grid_dim_z: " << grid_dim_z << std::endl;

    // Define a 3D voxel grid
    ml::Grid3f voxel_grid_input(grid_dim_x, grid_dim_y, grid_dim_z, 0.0f);
    ml::Grid3f voxel_grid_occ(grid_dim_x, grid_dim_y, grid_dim_z, 0.0f);
    ml::Grid3f voxel_grid_known(grid_dim_x, grid_dim_y, grid_dim_z, 0.0f);

    // Read ply point cloud from colmap_ply_file
    ml::PointCloudf pcd_colmap;
    ml::PointCloudIOf::loadFromFile(colmap_ply_file, pcd_colmap);
    for (auto &point : pcd_colmap.m_points) {
        const ml::vec3i target_voxel = get_voxel_indices(point, min_bound);
        if (voxel_in_grid(target_voxel, grid_dim)) {
            voxel_grid_input(target_voxel.x, target_voxel.y, target_voxel.z) = 1.0f;
        }
    }

    // Read the e57 file
    e57::Reader eReader(e57_file);
    e57::Data3D	scanHeader;

    const int data3DCount = eReader.GetData3DCount();
    std::cout << "Data3DCount: " << data3DCount << std::endl;

    std::vector<ml::vec3f> pcd;
    for (int scanIndex = 0; scanIndex < data3DCount; scanIndex++) {
        eReader.ReadData3D(scanIndex, scanHeader);
        int64_t nColumn = 0;		//Number of Columns in a structure scan (from "indexBounds" if structure data)
        int64_t nRow = 0;			//Number of Rows in a structure scan
        int64_t nPointsSize = 0;	//Number of points
        int64_t nGroupsSize = 0;	//Number of groups (from "groupingByLine" if present)
        int64_t nCountsSize = 0;	//Number of points per group
        bool bColumnIndex = false;	//Number of points per group
        eReader.GetData3DSizes(scanIndex, nRow, nColumn, nPointsSize, nGroupsSize, nCountsSize, bColumnIndex);

        ml::mat4f pose = get_pose(scanHeader);
        pose = transform * pose;    // Apply the transformation matrix
        // Read point datas
        e57::Data3DPointsDouble data3DPoints(scanHeader);
        e57::CompressedVectorReader reader = eReader.SetUpData3DPointsData(scanIndex, nPointsSize, data3DPoints);
        const uint64_t cNumRead = reader.read();
        std::cout << "cNumRead: " << cNumRead << "from scan " << scanIndex << std::endl;

        // Random sample NUM_RAYS_PER_SCAN from nPointsSize
        std::vector<int> elems;
        random_sample(elems, NUM_RAYS_PER_SCAN, nPointsSize);

        ml::vec3f camera_center = pose.getTranslation();
        #pragma omp parallel for
        for (auto pointIdx : elems) {
            ml::vec3f pointXYZ(
                data3DPoints.cartesianX[pointIdx],
                data3DPoints.cartesianY[pointIdx],
                data3DPoints.cartesianZ[pointIdx]);
            pointXYZ = pose * pointXYZ;

            const ml::vec3i target_voxel = get_voxel_indices(pointXYZ, min_bound);

            // if (voxel_in_grid(target_voxel, grid_dim)) {
            //     voxel_grid_occ(target_voxel.x, target_voxel.y, target_voxel.z) = 1.0f;
            // }

            ray_voxel_interesection(
                voxel_grid_occ,
                voxel_grid_known,
                camera_center,
                pointXYZ,
                grid_dim,
                min_bound
            );
        }
        // break;      // debug
    }

    // Convert the voxel grid to point cloud and save it
    std::vector<ml::vec3f> points_occ, points_known;
    int num_points_occ = voxel_to_points(points_occ, voxel_grid_occ, grid_dim, min_bound);
    int num_points_known = voxel_to_points(points_known, voxel_grid_known, grid_dim, min_bound);
    std::cout << "num_points_occ: " << num_points_occ << std::endl;
    ml::PointCloudf pcd_occ(points_occ);
    ml::PointCloudIOf::saveToFile("voxel_grid_occ1.ply", pcd_occ);
    std::cout << "num_points_known: " << num_points_known << std::endl;
    ml::PointCloudf pcd_known(points_known);
    ml::PointCloudIOf::saveToFile("voxel_grid_known1.ply", pcd_known);

    // save_dense_npy(voxel_grid_occ, voxel_grid_known, grid_dim, "voxel_grid.npy");

    int counter = 0;
    for (int x = 0; x < grid_dim.x + CHUNK_SIZE - 1; x += CHUNK_SIZE / 2) {
        for (int y = 0; y < grid_dim.y + CHUNK_SIZE - 1; y += CHUNK_SIZE / 2) {
            ml::Grid3f voxel_grid_input_chunk(CHUNK_SIZE, CHUNK_SIZE, grid_dim.z, 0.0f);
            ml::Grid3f voxel_grid_occ_chunk(CHUNK_SIZE, CHUNK_SIZE, grid_dim.z, 0.0f);
            ml::Grid3f voxel_grid_known_chunk(CHUNK_SIZE, CHUNK_SIZE, grid_dim.z, 0.0f);

            int num_points_occ = 0;
            for (int i = 0; i < CHUNK_SIZE; i++) {
                for (int j = 0; j < CHUNK_SIZE; j++) {
                    for (int z = 0; z < grid_dim.z; z++) {
                        if (x + i < grid_dim.x && y + j < grid_dim.y) {
                            voxel_grid_input_chunk(i, j, z) = voxel_grid_input(x + i, y + j, z);
                            voxel_grid_occ_chunk(i, j, z) = voxel_grid_occ(x + i, y + j, z);
                            voxel_grid_known_chunk(i, j, z) = voxel_grid_known(x + i, y + j, z);
                            num_points_occ += voxel_grid_occ(x + i, y + j, z) > 0.5f ? 1 : 0;
                        }
                    }
                }
            }

            if (num_points_occ < CHUNK_MIN_POINTS) {
                continue;
            }
            save_dense_npy(
                voxel_grid_input_chunk,
                voxel_grid_occ_chunk,
                voxel_grid_known_chunk,
                ml::vec3i(CHUNK_SIZE, CHUNK_SIZE, grid_dim.z),
                out_dir + "/" + scene_id + "_" + std::to_string(counter) + ".npy");

            counter += 1;
        }
    }
    // save_dense_npy(voxel_grid_occ, voxel_grid_known, grid_dim, out_dir + "/" + scene_id + ".npy");
}

// TODO: Handle the input (colmap point cloud) and output (npy) directories