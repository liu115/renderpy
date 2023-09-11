#pragma once
#include <vector>
#include <mLibCore.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
static const int MAX_LABELS = 1500;
static const int MAX_LABELS_PER_VERTEX = 3;


class SemanticMesh {
public:
    SemanticMesh() {}
    SemanticMesh(
        const std::string& mesh_path,
        const std::string& segment_path,
        const std::string& anno_path,
        const std::string& labels_path
    ) {
        readLabelsFile(labels_path);

        // ml::MeshDataf meshdata;
        meshdata = new ml::MeshDataf();
        ml::MeshIOf::loadFromPLY(mesh_path, *meshdata);
        std::cout << *meshdata << std::endl;

        std::cout << "Reading segment file: " << segment_path << std::endl;
        std::ifstream f(segment_path);
        json segment_data = json::parse(f);
        std::cout << segment_data["sceneId"] << std::endl;
        // std::cout << segment_data["segIndices"].size() << std::endl;

        // The segIndices should have the same size as the number of mesh vertices
        assert (meshdata->m_Vertices.size() == segment_data["segIndices"].size());
        for (int i = 0; i < segment_data["segIndices"].size(); i++) {
            if (seg2vert.find(segment_data["segIndices"][i]) == seg2vert.end()) {
                seg2vert[segment_data["segIndices"][i]] = std::vector<int>();
            }
            seg2vert[segment_data["segIndices"][i]].push_back(i);
            // seg2vert.insert(std::make_pair(segment_data["segIndices"][i], i));
        }

        // Read the annotation json file
        std::cout << "Reading annotation file: " << anno_path << std::endl;
        f = std::ifstream(anno_path);
        json anno_data = json::parse(f);
        auto groups = anno_data["segGroups"];
        std::cout << "Number of groups: " << groups.size() << std::endl;
        // Parse retrieve annotation on each vertices
        v_labels.resize(meshdata->m_Vertices.size());
        for (auto instance: groups) {
            std::string label_name = instance["label"];
            // std::cerr << label_name << std::endl;
            if (name2label.find(label_name) == name2label.end()) {
                std::cerr << "Label not found: " << label_name << std::endl;
                continue;
            }

            int label_id = name2label[label_name];
            // std::cerr << " id: " << label_id << std::endl;

            // std::cerr << instance["segments"][0] << std::endl;
            int num_verts = 0;
            for (int seg_id: instance["segments"]) {
                // std::cerr << seg_id << std::endl;
                for (int vert_id: seg2vert[seg_id]) {
                    if (v_labels[vert_id].size() >= MAX_LABELS_PER_VERTEX) {
                        // The vertex has too many labels
                        continue;
                    }
                    v_labels[vert_id].push_back(label_id);
                    // assert(v_labels[vert_id].size() <= MAX_LABELS_PER_VERTEX);
                }
                num_verts += seg2vert[seg_id].size();
            }
            // std::cout << " nsegs: " << instance["segments"].size();
            // std::cout << " nverts: " << num_verts << std::endl;
        }
    }

    void getColors(std::vector<ml::vec4f> &colors) {
        colors.resize(v_labels.size());
        for (int i = 0; i < v_labels.size(); i++) {
            assert(v_labels[i].size() <= MAX_LABELS_PER_VERTEX);

            if (v_labels[i].size() == 0) {
                colors[i] = ml::vec4f(0.0f, 0.0f, 0.0f, 1.0f);
            } else {
                int sum = 0;
                int base = 1;
                for (int label_id: v_labels[i]) {
                    sum += label_id * base;
                    base *= MAX_LABELS;
                }
                float r = (sum % 256) / 255.0f;
                float g = ((sum / 256) % 256) / 255.0f;
                float b = ((sum / 256 / 256) % 256) / 255.0f;
                colors[i] = ml::vec4f(r, g, b, 1.0f);
            }
        }
    }

    void getSemantics(std::vector<int> &semantics) {
        semantics.resize(v_labels.size());
        for (int i = 0; i < v_labels.size(); i++) {
            if (v_labels[i].size() == 0)
                semantics[i] = 0;
            else {
                // Simply select the first label
                semantics[i] = v_labels[i][0];
            }
        }
    }

    // void getSemantics(std::vector<int> &semantics) {
    //     semantics.resize(v_labels.size());
    //     for (int i = 0; i < v_labels.size(); i++) {
    //         assert(v_labels[i].size() <= MAX_LABELS_PER_VERTEX);
    //         if (v_labels[i].size() == 0) {
    //             semantics[i] = -1;
    //         } else {
    //             int sum = 0;
    //             int base = 1;
    //             for (int label_id: v_labels[i]) {
    //                 sum += label_id * base;
    //                 base *= MAX_LABELS;
    //             }
    //             semantics[i] = sum;
    //         }
    //     }
    // }

    void getVisColors(std::vector<ml::vec4f> &colors) {
        colors.resize(v_labels.size());
        for (int i = 0; i < v_labels.size(); i++) {
            assert(v_labels[i].size() <= MAX_LABELS_PER_VERTEX);

            if (v_labels[i].size() == 0) {
                colors[i] = ml::vec4f(0.0f, 0.0f, 0.0f, 1.0f);
            } else {
                int sum = 0;
                int base = 1;
                for (int label_id: v_labels[i]) {
                    sum += label_id * base;
                    base *= MAX_LABELS;
                }
                sum = sum * 100000 + 128901;
                colors[i] = getSemanticColor(sum);
            }
        }
    }

    ml::vec4f getSemanticColor(int label_id) {
        float r = (label_id % 10007 % 256) / 255.0f;
        float g = (label_id % 10037 % 256) / 255.0f;
        float b = (label_id % 3397 % 256) / 255.0f;
        return ml::vec4f(r, g, b, 1.0f);
    }

    ml::MeshDataf* getMeshData() {
        return meshdata;
    }

private:
    void readLabelsFile(const std::string& labels_path) {
        std::ifstream f(labels_path);
        std::string line;
        // 0 is for background (ignore)
        // Start from 1
        int cnt = 1;
        while (std::getline(f, line)) {
            name2label[line] = cnt;
            label2name[cnt] = line;
            cnt++;
        }
        assert (cnt < MAX_LABELS);
    }

    std::vector<std::vector<int>> v_labels;
    std::map<int, std::vector<int>> seg2vert;
    std::map<int, std::string> label2name;
    std::map<std::string, int> name2label;
    ml::MeshDataf *meshdata;
};