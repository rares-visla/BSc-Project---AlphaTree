#include "BoundingBoxLabeler.hpp"

std::vector<BoundingBox> BoundingBoxLabeler::loadBoundingBoxes(const std::string& filename, int imageWidth, int imageHeight) {
    std::vector<BoundingBox> boxes;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open bounding box file: " << filename << std::endl;
        return boxes;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        BoundingBox box;

        if (iss >> box.classId >> box.x_center >> box.y_center >> box.width >> box.height) {
            // Convert normalized coordinates to absolute pixel coordinates
            box.x_min = static_cast<int>((box.x_center - box.width/2.0) * imageWidth);
            box.y_min = static_cast<int>((box.y_center - box.height/2.0) * imageHeight);
            box.x_max = static_cast<int>((box.x_center + box.width/2.0) * imageWidth);
            box.y_max = static_cast<int>((box.y_center + box.height/2.0) * imageHeight);

            // Clamp to image boundaries
            box.x_min = std::max(0, std::min(box.x_min, imageWidth - 1));
            box.y_min = std::max(0, std::min(box.y_min, imageHeight - 1));
            box.x_max = std::max(0, std::min(box.x_max, imageWidth - 1));
            box.y_max = std::max(0, std::min(box.y_max, imageHeight - 1));

            boxes.push_back(box);
        }
    }

    file.close();
    std::cout << "Loaded " << boxes.size() << " bounding boxes" << std::endl;
    return boxes;
}

bool BoundingBoxLabeler::isInsideBoundingBox(double x, double y, const BoundingBox& box) {
    return x >= box.x_min && x <= box.x_max && y >= box.y_min && y <= box.y_max;
}

std::string BoundingBoxLabeler::assignLabel(const AlphaTree<uint8_t>& tree, int nodeIdx,
                                            const std::vector<BoundingBox>& boxes) {
    double centroidX = static_cast<double>(tree._node[nodeIdx].sumX) / tree._node[nodeIdx].area;
    double centroidY = static_cast<double>(tree._node[nodeIdx].sumY) / tree._node[nodeIdx].area;

    // Check each bounding box
    for (const auto& box : boxes) {
        if (isInsideBoundingBox(centroidX, centroidY, box)) {
            return (box.classId == 0) ? "healthy" : "diseased";
        }
    }

    // If not inside any bounding box, label as background/dirt
    return "none";
}
