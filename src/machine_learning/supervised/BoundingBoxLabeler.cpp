#include "BoundingBoxLabeler.hpp"

std::vector<double> BoundingBoxLabeler::getBoundingBoxAreas() const {
    std::vector<double> areas;
    for (const auto& box : boundingBoxes_) {
        double pixelWidth = box.width * imageWidth_;
        double pixelHeight = box.height * imageHeight_;
        double area = pixelWidth * pixelHeight;
        areas.push_back(area);
    }
    return areas;
}

std::pair<double, double> BoundingBoxLabeler::getAreaRange() const {
    auto areas = getBoundingBoxAreas();
    if (areas.empty()) return {0.0, 0.0};

    auto minMax = std::minmax_element(areas.begin(), areas.end());
    return {*minMax.first, *minMax.second};
}


void BoundingBoxLabeler::printBoundingBoxInfo() const {
    std::cout << "\n=== Bounding Box Information ===" << std::endl;
    std::cout << "Image dimensions: " << imageWidth_ << "x" << imageHeight_ << std::endl;
    std::cout << "Total bounding boxes: " << boundingBoxes_.size() << std::endl;

    if (boundingBoxes_.empty()) {
        std::cout << "No bounding boxes loaded!" << std::endl;
        return;
    }

    auto areas = getBoundingBoxAreas();
    auto [minArea, maxArea] = getAreaRange();

    std::cout << "Bounding box areas:" << std::endl;
    std::cout << "  Min area: " << minArea << " pixels" << std::endl;
    std::cout << "  Max area: " << maxArea << " pixels" << std::endl;

    // Count by class
    std::map<int, int> classCounts;
    std::map<int, std::vector<double>> classAreas;

    for (size_t i = 0; i < boundingBoxes_.size(); i++) {
        const auto& box = boundingBoxes_[i];
        classCounts[box.classId]++;
        classAreas[box.classId].push_back(areas[i]);

        double pixelWidth = box.width * imageWidth_;
        double pixelHeight = box.height * imageHeight_;

        std::cout << "  Box " << i << " (Class " << box.classId << "): "
                  << "area=" << areas[i] << ", "
                  << "size=" << pixelWidth << "x" << pixelHeight << std::endl;
    }

    std::cout << "\nClass statistics:" << std::endl;
    for (const auto& pair : classCounts) {
        int classId = pair.first;
        const auto& classAreaVec = classAreas[classId];
        double avgArea = std::accumulate(classAreaVec.begin(), classAreaVec.end(), 0.0) / classAreaVec.size();

        std::cout << "  Class " << classId << ": " << pair.second << " boxes, "
                  << "avg area=" << avgArea << std::endl;
    }
    std::cout << "================================\n" << std::endl;
}

std::string BoundingBoxLabeler::labelNode(const AlphaTree<uint8_t>& tree, int nodeIdx) {
    return assignLabel(tree, nodeIdx, boundingBoxes_);
}

bool BoundingBoxLabeler::isNodeInAreaRange(double nodeArea) const {
    auto [minArea, maxArea] = getAreaRange();

    // Add some tolerance (e.g., 50% smaller to 200% larger than bounding box range)
    double minThreshold = minArea * 0.5;
    double maxThreshold = maxArea * 2.0;

    return (nodeArea >= minThreshold && nodeArea <= maxThreshold);
}


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

double BoundingBoxLabeler::overlap(const AlphaNode<uint8_t>& node, const BoundingBox& box) {
    double nodeX1 = node.minX;
    double nodeY1 = node.minY;
    double nodeX2 = node.maxX;
    double nodeY2 = node.maxY;

    bool xOverlap = (nodeX1 <= box.x_max) && (nodeX2 >= box.x_min);
    bool yOverlap = (nodeY1 <= box.y_max) && (nodeY2 >= box.y_min);

    if (!xOverlap || !yOverlap) return 0.0f;  // No overlap

    double overlapX1 = std::max(nodeX1, (double)box.x_min);
    double overlapY1 = std::max(nodeY1, (double)box.y_min);
    double overlapX2 = std::min(nodeX2, (double)box.x_max);
    double overlapY2 = std::min(nodeY2, (double)box.y_max);

    double overlapArea = std::max(0.0, (overlapX2 - overlapX1)) * std::max(0.0, (overlapY2 - overlapY1));
    return overlapArea;
}

double BoundingBoxLabeler::IoU(const AlphaNode<uint8_t> &node, const BoundingBox &box) {
    double overlapArea = overlap(node, box);
    if (overlapArea == 0.0) return 0.0;

    double nodeArea = node.area;
    double boxArea = (box.x_max - box.x_min) * (box.y_max - box.y_min);

    return overlapArea / (nodeArea + boxArea - overlapArea);
}

std::string BoundingBoxLabeler::assignLabel(const AlphaTree<uint8_t>& tree, int nodeIdx,
                                            const std::vector<BoundingBox>& boxes) {
    const auto& node = tree._node[nodeIdx];

    if (node.area == 0) return "none";  // skip empty nodes

    double centroidX = static_cast<double>(node.sumX) / node.area;
    double centroidY = static_cast<double>(node.sumY) / node.area;

    for (const auto& box : boxes) {
        // Check 1: Centroid inside bounding box (fast & effective)
        if (isInsideBoundingBox(centroidX, centroidY, box)) {
            return (box.classId == 0) ? "healthy" : "diseased";
        }

//        //Check 2: bounding box overlap
        double overlapArea = overlap(node, box);
        if (overlapArea / node.area > 0.5) {
            return (box.classId == 0) ? "healthy" : "diseased";
        }

        //Check 3: IoU
//        double iou = IoU(node, box);
//        if (iou > 0.2) {
//            return (box.classId == 0) ? "healthy" : "diseased";
//        }
    }

    return "none";  // No matching bounding box found
}



std::vector<BoundingBox> BoundingBoxLabeler::getBoundingBoxes() {
    return boundingBoxes_;
}
