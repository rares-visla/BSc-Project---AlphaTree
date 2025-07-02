//
// Created by Rares on 02/07/2025.
//

#ifndef BSC_PROJECT_BOUNDINGBOX_HPP
#define BSC_PROJECT_BOUNDINGBOX_HPP

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include "../../alpha_tree/AlphaTree.hpp"

struct BoundingBox {
    int classId;
    double x_center;
    double y_center;
    double width;
    double height;

    //absolute coordinates
    int x_min, y_min, x_max, y_max;
};

class BoundingBoxLabeler {
  public:
    static std::vector<BoundingBox> loadBoundingBoxes(const std::string& filename, int imageWidth, int imageHeight);
    static bool isInsideBoundingBox(double x, double y, const BoundingBox& box);
    static std::string assignLabel(const AlphaTree<uint8_t>& tree, int nodeIdx, const std::vector<BoundingBox>& boxes);

};



#endif // BSC_PROJECT_BOUNDINGBOX_HPP
