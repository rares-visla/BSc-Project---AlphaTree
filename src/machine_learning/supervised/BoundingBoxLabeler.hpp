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
#include <algorithm>
#include <map>
#include <numeric>
#include <math.h>
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

//Potential problem: what if boxes overlap
//Now it chooses the first box that it finds
class BoundingBoxLabeler {
  private:
    std::vector<BoundingBox> boundingBoxes_;
    double imageWidth_;
    double imageHeight_;

  public:
    BoundingBoxLabeler(const std::string& filename, int imageWidth, int imageHeight) {
        imageWidth_ = imageWidth;
        imageHeight_ = imageHeight;
        boundingBoxes_ = loadBoundingBoxes(filename, imageWidth, imageHeight);
    }

    //Bounding box stats
    std::vector<double> getBoundingBoxAreas() const;
    std::pair<double, double> getAreaRange() const;
    void printBoundingBoxInfo() const;

    std::string labelNode(const AlphaTree<uint8_t>& tree, int nodeIdx);

    bool isNodeInAreaRange(double nodeArea) const;

    static std::vector<BoundingBox> loadBoundingBoxes(const std::string& filename, int imageWidth, int imageHeight);
    static bool isInsideBoundingBox(double x, double y, const BoundingBox& box);
    static std::string assignLabel(const AlphaTree<uint8_t>& tree, int nodeIdx, const std::vector<BoundingBox>& boxes);

    std::vector<BoundingBox> getBoundingBoxes();
};



#endif // BSC_PROJECT_BOUNDINGBOX_HPP
