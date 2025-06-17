#include "FeatureComputer.hpp"
#include "AlphaNodeFeatures.hpp"
#include "../AlphaTree.hpp"

template <class Pixel>
bool FeatureComputer<Pixel>::pixelBelongsToNode(
    const AlphaTree<Pixel>& tree,
    ImgIdx pixelIndex,
    ImgIdx nodeIdx)
{
    ImgIdx index = tree._parentAry ? tree._parentAry[nodeIdx] : nodeIdx;
    while (index != ROOTIDX && index != nodeIdx) {
        index = tree._parentAry[index];
    }
    return (index != ROOTIDX);
}

template <class Pixel>
typename FeatureComputer<Pixel>::Moments
FeatureComputer<Pixel>::computeMoments(const AlphaTree<Pixel>& tree, ImgIdx nodeIdx) {
    Moments moments;
    for (int y = 0; y < tree._height; ++y) {
        for (int x = 0; x < tree._width; ++x) {
            ImgIdx pixelIndex = y * tree._width + x;
            if (pixelBelongsToNode(tree, pixelIndex, nodeIdx)) {
                moments.sumX += x;
                moments.sumY += y;
                moments.sumX2 += x * x;
                moments.sumY2 += y * y;
            }
        }
    }
    return moments;
}

template <class Pixel>
void FeatureComputer<Pixel>::computeElongation(
    const Moments& moments,
    double area,
    AlphaNodeFeatures& features)
{
    double inertia = moments.sumX2 + moments.sumY2 -
              (moments.sumX * moments.sumX +
               moments.sumY * moments.sumY) / area
              + area / 6.0;
    features.elongation = inertia * 2.0 * M_PI / (area * area);
}

template <class Pixel>
AlphaNodeFeatures
FeatureComputer<Pixel>::computeFeatures(const AlphaTree<Pixel>& tree, ImgIdx nodeIdx) {
    AlphaNodeFeatures features;
    const AlphaNode<Pixel>& node = tree._node[nodeIdx];

    features.area = node.area;
    features.meanIntensity = node.sumPix / node.area;

    if (node.area <= 1) {
        features.elongation = 2.0 * M_PI;
        return features;
    }
//
//    Moments moments = computeMoments(tree, nodeIdx);
//    computeElongation(moments, node.area, features);
    return features;
}

template class FeatureComputer<uint8_t>;
template class FeatureComputer<uint16_t>;
template class FeatureComputer<uint32_t>;
template class FeatureComputer<uint64_t>;


