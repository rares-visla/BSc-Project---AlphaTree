//#ifndef BSC_PROJECT_FEATURECOMPUTER_HPP
//#define BSC_PROJECT_FEATURECOMPUTER_HPP
//
//#include <vector>
//#include "../defines.hpp"
//#include "AlphaNodeFeatures.hpp"
//
////Forward declarations
//template <class Pixel> class AlphaTree;
//template <class Pixel> class AlphaNode;
//
//template <class Pixel>
//class FeatureComputer {
//    struct Moments {
//        double sumX = 0.0;
//        double sumY = 0.0;
//        double sumX2 = 0.0;
//        double sumY2 = 0.0;
//    };
//
//    static bool pixelBelongsToNode(const AlphaTree<Pixel>& tree, ImgIdx pixelIndex, ImgIdx nodeIdx);
//    static Moments computeMoments(const AlphaTree<Pixel>& tree, ImgIdx nodeIdx);
//    static void computeElongation(const Moments& moments, double area, AlphaNodeFeatures& features);
//
//  public:
//    static AlphaNodeFeatures computeFeatures(const AlphaTree<Pixel>& tree, ImgIdx nodeIdx);
//};
//
//#endif // BSC_PROJECT_FEATURECOMPUTER_HPP
