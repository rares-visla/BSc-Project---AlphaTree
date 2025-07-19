#ifndef BSC_PROJECT_ALPHANODEFEATURES_HPP
#define BSC_PROJECT_ALPHANODEFEATURES_HPP

#include <vector>

struct AlphaNodeFeatures{
    float area;
    float compactness;
    float avgRed;
    float avgGreen;
    float avgBlue;

    AlphaNodeFeatures():
        area(0),
        compactness(0),
        avgRed(0),
        avgGreen(0),
        avgBlue(0)
    {}

    static int featureCount() {
        return 5; // area, compactness, avgRed, avgGreen, avgBlue
    }

    std::vector<double> toVector() const {
        return {static_cast<double>(area), static_cast<double>(compactness),
                static_cast<double>(avgRed), static_cast<double>(avgGreen),
                static_cast<double>(avgBlue)};
    }

};

#endif // BSC_PROJECT_ALPHANODEFEATURES_HPP
