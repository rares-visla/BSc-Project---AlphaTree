#ifndef BSC_PROJECT_ALPHANODEFEATURES_HPP
#define BSC_PROJECT_ALPHANODEFEATURES_HPP

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
};

#endif // BSC_PROJECT_ALPHANODEFEATURES_HPP
