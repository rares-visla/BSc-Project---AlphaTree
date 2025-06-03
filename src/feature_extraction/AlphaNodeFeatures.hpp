#ifndef BSC_PROJECT_ALPHANODEFEATURES_HPP
#define BSC_PROJECT_ALPHANODEFEATURES_HPP

struct AlphaNodeFeatures{
    int area;
    double elongation;
    double meanIntensity;
    double contrast;

    AlphaNodeFeatures():
        area(0),
        elongation(0),
        meanIntensity(0),
        contrast(0)
    {}
};

#endif // BSC_PROJECT_ALPHANODEFEATURES_HPP
