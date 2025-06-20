#ifndef BSC_PROJECT_ALPHANODEFEATURES_HPP
#define BSC_PROJECT_ALPHANODEFEATURES_HPP

struct AlphaNodeFeatures{
    int area;
    double elongation;
    double avgRed;
    double avgGreen;
    double avgBlue;

    AlphaNodeFeatures():
        area(0),
        elongation(0),
        avgRed(0),
        avgGreen(0),
        avgBlue(0)
    {}
};

#endif // BSC_PROJECT_ALPHANODEFEATURES_HPP
