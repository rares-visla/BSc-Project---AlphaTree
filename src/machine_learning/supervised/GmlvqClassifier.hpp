#ifndef BSC_PROJECT_GMLVQCLASSFIER_HPP
#define BSC_PROJECT_GMLVQCLASSFIER_HPP

#include "../../feature_extraction/AlphaNodeFeatures.hpp"
#include <string>
#include <vector>

class GMLVQClassifier {
  public:
    GMLVQClassifier(int nPrototypesPerClass, double lrProto, double lrMatrix, int nEpochs);
    void train(const std::vector<AlphaNodeFeatures> &features, const std::vector<std::string> &labels);
    std::string predict(const AlphaNodeFeatures &features) const;
    double crossValidate(const std::vector<AlphaNodeFeatures> &features, const std::vector<std::string> &labels,
                         int folds);

  private:
    std::vector<std::vector<double>> prototypes;
    std::vector<std::string> prototypeLabels;
    std::vector<std::vector<double>> relevanceMatrix; // Lambda
    double lrProto, lrMatrix;
    int nEpochs;
    double distance(const AlphaNodeFeatures &x, const std::vector<double> &w) const;
    void updatePrototypesAndMatrix(const AlphaNodeFeatures &x, int winnerIdx, int loserIdx);
};

#endif // BSC_PROJECT_GMLVQCLASSFIER_HPP
