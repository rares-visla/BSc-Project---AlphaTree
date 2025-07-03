//
// Created by Rares on 03/07/2025.
//

#ifndef BSC_PROJECT_LVQCLASSIFIER_HPP
#define BSC_PROJECT_LVQCLASSIFIER_HPP

#include <vector>
#include <string>
#include <random>
#include <map>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include "../../feature_extraction/AlphaNodeFeatures.hpp"

struct LVQPrototype {
    std::vector<double> weights;
    std::string label;
    int classId;

    LVQPrototype(int dimensions, const std::string& lbl, int clsId)
        : weights(dimensions, 0.0), label(lbl), classId(clsId) {}
};

struct LVQTrainingPoint {
    std::vector<double> features;
    std::string label;
    int classId;

    LVQTrainingPoint() = default;
    LVQTrainingPoint(const AlphaNodeFeatures& nodeFeatures, const std::string& lbl)
        : label(lbl) {
        features = {
            nodeFeatures.area,
            nodeFeatures.compactness,
            nodeFeatures.avgRed,
            nodeFeatures.avgGreen,
            nodeFeatures.avgBlue
        };

        // Convert label to class ID
        if (lbl == "healthy") classId = 0;
        else if (lbl == "diseased") classId = 1;
        else classId = 2; // background/none
    }
};


class LVQClassifier {
  private:
    std::vector<LVQPrototype> prototypes_;
    std::map<std::string, int> labelToId_;
    std::map<int, std::string> idToLabel_;
    std::mt19937 rng_;

    // Hyperparameters
    double learningRate_;
    double learningRateDecay_;
    int maxEpochs_;
    int prototypesPerClass_;

    // Internal methods
    void initializePrototypes(const std::vector<LVQTrainingPoint>& trainingData);
    void normalizeFeatures(std::vector<LVQTrainingPoint>& data);
    void normalizePrototypes();
    double calculateDistance(const std::vector<double>& point1, const std::vector<double>& point2);
    int findClosestPrototype(const std::vector<double>& features);
    void updatePrototype(int prototypeIdx, const std::vector<double>& features, bool moveTowards);
    double calculateAccuracy(const std::vector<LVQTrainingPoint>& testData);

    // Statistics tracking
    std::vector<double> minVals_, maxVals_;

  public:
    // Different LVQ algorithms
    enum LVQType {
        LVQ1,    // Basic LVQ
        LVQ2_1,  // LVQ2.1 with window rule
        LVQ3     // LVQ3 with epsilon rule
    };

    LVQClassifier(int PrototypesPerClass = 2, double learningRate = 0.1,
                  double learningRateDecay = 0.95, int maxEpochs = 100);

    // Main training and prediction methods
    void train(const std::vector<AlphaNodeFeatures>& features,
               const std::vector<std::string>& labels,
               LVQType algorithm = LVQ1);

    void train(const std::vector<LVQTrainingPoint>& trainingData,
               LVQType algorithm = LVQ1);

    std::string predict(const AlphaNodeFeatures& features);
    std::vector<std::string> predict(const std::vector<AlphaNodeFeatures>& features);

    // LVQ algorithm implementations
    void trainLVQ1(const std::vector<LVQTrainingPoint>& trainingData);
    void trainLVQ2_1(const std::vector<LVQTrainingPoint>& trainingData); //maybe useful later
    void trainLVQ3(const std::vector<LVQTrainingPoint>& trainingData); //maybe useful later

    // Analysis and utility methods
    void printPrototypeInfo() const;
    void savePrototypes(const std::string& filename) const;
    void loadPrototypes(const std::string& filename);
    double evaluateOnTestSet(const std::vector<AlphaNodeFeatures>& testFeatures,
                             const std::vector<std::string>& testLabels);

    // Cross-validation
    double crossValidate(const std::vector<AlphaNodeFeatures>& features,
                         const std::vector<std::string>& labels,
                         int folds = 5, LVQType algorithm = LVQ1);

    // Visualization helpers
    std::vector<std::vector<double>> getPrototypeWeights() const;
    std::vector<std::string> getPrototypeLabels() const;

};

#endif // BSC_PROJECT_LVQCLASSIFIER_HPP
