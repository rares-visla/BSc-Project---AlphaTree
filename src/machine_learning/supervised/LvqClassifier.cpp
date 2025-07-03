//
// Created by Rares on 03/07/2025.
//

#include "LvqClassifier.hpp"

LVQClassifier::LVQClassifier(int PrototypesPerClass, double learningRate, double learningRateDecay, int maxEpochs)
    :learningRate_(learningRate),
      learningRateDecay_(learningRateDecay),
      maxEpochs_(maxEpochs),
      prototypesPerClass_(PrototypesPerClass)
{}

void LVQClassifier::train(const std::vector<AlphaNodeFeatures> &features, const std::vector<std::string> &labels, LVQClassifier::LVQType algorithm) {
    if (features.size() != labels.size()){
        std::cerr << "Error: features and labels must have the same size" << std::endl;
        return;
    }

    // Convert to training points
    std::vector<LVQTrainingPoint> trainingData;
    trainingData.reserve(features.size());

    for (size_t i = 0; i < features.size(); i++) {
        trainingData.emplace_back(features[i], labels[i]);
    }

    train(trainingData, algorithm);
}

void LVQClassifier::train(const std::vector<LVQTrainingPoint> &trainingData, LVQClassifier::LVQType algorithm) {
    if (trainingData.empty()) {
        std::cerr << "Error: No training data provided" << std::endl;
        return;
    }

    // Make a copy for normalization
    std::vector<LVQTrainingPoint> normalizedData = trainingData;
    normalizeFeatures(normalizedData);

    // Build label mappings
    labelToId_.clear();
    idToLabel_.clear();
    for (const auto& point : normalizedData) {
        if (labelToId_.find(point.label) == labelToId_.end()) {
            int id = labelToId_.size();
            labelToId_[point.label] = id;
            idToLabel_[id] = point.label;
        }
    }

    std::cout << "Training LVQ with " << labelToId_.size() << " classes:" << std::endl;
    for (const auto& pair : labelToId_) {
        std::cout << "  " << pair.first << " (ID: " << pair.second << ")" << std::endl;
    }

    // Initialize codebooks
    initializePrototypes(normalizedData);

    // Train using selected algorithm
    switch (algorithm) {
    case LVQ1:
        std::cout << "Using LVQ1 algorithm" << std::endl;
        trainLVQ1(normalizedData);
        break;
    case LVQ2_1:
        std::cout << "Using LVQ2.1 algorithm" << std::endl;
        trainLVQ2_1(normalizedData);
        break;
    case LVQ3:
        std::cout << "Using LVQ3 algorithm" << std::endl;
        trainLVQ3(normalizedData);
        break;
    }

    std::cout << "Training completed!" << std::endl;
}

void LVQClassifier::normalizeFeatures(std::vector<LVQTrainingPoint> &data) {
    if (data.empty()) return;

    int dimensions = data[0].features.size();
    minVals_.assign(dimensions, std::numeric_limits<double>::max());
    maxVals_.assign(dimensions, std::numeric_limits<double>::lowest());

    // Find min and max values
    for (const auto& point : data) {
        for (int d = 0; d < dimensions; d++) {
            minVals_[d] = std::min(minVals_[d], point.features[d]);
            maxVals_[d] = std::max(maxVals_[d], point.features[d]);
        }
    }

    // Normalize to [0, 1] range
    for (auto& point : data) {
        for (int d = 0; d < dimensions; d++) {
            if (maxVals_[d] != minVals_[d]) {
                point.features[d] = (point.features[d] - minVals_[d]) / (maxVals_[d] - minVals_[d]);
            } else {
                point.features[d] = 0.0;
            }
        }
    }
}

void LVQClassifier::initializePrototypes(const std::vector<LVQTrainingPoint> &trainingData) {
    prototypes_.clear();

    int dimensions = trainingData[0].features.size();

    //Create prototypes for each class
    for (const auto& labelPair : labelToId_) {
        const std::string& label = labelPair.first;
        int classId = labelPair.second;

        //Collect points for this class
        std::vector<std::vector<double>> classPoints;
        for (const auto& point : trainingData) {
            if (point.classId == classId) {
                classPoints.push_back(point.features);
            }
        }

        if (classPoints.empty()) continue;

        //Create prototypes for this class
        for (int i = 0; i < prototypesPerClass_; i++) {
            LVQPrototype prototype(dimensions, label, classId);

            if (classPoints.size() == 1) {
                prototype.weights = classPoints[0];
            } else {
                std::uniform_int_distribution<int> distribution(0, classPoints.size() - 1);
                int randomPointIdx = distribution(rng_);
                prototype.weights = classPoints[randomPointIdx];
            }
            prototypes_.push_back(prototype);
        }
    }

    std::cout << "Initialized " << prototypes_.size() << " prototypes ("
              << prototypesPerClass_ << " per class)" << std::endl;
}

double LVQClassifier::calculateDistance(const std::vector<double> &point1, const std::vector<double> &point2) {
    double distance = 0.0;
    for (int d = 0; d < point1.size(); d++) {
        distance += std::pow(point1[d] - point2[d], 2);
    }
    return std::sqrt(distance);
}

int LVQClassifier::findClosestPrototype(const std::vector<double> &features) {
    if (prototypes_.empty()) return -1;

    double minDistance = std::numeric_limits<double>::max();
    int closestPrototypeIdx = -1;

    for (int i = 0; i < prototypes_.size(); i++) {
        double distance = calculateDistance(features, prototypes_[i].weights);
        if (distance < minDistance) {
            minDistance = distance;
            closestPrototypeIdx = i;
        }
    }

    return closestPrototypeIdx;
}

void LVQClassifier::updatePrototype(int prototypeIdx, const std::vector<double> &features, bool moveTowards) {
    if (prototypeIdx < 0 || prototypeIdx >= prototypes_.size()) return;

    auto& prototype = prototypes_[prototypeIdx];
    double factor = moveTowards ? learningRate_ : -learningRate_;

    for (int d = 0; d < prototype.weights.size(); d++) {
        prototype.weights[d] += factor * (features[d] - prototype.weights[d]);
    }
}

void LVQClassifier::trainLVQ1(const std::vector<LVQTrainingPoint> &trainingData) {
    double currentLearningRate = learningRate_;

    for (int epoch = 0; epoch < maxEpochs_; epoch++) {
        //Shuffle training data
        std::vector<int> indices(trainingData.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng_);

        int correct = 0;

        for (int idx : indices) {
            const auto& point = trainingData[idx];

            int closestPrototypeIdx = findClosestPrototype(point.features);
            if (closestPrototypeIdx == -1) continue;

            bool sameClass = (prototypes_[closestPrototypeIdx].classId == point.classId);
            updatePrototype(closestPrototypeIdx, point.features, sameClass);

            if (sameClass) correct++;
        }

        //Decay learning rate
        learningRate_ = currentLearningRate * pow(learningRateDecay_, epoch);

        if ((epoch + 1) % 10 == 0) {
            double accuracy = static_cast<double>(correct) / trainingData.size();
            std::cout << "Epoch " << epoch + 1 << "/" << maxEpochs_
                      << ": " << std::fixed << std::setprecision(3)
                      << accuracy << ", LR: " << learningRate_ << std::endl;
        }
    }

    //Restore original learning rate
    learningRate_ = currentLearningRate;
}

void LVQClassifier::trainLVQ2_1(const std::vector<LVQTrainingPoint> &trainingData) {
    std::cout << "Not implemented yet" << std::endl;
    return;
}

void LVQClassifier::trainLVQ3(const std::vector<LVQTrainingPoint> &trainingData) {
    std::cout << "Not implemented yet" << std::endl;
    return;
}

std::string LVQClassifier::predict(const AlphaNodeFeatures &features) {
    std::vector<double> normalizedFeatures = {
        features.area, features.compactness,
        features.avgRed, features.avgGreen, features.avgBlue,
    };

    //Normalize using training statistics
    for (int i = 0; i < normalizedFeatures.size() && i < minVals_.size(); i++) {
        if (maxVals_[i] != minVals_[i]) {
            normalizedFeatures[i] = (normalizedFeatures[i] - minVals_[i]) / (maxVals_[i] - minVals_[i]);
        } else {
            normalizedFeatures[i] = 0.0;
        }
    }

    int closestIdx = findClosestPrototype(normalizedFeatures);
    if (closestIdx == -1) return "unknown";

    return prototypes_[closestIdx].label;
}

std::vector<std::string> LVQClassifier::predict(const std::vector<AlphaNodeFeatures> &features) {
    std::vector<std::string> predictions;
    predictions.reserve(features.size());

    for (const auto& feature : features) {
        predictions.push_back(predict(feature));
    }

    return predictions;
}

void LVQClassifier::printPrototypeInfo() const {
    std::cout << "\n=== LVQ Prototype Information ===" << std::endl;
    std::cout << "Total prototypes: " << prototypes_.size() << std::endl;
    std::cout << "Prototypes per class: " << prototypesPerClass_ << std::endl;

    for (size_t i = 0; i < prototypes_.size(); i++) {
        const auto& cb = prototypes_[i];
        std::cout << "\nPrototype " << i << " (Class: " << cb.label << "):" << std::endl;
        std::cout << "  Weights: [";
        for (size_t j = 0; j < cb.weights.size(); j++) {
            std::cout << std::fixed << std::setprecision(3) << cb.weights[j];
            if (j < cb.weights.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "================================\n" << std::endl;

}

double LVQClassifier::evaluateOnTestSet(const std::vector<AlphaNodeFeatures> &testFeatures, const std::vector<std::string> &testLabels) {
    if (testFeatures.size() != testLabels.size()) {
        std::cerr << "Error: test features and labels must have the same size" << std::endl;
        return 0.0;
    }

    std::vector<std::string> predictions = predict(testFeatures);

    int correct = 0;
    std::map<std::string, std::map<std::string, int>> confusionMatrix;

    for (int i = 0; i < predictions.size(); i++) {
        confusionMatrix[testLabels[i]][predictions[i]]++;
        if (testLabels[i] == predictions[i]) correct++;
    }

    double accuracy = static_cast<double>(correct) / testFeatures.size();
    std::cout << "\n=== LVQ Test Results ===" << std::endl;
    std::cout << "Test accuracy: " << std::fixed << std::setprecision(3) << accuracy << std::endl;

    std::cout << "\nConfusion matrix:" << std::endl;
    std::cout << "Actual\\Predicted\t";
    for (const auto& label : labelToId_) {
        std::cout << label.first << "\t";
    }
    std::cout << std::endl;

    for (const auto& actual : labelToId_) {
        std::cout << actual.first << "\t\t";
        for (const auto& predicted : labelToId_) {
            std::cout << confusionMatrix[actual.first][predicted.first] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << "========================\n" << std::endl;

    return accuracy;
}