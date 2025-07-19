#include "GmlvqClassifier.hpp"
#include <cmath>
#include <algorithm>
#include <random>

GMLVQClassifier::GMLVQClassifier(int nPrototypesPerClass, double lrProto, double lrMatrix, int nEpochs)
    : lrProto(lrProto), lrMatrix(lrMatrix), nEpochs(nEpochs) {
    // Initialize relevance matrix as identity
    int dim = AlphaNodeFeatures::featureCount();
    relevanceMatrix.resize(dim, std::vector<double>(dim, 0.0));
    for (int i = 0; i < dim; ++i) relevanceMatrix[i][i] = 1.0;
}

double GMLVQClassifier::distance(const AlphaNodeFeatures& features, const std::vector<double>& w) const {
    // Mahalanobis-like distance: (x-w)^T * Lambda * (x-w)
    int dim = w.size();
    std::vector<double> diff(dim);
    std::vector<double> x = features.toVector();
    for (int i = 0; i < dim; ++i) diff[i] = x[i] - w[i];
    double d = 0.0;
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j)
            d += diff[i] * relevanceMatrix[i][j] * diff[j];
    return d;
}

void GMLVQClassifier::train(const std::vector<AlphaNodeFeatures>& features, const std::vector<std::string>& labels) {
    int dim = AlphaNodeFeatures::featureCount();
    // Initialize prototypes (random samples per class)
    std::vector<std::string> classes;
    for (const auto& l : labels)
        if (std::find(classes.begin(), classes.end(), l) == classes.end())
            classes.push_back(l);

    prototypes.clear();
    prototypeLabels.clear();
    std::mt19937 rng(42);
    for (const auto& c : classes) {
        std::vector<int> idxs;
        for (size_t i = 0; i < labels.size(); ++i)
            if (labels[i] == c) idxs.push_back(i);
        std::shuffle(idxs.begin(), idxs.end(), rng);
        for (int p = 0; p < 1; ++p) { // 1 prototype per class
            prototypes.push_back(features[idxs[p]].toVector());
            prototypeLabels.push_back(c);
        }
    }

    // Training loop
    for (int epoch = 0; epoch < nEpochs; ++epoch) {
        // Shuffle training data
        std::vector<size_t> indices(features.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        std::vector<AlphaNodeFeatures> shuffledFeatures;
        std::vector<std::string> shuffledLabels;
        for (size_t idx : indices) {
            shuffledFeatures.push_back(features[idx]);
            shuffledLabels.push_back(labels[idx]);
        }

        for (size_t i = 0; i < shuffledFeatures.size(); ++i) {
            std::vector<double> x = shuffledFeatures[i].toVector();
            int label = std::distance(classes.begin(), std::find(classes.begin(), classes.end(), labels[i]));
            // Find winner (same class) and loser (different class)
            double minWinDist = INFINITY, minLoseDist = INFINITY;
            int winnerIdx = -1, loserIdx = -1;
            for (size_t p = 0; p < prototypes.size(); ++p) {
                double d = distance(shuffledFeatures[i], prototypes[p]);
                if (prototypeLabels[p] == shuffledLabels[i] && d < minWinDist) {
                    minWinDist = d; winnerIdx = p;
                }
                if (prototypeLabels[p] != shuffledLabels[i] && d < minLoseDist) {
                    minLoseDist = d; loserIdx = p;
                }
            }
            if (winnerIdx == -1 || loserIdx == -1) continue;
            updatePrototypesAndMatrix(shuffledFeatures[i], winnerIdx, loserIdx);
        }
    }
}

void GMLVQClassifier::updatePrototypesAndMatrix(const AlphaNodeFeatures& x, int winnerIdx, int loserIdx) {
    std::vector<double> xvec = x.toVector();
    int dim = xvec.size();

    // Compute difference vectors
    std::vector<double> winDiff(dim), loseDiff(dim);
    for (int i = 0; i < dim; ++i) {
        winDiff[i] = xvec[i] - prototypes[winnerIdx][i];
        loseDiff[i] = xvec[i] - prototypes[loserIdx][i];
    }

    // Compute distances
    double dw = 0.0, dl = 0.0; //distance winner and distance loser
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            dw += winDiff[i] * relevanceMatrix[i][j] * winDiff[j];
            dl += loseDiff[i] * relevanceMatrix[i][j] * loseDiff[j];
        }
    }

    // Compute D
    double denom = dw + dl;
    if (denom == 0) denom = 1e-8; // Avoid division by zero
    double D = (dw - dl) / denom;

    // Gradient for relevance matrix
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            double grad = (2.0 / denom) * (dl * loseDiff[i] * loseDiff[j] - dw * winDiff[i] * winDiff[j]);
            relevanceMatrix[i][j] -= lrMatrix * D * grad;
        }
    }

    // Trace normalization
    double trace = 0.0;
    for (int i = 0; i < dim; ++i) trace += relevanceMatrix[i][i];
    if (trace > 0) {
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                relevanceMatrix[i][j] /= trace;
            }
        }
    }

    // Update prototypes
    for (int i = 0; i < dim; ++i) {
        double gradWin = 0.0, gradLose = 0.0;
        for (int j = 0; j < dim; ++j) {
            gradWin  += relevanceMatrix[i][j] * winDiff[j];
            gradLose += relevanceMatrix[i][j] * loseDiff[j];
        }
        double scale = 2.0 / (denom * denom);
        prototypes[winnerIdx][i] -= lrProto * scale * dl * gradWin;
        prototypes[loserIdx][i]  += lrProto * scale * dw * gradLose;
    }
}

std::string GMLVQClassifier::predict(const AlphaNodeFeatures& features) const {
    double minDist = INFINITY;
    int bestIdx = -1;
    for (size_t p = 0; p < prototypes.size(); ++p) {
        double d = distance(features, prototypes[p]);
        if (d < minDist) { minDist = d; bestIdx = p; }
    }
    return bestIdx == -1 ? "" : prototypeLabels[bestIdx];
}

double GMLVQClassifier::crossValidate(const std::vector<AlphaNodeFeatures>& features, const std::vector<std::string>& labels, int folds) {
    int n = features.size();
    int foldSize = n / folds;
    int correct = 0, total = 0;
    for (int f = 0; f < folds; ++f) {
        int start = f * foldSize, end = (f + 1) * foldSize;
        std::vector<AlphaNodeFeatures> trainX, testX;
        std::vector<std::string> trainY, testY;
        for (int i = 0; i < n; ++i) {
            if (i >= start && i < end) {
                testX.push_back(features[i]);
                testY.push_back(labels[i]);
            } else {
                trainX.push_back(features[i]);
                trainY.push_back(labels[i]);
            }
        }
        train(trainX, trainY);
        for (size_t i = 0; i < testX.size(); ++i) {
            if (predict(testX[i]) == testY[i]) correct++;
            total++;
        }
    }
    return total == 0 ? 0.0 : (double)correct / total;
}
