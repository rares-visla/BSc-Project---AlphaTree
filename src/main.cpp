#include "alpha_tree/image_handling/PNGcodec.hpp"
#include "alpha_tree/utility/defines.hpp"

#include "alpha_tree/AlphaTree.hpp"
#include "alpha_tree/AlphaTreeConfig.hpp"
#include "alpha_tree/image_handling/RandGenImage.hpp"
#include "feature_extraction/FeatureComputer.hpp"
#include "machine_learning/supervised/BoundingBoxLabeler.hpp"
#include "machine_learning/supervised/GmlvqClassifier.hpp"
#include "machine_learning/supervised/LvqClassifier.hpp"
#include <filesystem>

// args: Filename, nchannels, numthreads, testimgsize, algorithmcode, bitdepth, tseflag
int main(int argc, char **argv) {
    srand(time(NULL));

    const auto configFileName = argc < 2 ? "config.txt" : std::string(argv[1]);
    alphatreeConfig.initialize(configFileName);
    auto config = alphatreeConfig.load(argc, argv);

    if (config.has_value() == false) {
        std::cerr << "Unable to open configuration file." << std::endl;
        return -1;
    }

    AlphaTreeConfig::AlphaTreeParameters params = config.value();

    const std::string filePath = params.imageFileName;

    auto [image, w, h, ch] = PNGCodec::imread(params.UseRandomlyGeneratedImages ? "RAND" : filePath);

    const bool reduceImageBitdepth = false;

    const auto &width = params.UseRandomlyGeneratedImages ? params.randomGenImageWidth : w;
    const auto &height = params.UseRandomlyGeneratedImages ? params.randomGenImageHeight : h;
    const auto &bitdepth = params.UseRandomlyGeneratedImages ? params.bitdepth : 8 * ch;
    const auto &nch = params.UseRandomlyGeneratedImages ? params.nchannels : ch;
    const auto &dMetric = params.dissimilarityMetric;
    const auto &conn = params.connectivity;
    const auto &algCode = params.alphaTreeAlgorithmCode;
    const auto &nthr = params.numthreads;
    const auto &nitr = params.numitr;
    const auto &tse = params.tse;
    const auto &iparam1 = params.iparam1;
    const auto &fparam1 = params.fparam1;
    const auto &fparam2 = params.fparam2;

    if (!params.UseRandomlyGeneratedImages)
        printf("Image file name: %s\n", params.imageFileName.c_str());
    printf("=======================================================================\n");
    printf("========== imgsize = %d x %d (%d bits, %d ch, %dN) ================\n", (int)height, (int)width,
           (int)bitdepth, (int)nch, params.connectivity);
    printf("=======================================================================\n");
    printf("-----------------------------------------------------------------------------------\n");
    printf("%d Running %s (%d threads)\n", (int)algCode, alphatreeConfig.getAlphaTreeAlgorithmName(algCode).c_str(),
           (int)nthr);
    printf("-----------------------------------------------------------------------------------\n");
    std::vector<double> runtimes;

    double tStart = 0, tEnd = INFINITY;

    std::vector<uint8_t> image8(image.size());
    for (size_t i = 0; i < image.size(); i++)
        image8[i] = (uint8_t)image[i];
    AlphaTree<uint8_t> tree;
    tStart = get_wall_time();
    tree.BuildAlphaTree(image8.data(), height, width, nch, dMetric, conn, algCode, nthr, tse, fparam1,
                        fparam2, iparam1);
    tEnd = get_wall_time();

    auto runtime = tEnd - tStart;
    printf("-------------------Run %d/%d: %.3f------------------\n", (int)1, nitr, runtime);
    runtimes.push_back(runtime);

    //TREE HAS BEEN BUILT

    BoundingBoxLabeler labeler("images/image1_boxes.txt", width, height);

    std::cout << "Number of nodes: " << tree._curSize << std::endl;
    std::vector<AlphaNodeFeatures> featureVectors;
    std::vector<std::string> nodeLabels;
    std::map<unsigned long, unsigned long>featureIdxToNodeIdx; // Maps feature vector index to node index

    for (int i = 0; i < tree._curSize; i++) {
        if (tree._node[i].area > 5000 && tree._node[i].area < 100000) {
            tree._node[i].computeFeatures();
            if (tree._node[i].featuresComputed) {
                //Assign label based on bounding box
                std::string label = labeler.labelNode(tree, i);

                featureVectors.push_back(tree._node[i].features);
                nodeLabels.push_back(label);
                featureIdxToNodeIdx[featureVectors.size() - 1] = i; // Store the mapping
            }
        }
    }

    //FEATURES HAVE BEEN COMPUTED

    // Add this after computing features
    std::cout << "\n=== Alpha Tree Node Coverage ===" << std::endl;
    std::cout << "Image dimensions: " << width << "x" << height << std::endl;
    std::cout << "Total alpha tree nodes: " << featureVectors.size() << std::endl;

    // Check node area distribution
    std::vector<double> areas;
    for (const auto& features : featureVectors) {
        areas.push_back(features.area);
    }
    std::sort(areas.begin(), areas.end());

    std::cout << "Node area range: " << areas.front() << " to " << areas.back() << std::endl;
    std::cout << "Median node area: " << areas[areas.size()/2] << std::endl;


    // Count and display label distribution
    int healthyCount = 0, diseasedCount = 0, noneCount = 0;
    for (const auto& label : nodeLabels) {
        if (label == "healthy") healthyCount++;
        else if (label == "diseased") diseasedCount++;
        else noneCount++;
    }

    std::cout << "Label distribution:" << std::endl;
    std::cout << "  Healthy: " << healthyCount << " ("
              << (100.0 * healthyCount / nodeLabels.size()) << "%)" << std::endl;
    std::cout << "  Diseased: " << diseasedCount << " ("
              << (100.0 * diseasedCount / nodeLabels.size()) << "%)" << std::endl;
    std::cout << "  Background/None: " << noneCount << " ("
              << (100.0 * noneCount / nodeLabels.size()) << "%)" << std::endl;

    std::ofstream out("nodes_features.csv");
    out << "area,compactness,avgRed,avgGreen,avgBlue,label\n";
    for (size_t i = 0; i < featureVectors.size(); ++i) {
        const auto& f = featureVectors[i];
        out << f.area << "," << f.compactness << "," << f.avgRed << ","
            << f.avgGreen << "," << f.avgBlue << "," << nodeLabels[i] << "\n";
    }
    out.close();

//    LVQClassifier classifier(2, 0.1, 0.95, 100);
//    int folds = 5;
//    double avgAccuracy = classifier.crossValidate(featureVectors, nodeLabels, folds, LVQClassifier::LVQType::LVQ1);
//    std::cout << "Average accuracy over " << folds << " folds: " << avgAccuracy * 100.0 << "%" << std::endl;

    GMLVQClassifier gmlvq(1, 0.05, 0.01, 100); // 1 prototype/class, proto LR=0.05, matrix LR=0.01, 100 epochs
    int folds = 5;
    double avgAccuracy = gmlvq.crossValidate(featureVectors, nodeLabels, folds);
    std::cout << "GMLVQ average accuracy over " << folds << " folds: " << avgAccuracy * 100.0 << "%" << std::endl;


//    // Convert your image8 vector to cv::Mat
//    cv::Mat visImg(height, width, CV_8UC1, image8.data());
//    cv::Mat colorImg;
//    cv::cvtColor(visImg, colorImg, cv::COLOR_GRAY2BGR);
//
////    int minsize = 10000;
////    int maxsize = 100000;
//    // Draw colored bounding boxes for filtered nodes
//    for (size_t idx = 0; idx < featureVectors.size(); ++idx) {
//        const auto& features = featureVectors[idx];
//        const std::string& label = nodeLabels[idx];
//        const auto& node = tree._node[featureIdxToNodeIdx[idx]]; // Get the corresponding node
//
//        int minX = node.minX;
//        int minY = node.minY;
//        int maxX = node.maxX;
//        int maxY = node.maxY;
//
//        cv::Scalar color;
//        if (label == "healthy")
//            color = cv::Scalar(0, 255, 0); // Green
//        else if (label == "diseased")
//            color = cv::Scalar(0, 0, 255); // Red
//        else
//            color = cv::Scalar(255, 0, 0); // Blue
//
//        cv::rectangle(
//            colorImg,
//            cv::Point(minX, minY),
//            cv::Point(maxX, maxY),
//            color,
//            2
//        );
//    }

//    std::string outputFileName = "nodes_bounding_boxes_labeled_0.3_IoU.png";
//    cv::imwrite(outputFileName, colorImg);

    if (!runtimes.empty()) {
        double minRuntime = *std::min_element(runtimes.begin(), runtimes.end());
        double imgsize = (double)(width * height);

        printf("================== Summary ==================\n");
        printf("Processing speed: %.3fMpix/s / Memory use %.3fB/pix \n", (imgsize / minRuntime) * 1e-6,
               (double)max_memuse / imgsize);
        printf("=============================================\n");
    }

    return 0;
}
