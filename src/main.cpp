#include "alpha_tree/image_handling/PNGcodec.hpp"
#include "alpha_tree/utility/defines.hpp"

#include "alpha_tree/AlphaTree.hpp"
#include "alpha_tree/AlphaTreeConfig.hpp"
#include "alpha_tree/image_handling/RandGenImage.hpp"
#include "feature_extraction/FeatureComputer.hpp"
#include "machine_learning/supervised/BoundingBoxLabeler.hpp"
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

//    for (int itr = 0; itr < nitr; itr++) {
//        double tStart = 0, tEnd = INFINITY;
//        if (params.UseRandomlyGeneratedImages == true) {
//            if (bitdepth > 32 && bitdepth <= 64) {
//                uint64_t *image = (uint64_t *)Malloc(width * height * sizeof(uint64_t));
//                RandGenImage::randomize64(image, width, height, bitdepth);
//                AlphaTree<uint64_t> tree;
//                tStart = get_wall_time();
//                tree.BuildAlphaTree(image, height, width, nch, dMetric, conn, algCode, nthr, tse, fparam1, fparam2,
//                                    iparam1);
//                tEnd = get_wall_time();
//                Free(image);
//            } else if (bitdepth > 16) {
//                uint32_t *image = (uint32_t *)Malloc(width * height * sizeof(uint32_t));
//                RandGenImage::randomize32(image, width, height, bitdepth);
//                AlphaTree<uint32_t> tree;
//                tStart = get_wall_time();
//                tree.BuildAlphaTree(image, height, width, nch, dMetric, conn, algCode, nthr, tse, fparam1, fparam2,
//                                    iparam1);
//                tEnd = get_wall_time();
//                std::cout << "\n" << tree._curSize << " nodes\n" << std::endl;;
////                for (ImgIdx i = 0; i < tree._curSize; i++) {
////                    if (!tree._node[i].featuresComputed) {
////                        tree._node[i].features = FeatureComputer<uint32_t>::computeFeatures(tree, i);
////                        tree._node[i].featuresComputed = true;
//////                        std::cout << "Node " << i << " features:\n";
//////                        std::cout << "Area: " << tree._node[i].features.area << "\n";
//////                        std::cout << "Elongation: " << tree._node[i].features.elongation << "\n";
//////                        std::cout << "Mean Intensity: " << tree._node[i].features.meanIntensity << "\n";
//////                        std::cout << "Contrast: " << tree._node[i].features.contrast << "\n";
////
////                    }
////                    if (tree._node[i].area > 25 && tree._node[i].alpha > 100) {
////                        std::cout << "Node " << i << " features:\n";
////                        std::cout << "Area: " << tree._node[i].features.area << "\n";
////                        std::cout << "Elongation: " << tree._node[i].features.elongation << "\n";
////                        std::cout << "Mean Intensity: " << tree._node[i].features.meanIntensity << "\n";
////                        std::cout << "Contrast: " << tree._node[i].features.contrast << "\n";
////                    }
////                }
//                Free(image);
//            } else if (bitdepth > 8) {
//                uint16_t *image = (uint16_t *)Malloc(width * height * nch * sizeof(uint16_t));
//                RandGenImage::randomize16(image, width, height, bitdepth, nch);
//                AlphaTree<uint16_t> tree;
//                tStart = get_wall_time();
//                tree.BuildAlphaTree(image, height, width, nch, dMetric, conn, algCode, nthr, tse, fparam1, fparam2,
//                                    iparam1);
//                tEnd = get_wall_time();
//                Free(image);
//            } else if (bitdepth > 0) {
//                uint8_t *image = (uint8_t *)Malloc(width * height * nch * sizeof(uint8_t));
//                RandGenImage::randomize8(image, width, height, bitdepth, nch);
//                AlphaTree<uint8_t> tree;
//                tStart = get_wall_time();
//                tree.BuildAlphaTree(image, height, width, nch, dMetric, conn, algCode, nthr, tse, fparam1, fparam2,
//                                    iparam1);
//                tEnd = get_wall_time();
//                Free(image);
//            } else {
//                std::cerr << "Invalid bit-depth. " << std::endl;
//                return -1;
//            }
//
//        } else {
//
//            if (reduceImageBitdepth) {
//                const size_t shamt = 16 - params.bitdepth;
//                uint16_t pMin = 65535;
//                uint16_t pMax = 0;
//
//                for (auto &pixel : image) {
//                    pixel = pixel >> shamt;
//                    pMax = std::max(pMax, pixel);
//                    pMin = std::min(pMin, pixel);
//                }
//
//                printf("reduceImageBitdepth - BitDepth = params.bitdepth = %d / DR = %d - %d\n", (int)params.bitdepth,
//                       (int)pMin, (int)pMax);
//            }
//
//            uint16_t maxVal = *std::max_element(image.begin(), image.end());
//            if ((int)maxVal > (int)255) {
//                AlphaTree<uint16_t> tree;
//                tStart = get_wall_time();
//                tree.BuildAlphaTree(image.data(), height, width, nch, dMetric, conn, algCode, nthr, tse, fparam1,
//                                    fparam2, iparam1);
//
//                const bool rgbFilter = false;
//                if (rgbFilter) {
//
//                    int sizeThr = 16;
//                    for (int i = 1; i < 300; i++) {
//                        int alphaThr = 0.1 + i * 10;
//                        tree.AlphaFilter(image.data(), alphaThr, sizeThr);
//                        std::string str = "out_" + std::to_string(alphaThr) + ".png";
//                        PNGCodec::imwrite(image, w, h, ch, str.c_str());
//                    }
//                }
//
//                tEnd = get_wall_time();
//            } else {
//                std::vector<uint8_t> image8(image.size());
//                for (size_t i = 0; i < image.size(); i++)
//                    image8[i] = (uint8_t)image[i];
//                AlphaTree<uint8_t> tree;
//                tStart = get_wall_time();
//                tree.BuildAlphaTree(image8.data(), height, width, nch, dMetric, conn, algCode, nthr, tse, fparam1,
//                                    fparam2, iparam1);
//                tEnd = get_wall_time();
//            }
//        }
//
//        auto runtime = tEnd - tStart;
//        printf("-------------------Run %d/%d: %.3f------------------\n", (int)itr + 1, nitr, runtime);
//        runtimes.push_back(runtime);
//    }

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

//    ImgIdx nodeID = 16094;
//    std::cout << "Bounding Box of node " << nodeID << ":\n";
//    std::cout << "MinX: " << tree._node[nodeID].minX << ", MinY: " << tree._node[nodeID].minY << "\n";
//    std::cout << "MaxX: " << tree._node[nodeID].maxX << ", MaxY: " << tree._node[nodeID].maxY << "\n";
//
//    ImgIdx bboxWidth = tree._node[nodeID].maxX - tree._node[nodeID].minX + 1;
//    ImgIdx bboxHeight = tree._node[nodeID].maxY - tree._node[nodeID].minY + 1;
//
//    std::cout << "Width: " << bboxWidth << ", Height: " << bboxHeight << "\n";

//    std::vector<BoundingBox> boundingBoxes = BoundingBoxLabeler::loadBoundingBoxes("images/image1_boxes.txt", width, height);

    BoundingBoxLabeler labeler("images/image1_boxes.txt", width, height);
//    labeler.printBoundingBoxInfo();

    std::cout << "Number of nodes: " << tree._curSize << std::endl;
    std::vector<AlphaNodeFeatures> featureVectors;
    std::vector<std::string> nodeLabels;
    std::map<int, int>featureIdxToNodeIdx; // Maps feature vector index to node index

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
//    std::cout << "Number of features: " << featureVectors.size() << std::endl;
//    for (int i = 0; i < tree._curSize; i++) {
//        if (tree._node[i].area > 100) {
//            std::string label = labeler.labelNode(tree, i);
//            nodeLabels.push_back(label);
//        }
//    }


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

    // Convert your image8 vector to cv::Mat
    cv::Mat visImg(height, width, CV_8UC1, image8.data());
    cv::Mat colorImg;
    cv::cvtColor(visImg, colorImg, cv::COLOR_GRAY2BGR);

//    int minsize = 10000;
//    int maxsize = 100000;
    // Draw colored bounding boxes for filtered nodes
    for (size_t idx = 0; idx < featureVectors.size(); ++idx) {
        const auto& features = featureVectors[idx];
        const std::string& label = nodeLabels[idx];
        const auto& node = tree._node[featureIdxToNodeIdx[idx]]; // Get the corresponding node

        // Find the corresponding node index (optional: if you store node index, use it directly)
        // Here, assume featureVectors and nodeLabels are in the same order as the filtered nodes

        int minX = node.minX;
        int minY = node.minY;
        int maxX = node.maxX;
        int maxY = node.maxY;

        cv::Scalar color;
        if (label == "healthy")
            color = cv::Scalar(0, 255, 0); // Green
        else if (label == "diseased")
            color = cv::Scalar(0, 0, 255); // Red
        else
            color = cv::Scalar(255, 0, 0); // Blue

        cv::rectangle(
            colorImg,
            cv::Point(minX, minY),
            cv::Point(maxX, maxY),
            color,
            2
        );
    }

    std::string outputFileName = "nodes_bounding_boxes_labeled_0.3_IoU.png";
    //add minsize and maxsize to the filename
//    outputFileName = "nodes_bounding_boxes_" + std::to_string(minsize) + "_" + std::to_string(maxsize) + ".png";
    cv::imwrite(outputFileName, colorImg);

//    int count = 0;
//    for (int i = 0; i < tree._curSize; i++) {
//        if (tree._node[i].area > 5000) {
//            std::cout << "Bounding Box of node " << i << ":\n";
//            std::cout << "MinX: " << tree._node[i].minX << ", MinY: " << tree._node[i].minY << "\n";
//            std::cout << "MaxX: " << tree._node[i].maxX << ", MaxY: " << tree._node[i].maxY << "\n";
//
//            ImgIdx bboxWidth = tree._node[i].maxX - tree._node[i].minX + 1;
//            ImgIdx bboxHeight = tree._node[i].maxY - tree._node[i].minY + 1;
//
//            std::cout << "Width: " << bboxWidth << ", Height: " << bboxHeight << "\n";

            // Clip to image bounds (safe)
//            cv::Rect roi(tree._node[i].minX, tree._node[i].minY, bboxWidth, bboxHeight);
//            roi = roi & cv::Rect(0, 0, width, height);  // Ensure within image

//            cv::rectangle(image, roi, cv::Scalar(0, 0, 255), 2);  // Draw red rectangle

//            if (tree._node[i].minX != 0 || tree._node[i].minY != 0){
//                std::cout << "FOUND" << std::endl;
//                std::cout << "minX: " << tree._node[i].minX << "minY: " << tree._node[i].minY << std::endl;
//            }
//        }
//    }

    // Save the image with ROIs drawn
//    cv::imwrite("image_with_rois.png", image);
//    std::cout << "Saved image_with_rois.png" << std::endl;


    //    // Normalize features using min-max normalization
//    if (!featureVectors.empty()) {
//        // Find min and max values for each feature
//        double minArea = featureVectors[0].area, maxArea = featureVectors[0].area;
//        double minCompactness = featureVectors[0].compactness, maxCompactness = featureVectors[0].compactness;
//        double minAvgRed = featureVectors[0].avgRed, maxAvgRed = featureVectors[0].avgRed;
//        double minAvgGreen = featureVectors[0].avgGreen, maxAvgGreen = featureVectors[0].avgGreen;
//        double minAvgBlue = featureVectors[0].avgBlue, maxAvgBlue = featureVectors[0].avgBlue;
//
//        for (const auto& feature : featureVectors) {
//            minArea = std::min(minArea, feature.area);
//            maxArea = std::max(maxArea, feature.area);
//            minCompactness = std::min(minCompactness, feature.compactness);
//            maxCompactness = std::max(maxCompactness, feature.compactness);
//            minAvgRed = std::min(minAvgRed, feature.avgRed);
//            maxAvgRed = std::max(maxAvgRed, feature.avgRed);
//            minAvgGreen = std::min(minAvgGreen, feature.avgGreen);
//            maxAvgGreen = std::max(maxAvgGreen, feature.avgGreen);
//            minAvgBlue = std::min(minAvgBlue, feature.avgBlue);
//            maxAvgBlue = std::max(maxAvgBlue, feature.avgBlue);
//        }
//
//        // Normalize each feature vector
//        for (auto& feature : featureVectors) {
//            feature.area = (maxArea != minArea) ? (feature.area - minArea) / (maxArea - minArea) : 0.0;
//            feature.compactness = (maxCompactness != minCompactness) ? (feature.compactness - minCompactness) / (maxCompactness - minCompactness) : 0.0;
//            feature.avgRed = (maxAvgRed != minAvgRed) ? (feature.avgRed - minAvgRed) / (maxAvgRed - minAvgRed) : 0.0;
//            feature.avgGreen = (maxAvgGreen != minAvgGreen) ? (feature.avgGreen - minAvgGreen) / (maxAvgGreen - minAvgGreen) : 0.0;
//            feature.avgBlue = (maxAvgBlue != minAvgBlue) ? (feature.avgBlue - minAvgBlue) / (maxAvgBlue - minAvgBlue) : 0.0;
//        }
//
//        std::cout << "Features normalized using Min-Max normalization" << std::endl;
//    }

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
