# Satellite Scene Classification with Linear Methods <small>*Tony Cannistra, CSE546 Au17*</small>

## Goal
The purpose of this project is to evaluate the performance of linear classification methods on satellite image scene classification. 

## Metrics
These data come from the Planet Labs [Kaggle Competition](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space). The "best-performing" kaggle competition results (in classification accuracy) will be used as the benchmark for our results herein. Most successful methods utilized deep neural network architectures rather than single-layer linear methods as we do, so we will evaluate performance across methods. 

## Approach:

1. **Data Acquisition and Access**:
    1. These data were downloaded from the Kaggle competition and stored on an AWS EC2 Instance EBS volume. 
2. **Feature Extraction:**
    1. Following from [Coates et al.](https://cs.stanford.edu/~acoates/papers/coatesleeng_aistats_2011.pdf) we utilize the following pipeline for extracting meaningful features from satellite imagery:
        1. Extract random patches from training images. 
        2. Apply pre-processing to each patch (whitening, normlization)
        3. Learn a feature mapping using an unsupervised learning algorithm (here: [K-means](http://en.wikipedia.org/wiki/k-means)). 
        4. Extract features from equally spaced sub-patched covering each input image. 
        5. Pool features together over regions of input images (*parameterizable*)
3. **Linear Classification**:
    1. We will train a linear classifier to predict the given image labels using the features in the above feature extraction procedure.


