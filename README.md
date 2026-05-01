---
name: Helmet Detection Cross-Data Validation
summary: >-
  Compare generalization of four CNNs trained on different helmet/hard-hat
  datasets using holdout data from two bounding-box datasets; demonstrate
  effects of dataset quality and preprocessing on model performance.
description: >-
  Four CNN classifiers trained separately on Dataset0 (classification), Dataset1
  (classification), Dataset2 (bounding-box → crops), and Dataset3 (bounding-box
  → crops). Models evaluated on holdout sets from Dataset2 and Dataset3 to
  measure cross-dataset generalization. Preprocessing included resizing
  classification images to 200x200 and cropping bounding-box features to 100x100
  after removing features smaller than 50x50.
license: gfdl
language:
- en
metrics:
- accuracy 91
- precision 95
- recall 94
- f1 94
---
# Project: Model Cross-Data Validation Purpose

* Compare the performance of four helmet-detection CNNs trained on different datasets using holdout data from the second two datasets for testing.
* Demonstrate how dataset quality and preprocessing choices affect final model performance.

## Datasets

In this project each of the four models was trained off an initial dataset, and the result of each model was tested on a cross validation dataset with holdout data from Dataset2, and Dataset3.

| Dataset  | Source                                                                                                                          | Datasets Type  | Samples Used | Model Accuracy |
| -------- | ------------------------------------------------------------------------------------------------------------------------------- | -------------- | ------------ | -------------- |
| Dataset0 | Great Learning:[HelmNet_ImageProcessing_Notebook](https://github.com/DanielLevenstein/HelmNet_ImageProcessing_UT_ML_Project6/)  | Classification | ~600         | 76%            |
| Dataset1 | Kaggle:[on-vehicle-helmet-detection-dataset](https://www.kaggle.com/datasets/rajeevsekar21/on-vehicle-helmet-detection-dataset) | Classification | ~500         | 66%            |
| Dataset2 | Kaggle:[helmet-detection](https://www.kaggle.com/datasets/andrewmvd/helmet-detection)                                           | Bounding Boxes | ~250         | 94%            |
| Dataset3 | Kaggle:[hard-hat-detection](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection)                                       | Bounding Boxes | ~3000        | 92%            |

**This table shows the source of each dataset used the dataset type and the number of sample features used after preprocessing.**
**Model Accuracy was calculated based on a holdout dataset from Dataset2, and Dataset3.**

* [Dataset0](https://github.com/DanielLevenstein/HelmNet_ImageProcessing_UT_ML_Project6/): This dataset is focused on workplace helmet detection, primarily using database images of workers. The "no
  helmet" class includes close-up face pictures of individuals without helmets.
* [Dataset1](https://www.kaggle.com/datasets/rajeevsekar21/on-vehicle-helmet-detection-dataset): A collection of images of motorcycle helmets that were manually captured by a college student using a
  consistent camera angle and lighting condition.
* [Dataset2](https://www.kaggle.com/datasets/andrewmvd/helmet-detection): A bounding box dataset designed for detecting helmets on motorcycles in traffic scenarios, emphasizing
  varied environments and perspectives.
* [Dataset3](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection): This is a bounding box dataset developed for detecting workplace hardhats. It includes diverse workplace
  scenarios and different wearer types to improve detection robustness.

## Streamlit App

This streamlit app was inspired by a model I built for a school project at UT that performed really well on training data but poorly on real-world data.

[Live Streamlit Demo](https://huggingface.co/spaces/DanielLevenstein/Helmet_Image_Classification)

## Preprocessing

* Classification datasets (Dataset0, Dataset1)
  * All images resized to 200×200 pixels prior to training.
* Detection datasets (Dataset2, Dataset3)
  * Each bounding box is processed to create training crops.
  * Features smaller than 50x50 pixels are removed from the dataset to ensure high-quality training data.
  * For each remaining feature a 100×100 pixel image was saved with the object's center positioned as close as possible to the crop center as possible.
* Holdout
  * A holdout set was extracted from Dataset2 and Dataset3 and saved prior to any training; used only for cross-model validation.
  * The final validation set consisted of 50 images taken from each of these holdout sets.
  * Due to the small sample size of Dataset2, it was not possible to create a larger final validation set.

## Validation

* All four models are evaluated based on from a hold-out dataset from Dataset2, and Dataset3.
* Because Dataset2 is significantly smaller than Dataset3, a sample of 50 images was taken from each holdout set for evaluation.
* When evaluating Model0 and Model1, the validation data is resized to 200x200 prior to evaluation.

### Data Leakage

* To prevent data leakage, the dataset directories are being wiped clean between runs so training data can't accidentally end up in the final testing dataset.

# Final Results


| Model  | Accuracy | Precision |   Recall | F1-Score |
| :----- | -------: | --------: | -------: | -------: |
| Model2 |     0.94 |   0.93617 |        1 | 0.967033 |
| Model3 |     0.91 |  0.954023 | 0.943182 | 0.948571 |
| Model0 |     0.76 |  0.890244 | 0.829545 | 0.858824 |
| Model1 |     0.74 |  0.984375 | 0.715909 | 0.828947 |

## Conclusion

The performance of Model2 and Model3 are close, with Model2 performing slightly better on the final validation set.

Model3 was a significantly bigger dataset but contained a lot of inverted or mirrored images. This might have
contributed to the worse performance. Also, most of the no-helmet samples in Dataset3 were managers, possibly leading
the model to attribute nicer clothing to that class. Given the small size of the validation set, it's important not to
read too much into these results.

After the validation data was normalized, the performance of Model0 improved enough to surpass Model1 on all metrics.
This was surprising given how limited the no-helmet class was in this dataset. Model1 consisted of only two individuals
and was taken by hand, but on initial inspection still felt like a higher-quality dataset than Dataset0.

Dataset origins and preprocessing steps played crucial roles in final model performance. Dataset0 originated from an
internal UT project, whereas Datasets1 through Dataset3 were publicly available Kaggle datasets. Consistent resizing and cropping
across datasets were key to reducing input-size variation and isolating dataset-quality effects.

It's worth noting that a difference in performance of less than 2% percentage points on the small validation dataset
should be disregarded for conclusive claims. In addition standardization of image size and removal of low quality features had an effect on the final results.
