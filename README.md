# Project: Model Cross-Data Validation Purpose

* Compare the performance of four helmet-detection CNNs trained on different datasets using holdout data from the second two datasets for testing.
* Demonstrate how dataset quality and preprocessing choices affect final model performance.

## Datasets


| Dataset  | Source                                                                                                                          | Datasets Type  | Author             | Samples Used | Model Accuracy |
| -------- | ------------------------------------------------------------------------------------------------------------------------------- | -------------- | ------------------ | ------------ | -------------- |
| Dataset0 | Great Learning:[HelmNet_ImageProcessing_Notebook](https://github.com/DanielLevenstein/HelmNet_ImageProcessing_UT_ML_Project6/)  | Classification | -                  | ~600         | 76%            |
| Dataset1 | Kaggle:[on-vehicle-helmet-detection-dataset](https://www.kaggle.com/datasets/rajeevsekar21/on-vehicle-helmet-detection-dataset) | Classification | Rajeevsekar21      | ~500         | 66%            |
| Dataset2 | Kaggle:[helmet-detection](https://www.kaggle.com/datasets/andrewmvd/helmet-detection)                                           | Bounding Boxes | Larxel (andrewmvd) | ~250         | 94%            |
| Dataset3 | Kaggle:[hard-hat-detection](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection)                                       | Bounding Boxes | Larxel (andrewmvd) | ~2725        | 92%            |

## Streamlit App

This streamlit app was inspired by a model I build for a school project at UT that performed really well on training data but poorly on real world data.
Datasets 1 to 3 were all attempts to improve the performance of this initial model. I have since moved the cross-model validation out of my streamlit app and into a standalone validation notebook which is uploaded to GitHub.
[Live Demo](https://huggingface.co/spaces/DanielLevenstein/Helmet_Image_Classification)

## Preprocessing

* Classification datasets (Dataset0, Dataset1)
  * All images resized to 200×200 pixels prior to training.
* Detection datasets (Dataset2, Dataset3)
  * Each bounding box is processed to create training crops.
  * For each object instance, a 100×100 pixel image was saved with the object's center positioned as close as possible to the crop center (given original bounding-box location).
* Holdout
  * A holdout set was extracted from Dataset2 and Dataset3 and saved prior to any training; used only for cross-model validation.
  * The final validation set consisted of 50 images taken from each of these holdout sets.
  * Due to the small sample size of Dataset2, it was not possible to create a larger final validation set.

## Validation

* All four models are evaluated based on from a hold-out dataset from Dataset2, and Dataset3
* Because Dataset2 is significantly smaller than Dataset3, a sample of 50 images was taken from each holdout set for evaluation.
* When evaluating Model0 and Model1, the validation data is resized to 200x200 prior to evaluation.

### Data Leakage

* To prevent data leakage, the dataset directories are being wiped clean between runs so training data can't accidentally end up in the final testing dataset.

# Final Results


|   | Model  | Accuracy | Precision | Recall   | F1-Score |
| - | ------ | -------- | --------- | -------- | -------- |
| 2 | Model2 | 0.94     | 0.93617   | 1        | 0.967033 |
| 3 | Model3 | 0.92     | 0.954545  | 0.954545 | 0.954545 |
| 0 | Model0 | 0.76     | 0.890244  | 0.829545 | 0.858824 |
| 1 | Model1 | 0.66     | 0.982143  | 0.625    | 0.763889 |

## Conclusion

The performance of Model2 and Model3 are close with Model2 performing slightly better on the final validation set.

Model3 was a significantly bigger dataset, but contained a lot of inverted or mirrored images. This might have contributed to the worse performance. Also most of the no-helmet samples in Dataset3 were managers so it is possible the model learned to attribute nicer clothing to that class. Given the small size of the validation set, it's important not to read too much into the these results.

After the validation data was normalized, the performance of Model0 improved enough surpassed Model1 on all metrics. This was surprising given how limited the no-helmet class was in this dataset. Model1 consisted of only two individuals and were taken by hand but on initial inspection still felt like a higher quality dataset than Dataset0.


### Notes

* Dataset origins: Dataset0 = internal UT project; Dataset1–3 = publicly available Kaggle datasets.
* Purpose of consistent resizing/cropping: reduce input-size variation between models and isolate dataset-quality effects on performance.
* Because of the small size of the validation sample set performance differences of less than 2% percentage points should be disregarded.
