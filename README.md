Project: Model Cross Data Validation Purpose

* Compare the performance of four helmet-detection CNNs trained on different datasets using holdout data from the second two datasets for testing.
* Demonstrate how dataset quality and preprocessing choices affect final model performance.

Datasets


| Dataset  | Source                                                                                                                          | Datasets Type  | Author             | Samples Used |
| -------- | ------------------------------------------------------------------------------------------------------------------------------- | -------------- | ------------------ | ------------ |
| Dataset0 | Great Learning:[HelmNet_ImageProcessing_Notebook](https://github.com/DanielLevenstein/HelmNet_ImageProcessing_UT_ML_Project6/)  | Classification | -                  | ~600         |
| Dataset1 | Kaggle:[on-vehicle-helmet-detection-dataset](https://www.kaggle.com/datasets/rajeevsekar21/on-vehicle-helmet-detection-dataset) | Classification | Rajeevsekar21      | ~500         |
| Dataset2 | Kaggle:[helmet-detection](https://www.kaggle.com/datasets/andrewmvd/helmet-detection)                                           | Bounding Boxes | Larxel (andrewmvd) | ~250         |
| Dataset3 | Kaggle:[hard-hat-detection](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection)                                       | Bounding Boxes | Larxel (andrewmvd) | ~2725        |

Streamlit App:

* Created a streamlit app for interactive testing.
* [DanielLevenstein/Helmet_CNN_Data_Quality_Case_Study](https://huggingface.co/spaces/DanielLevenstein/Helmet_Image_Classification)

Preprocessing

* Classification datasets (Dataset0, Dataset1)
  * All images resized to 200×200 pixels prior to training.
* Detection datasets (Dataset2, Dataset3)
  * Each bounding box is processed to create training crops.
  * For each object instance, a 100×100 pixel image was saved with the object's center positioned as close as possible to the crop center (given original bounding-box location).
* Holdout
  * A holdout set was extracted from Dataset2 and Dataset3 and saved prior to any training; used only for cross-model validation.
  * The final validation set consisted of 50 images taken from each of these holdout sets.
  * Due to the small sample size of Dataset2 it was not possible to create a larger final validation set.

Validation

* All four models are evaluated based on from a hold-out dataset from Dataset2, and Dataset3
* Because Dataset2 is significantly smaller than Dataset3 a sample of 50 images was taken from each holdout set for evaluation.
* When evaluating Model0 and Model1, the validation data is resized to 200x200 prior to evaluation.

Data Leakage

* To prevent data leakage the dataset directories are being wiped clean between runs so training data can't accidently end up in the final testing dataset.

Notes

* Dataset origins: Dataset0 = internal UT project; Dataset1–3 = publicly available Kaggle datasets.
* Purpose of consistent resizing/cropping: reduce input-size variation between models and isolate dataset-quality effects on performance.
* Because of the small size of the validation sample set performance differences of less than 2% percentage points should be disregarded.

Final Results

* Model2: 94% Accuracy
* Model3: 92% Accuracy
* Model0: 76% Accuracy
* Model1: 66% Accuracy

Conclusion

The performance of Model2 and Model3 are close with Model2 performing slightly better on the final validation set. This is a bit surprising as Model3 was a significantly bigger dataset, but when looking through the image manually, I saw a lot of inverted or mirrored images in the dataset so that could have contributed to the worse performance. Given the small size of the validation set, it's important not to read too much into the these results.

One surprising finding is after the validation data was normalized, my initial Model0 is now performing better than Model1 which was unexpected given that all the no helmet examples from that training set were super close-up face shots. I legitimately thought that the model couldn't have successfully learned anything useful given the constraints on its initial training data.

Model1 performing poorly wasn't surprising given the fact that the pictures in that dataset consisted of only two individuals and were taken by hand.
