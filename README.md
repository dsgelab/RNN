# RNN (pytorch) model and preprocessing of Finregistry data

## Preprocessing

### 0_additional_longitudinal_features
### 1_combine
### 2_dict_codes
### 3_delete_rare_redundant_codes
### 4_demo_features

This script is for extracting fixed-over-the-time features, which cannot be included in the model longitudinally. The data inputs are from pre-processed “minimal phenotype” file and from Birth, Malformations, Social assistance, Social Hilmo and Intensive care register. Smoking status was also derived from AvoHilmo and Birth registers. The features were, continuous, ordinal, and categorical (binary + one-hot-encoded if there were more than 2 categories). Continuous and ordinal features were rescaled to be in the range 0 to 1. 

### 5 label
### 6_final_data


