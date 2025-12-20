# IRESpy_xgboost

This repo is our reimplementation of IRESpy described here: 

Wang, J., Gribskov, M. IRESpy: an XGBoost model for prediction of internal ribosome entry sites. BMC Bioinformatics 20, 409 (2019). https://doi.org/10.1186/s12859-019-2999-7

Below are the detailed steps to follow to reproduce our work. This repository contains both the raw and processed data, however, the script below will also outline any instructions that we had to go through in terms of data acquisition.

## Instructions

### 1. Clone the repo and install python dependencies

```bash
git clone https://github.coecis.cornell.edu/rbs285/IRESpy_xgboost.git
cd IRESpy_xgboost
pip install -r requirements.txt
```

### 2. Make sure that the raw data file is present

```bash
/data/raw/55k_oligos_sequence_and_expression_measurements.tab.txt
```

If the file is missing, download it from the original source: https://bitbucket.org/alexeyg-com/irespredictor/src/v2/data/

### 3. Filter the data by label

```bash
python data/filter_by_label.py
```

### 4. Filter the data by splicing score and promoter activity

```bash
python data/filter_by_splicing_and_promoter.py
```

### 5. Add IRES vs non-IRES labels

```bash
python data/add_label_column.py
```

### 6. Split the data into train/test/val

```bash
python data/create_train_test_split.py
```

### 7. Install External Dependencies

We were not able to install the external packages via typical command line tools so we have
linked the external installation guide that we followed for both ushuffle and ViennaRNA. Please
follow the provided directions for both.

ushuffle: https://github.com/guma44/ushuffle 

ViennaRNA: https://www.tbi.univie.ac.at/RNA/#


### 8. Generate k-mer features

```bash
python data/feature_generation.py
```

### 9. Generate qmfe features

```bash
python data/q_mfe_feature_generation.py
```

### 10. Train XGBoost

```bash
python model/train_xgboost.py
```

### 11. Run LIME Explanations

```bash
python lime/lime_test.py
```

### 12. Notes

Many of the earlier scripts you will be running will ovveride datasets(the .h5 files) that we have already created. Thus, these steps are optional as we have provided all of the data we used and this process was tedious for certain portions(namely the qmfe feature generation as it was handled in batches). Nonetheless the steps outlined above entirely follow our computational pipeline in the order that we worked on it.  
