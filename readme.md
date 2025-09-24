# Uni-Reaction

Official Implementation of paper:

[A Unified Chemical Reaction Representation Learning Framework for Reaction Condition Recommendation and Performance Prediction](https://arxiv.org/abs/2411.17629)

## Environment

## Data, Checkpoints and Results

All the data used for model training and the checkpoints we have trained can be found through this [link](https://drive.google.com/drive/folders/1xruOB2ooBOaNGHQBT8_2ebGnwKMI2veG?usp=sharing). Below is a brief description of the format.

### Data

- **USPTO-Condition:** The folder contains two files. The `csv` file stores the training data, and the `json` file stores the reagent-index lookup table for the dataset.
- **USPTO-500MT:** The folder contains five `json` files. In addition to the training, validation, and test sets, we also provide a list of all reagents involved in the training labels and a list of chemical tokens.
- **Buchwald-Hartwig cross-coupling reaction:** The folder contains ten random splits and four OOD splits. Each split folder has three files representing the training, validation, and test sets.
- **radical C–H functionalization:** (corresponding to the `hx` folder) The folder contains ten groups of random splits. The names of their subfolders indicate the random seeds used for the splits. Each split folder has three files representing the training, validation, and test sets.
- **chiral phosphoric acid-catalyzed thiol addition:** (corresponding to the `dm` folder) The folder contains ten groups of random splits. The names of their subfolders indicate the random seeds used for the splits. Each split folder has three files representing the training, validation, and test sets.

### Checkpoints

- **USPTO-Condition:** The `model.pth` file represents the model weights, and the `pkl` file stores the reagent-index lookup table for easy metric calculation and validation.
- **USPTO-500MT:** The `model.pth` file represents the model weights, and the `pkl` file stores the reagent-index lookup table for easy metric calculation and validation.
- **Buchwald-Hartwig cross-coupling reaction:** The folder contains checkpoints for each split. For OOD splits, we provide model weights trained with different random seeds.
- **radical C–H functionalization:** (corresponding to the `hx` folder) Each split folder contains the model weights corresponding to that split.
- **chiral phosphoric acid-catalyzed thiol addition:** (corresponding to the `dm` folder) Each split folder contains the model weights corresponding to that split.

All checkpoints use the default parameters specified in the inference/training scripts. For the **Buchwald-Hartwig cross-coupling reaction** and **chiral phosphoric acid-catalyzed thiol addition** datasets, we provide two versions: one using a pretrained condition encoder and the other trained from scratch. You need to switch the model config as needed during inference. The model config is placed in the `config` folder.

### Results

We have uploaded the inference results for two sets of experiments that are relatively time-consuming, namely **USPTO-Condition** and **USPTO-500MT**, in the form of `json`.

## Data Preprocess

### USPTO-Condition

The Parrot project does not add atom-mapping for the reactions in USPTO-Condition. To address this, we need to use rxnmapper to add atom-mapping to USPTO-Condition. **To run the preprocess script, you need to create an environment that supports the operation of rxnmapper, following [its homepage](https://github.com/rxn4chemistry/rxnmapper).**

After obtaining the processed dataset from [Parrot](https://github.com/wangxr0526/Parrot), you can run the following commands to process the data. First, change the working directory:

```Shell
cd data_process_script/uspto_condition
```

Then, you need to execute the following script to add atom-mapping. Here, `$num_gpus` represents the number of GPUs required to add atom-mapping, `$batch_size` represents the batch size for rxnmapper inference, and `$share_size` represents the coarse-grained chunk size for the dataset. The script's execution logic is to divide the dataset into several chunks, write them into temporary files, and call another script to perform inference for different chunks on different devices. `$input_file` is the path to the CSV file containing the processed data from the Parrot project, and `$output_file` is the path where the atom-mapped dataset will be output.

```Shell
python add_am_uspto_condition.py --num_gpus $num_gpus --batch_size $batch_size --share_size $share_size --input_file $input_file --output_file $output_file
```

Next, we also need to canonicalize the labels of the dataset and generate a reagent-index lookup table. `$input_file` is the path to the atom-mapped data, and `$vocab_path` is the path to the lookup table. If left blank, we will store the lookup table in `label_to_idx.json` in the same directory as the input file by default.

```Shell
python cano_output.py --file_path $input_file --vocab_path $vocab_path
```

Of course, we also provide preprocessed datasets (see [**Data, Checkpoints and Results**](##Data, Checkpoints and Results)).

### USPTO-500MT

Due to the presence of ionic compounds, a reagent may contain multiple components separated by '.', and to facilitate model training, these reagents should have a fixed order. Therefore, we have reorganized the sample labels for the dataset. After obtaining the raw data from T5Chem, you need to modify `DATA_DIR` in the two `py` files under the path `data_process_script/uspto_500mt` to the path of the raw data, and then run the following commands in sequence:

```Shell
cd data_process_scripts/uspto_500mt
python regenerate_reagent_uspto_500_mt.py
python generate_token_list_uspto_500_mt.py
```

Then you can obtain the processed dataset from the `data_process_scripts/uspto_500mt` folder. Of course, we also provide preprocessed datasets (see [**Data, Checkpoints and Results**](##Data, Checkpoints and Results)).

### Buchwald-Hartwig cross-coupling reaction

Since all the reactions in this dataset follow the same reaction template, we use a rule-based approach to generate atom-mapping for the reactions. We first obtain the raw data from [YieldBert](https://github.com/rxn4chemistry/rxn_yields/tree/master/data/Buchwald-Hartwig). Then, run the following command to get the data. Here, `$input_file` represents the path to the Excel file containing the raw data, and `$output_folder` is used to store the processed dataset. The script will output 10 random splits and four out-of-distribution (OOD) splits, which are aligned with the [YieldBert](https://github.com/rxn4chemistry/rxn_yields/tree/master/data/Buchwald-Hartwig) and other baselines.

```shell
python data_process_script/process_cn_yield.py --input_file $input_file --output_dir $output_folder
```

Of course, we also provide preprocessed datasets (see [**Data, Checkpoints and Results**](##Data, Checkpoints and Results)).

### radical C–H functionalization

The data needs to be processed with rxnmapper to add atom-mapping and to perform random dataset splitting. Our splitting script is placed in the corresponding folder in the uploaded data. After downloading the data, change the working directory to the corresponding folder and then run

```Shell
python xxxx.py
```

(or run `xxxx.ipynb`)

### chiral phosphoric acid-catalyzed thiol addition

The data needs to be processed with rxnmapper to add atom-mapping and to perform random dataset splitting. Our splitting script is placed in the corresponding folder in the uploaded data. After downloading the data, change the working directory to the corresponding folder and then run

```Shell
python xxxx.py
```

(or run `xxxx.ipynb`)

## Training

## Inference and Evaluation

