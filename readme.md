# Uni-Reaction

Official Implementation of paper:

[A Unified Chemical Reaction Representation Learning Framework for Reaction Condition Recommendation and Performance Prediction](https://arxiv.org/abs/2411.17629)

## Environment

Data and Checkpoints

## Data Preprocess

### USPTO-Condition

The Parrot project does not add atom-mapping for the reactions in USPTO-Condition. To address this, we need to use rxnmapper to add atom-mapping to USPTO-Condition. To run the preprocess script, you need to create an environment that supports the operation of rxnmapper, following [its homepage](https://github.com/rxn4chemistry/rxnmapper).

After obtaining the processed dataset from [Parrot](https://github.com/wangxr0526/Parrot), you can run the following commands to process the data. First, change the working directory:

```Shell
cd data_process_script\uspto_condition
```

Then, you need to execute the following script to add atom-mapping. Here, `$num_gpus` represents the number of GPUs required to add atom-mapping, `$batch_size` represents the batch size for rxnmapper inference, and `$share_size` represents the coarse-grained chunk size for the dataset. The script's execution logic is to divide the dataset into several chunks, write them into temporary files, and call another script to perform inference for different chunks on different devices. `$input_file` is the path to the CSV file containing the processed data from the Parrot project, and `$output_file` is the path where the atom-mapped dataset will be output.

```Shell
python add_am_uspto_condition.py --num_gpus $num_gpus --batch_size $batch_size --share_size $share_size --input_file $input_file --output_file $output_file
```

Next, we also need to canonicalize the labels of the dataset and generate a reagent-index lookup table. `$input_file` is the path to the atom-mapped data, and `$vocab_path` is the path to the lookup table. If left blank, we will store the lookup table in `label_to_idx.json` in the same directory as the input file by default.

```Shell
python cano_output.py --file_path $input_file --vocab_path $vocab_path
```

### USPTO-500MT

### Buchwald-Hartwig cross-coupling reaction

### radical C–H functionalization

### chiral phosphoric acid-catalyzed thiol addition
## Training

## Inference and Evaluation

