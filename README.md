# README.md

## Facial Expression Classification using LeNet Architecture with Various Activation Functions

This repository contains the code to replicate experiments evaluating the impact of different activation functions—Sigmoid, ReLU, and SELU—on the performance of Convolutional Neural Networks (CNNs) in classifying facial expressions using the Real-World Affective Faces Database (RAF-DB). The main training script `train.py` trains a model using a specified activation function and saves the performance metrics and model weights.

## Table of Contents
1. [Requirements](#requirements)
2. [Dataset](#dataset)
3. [Usage](#usage)
4. [Training](#training)
5. [Results](#results)
6. [References](#references)

## Requirements

To run the code in this repository, you will need the following:

- Python 3.7 or higher
- PyTorch 1.8 or higher
- torchvision 0.9 or higher
- pandas 1.2 or higher
- tqdm 4.60 or higher

You can install the required packages using pip:

```sh
pip install torch torchvision pandas tqdm
```

## Dataset

The Real-World Affective Faces Database (RAF-DB) is used for this experiment. It contains 15,000 facial images tagged with basic or compound expressions. The dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets) and should be organized as follows:

```
data/
  train/
    surprised/
    fearful/
    disgusted/
    happy/
    sad/
    angry/
    neutral/
  test/
    surprised/
    fearful/
    disgusted/
    happy/
    sad/
    angry/
    neutral/
```

## Sample Images

![Sample Images](https://drive.google.com/uc?id=1nc47d7fUw0e8DsR6O3aoi3qW07-3SvXC)

## Usage

### Configuration

Edit the `config.py` file to set the parameters for the training. Here are the key parameters:

```python
EPOCHS = 20
LEARNING_RATE = 0.0003
BATCH_SIZE = 32
MODEL_DIR = './models'
```

### Running the Training Script

To train a model using a specific activation function, run the `train.py` script with the desired activation function as an argument. The available options are `sigmoid`, `relu`, and `selu`. For example, to train a model using the ReLU activation function, use the following command:

```sh
python train.py relu
```

If no activation function is specified, the default `sigmoid` activation function will be used.

## Training

The `train.py` script performs the following steps:

1. Initializes the model with the specified activation function.
2. Trains the model using the training dataset.
3. Validates the model using the validation dataset.
4. Saves the best model based on validation accuracy.
5. Logs the training and validation loss and accuracy for each epoch.

### Sample Training Command

```sh
python train.py relu
```

### Logging and Model Saving

During training, metrics are logged, and the best model is saved in the directory specified by `MODEL_DIR` in `config.py`. Training and validation metrics are saved as a pickle file in the `./metrics` directory.

## Results

The results of the experiments, including training loss, validation loss, training accuracy, and validation accuracy for each epoch, are saved in a pickle file in the `./metrics` directory. The best model weights are saved in the directory specified by `MODEL_DIR`.

## References

This experiment is part of the research work presented in:

Gbenle, Zainab. "Evaluating Activation Functions for Facial Expression Classification Using LeNet Architecture." Manchester Metropolitan University, 2023. 

For more details, refer to the full report available in the `https://drive.google.com/file/d/1EUvPIFAuqA8MpglhFo85wIV-lMt_-KTF/view?usp=sharing` file.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

Happy Training!