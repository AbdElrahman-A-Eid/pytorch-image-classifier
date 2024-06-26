# Image Classifier Project

This project contains a PyTorch-based image classifier that can be trained on a dataset of images and then used to predict the classes of new images. The classifier supports various architectures from EfficientNet and ResNet families and includes features like early stopping and best weight restoration.

## Table of Contents

- [Image Classifier Project](#image-classifier-project)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Training the Model](#training-the-model)
    - [Predicting with the Model](#predicting-with-the-model)
  - [Project Structure](#project-structure)
  - [Contributing](#contributing)
  - [License](#license)

## Features

- **Supports EfficientNet and ResNet architectures**: Train a model using a pre-trained EfficientNet or ResNet backbone.
- **Early Stopping**: Automatically stops training when validation performance stops improving.
- **Best Weight Restoration**: Restores the model weights to the best performing epoch if early stopping is enabled.
- **GPU Support**: Option to train and predict using GPU if available.
- **Data Preprocessing**: Preprocesses images using standard transformations for training, validation, and testing.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/image-classifier.git
    cd image-classifier
    ```

2. Create and activate a virtual environment:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Training the Model

To train the model, use the `train.py` script. Here is an example:

```sh
python train.py data_directory --save-dir checkpoints --arch resnet50 --learning-rate 0.01 --hidden-units 512 --no-classes 102 --epochs 10 --gpu --early-stopping --patience 3 --monitor validation_loss --restore-best-weights
```

Arguments:
- `data_dir` (required): Directory containing the dataset.
- `--save-dir`: Directory to save checkpoints (default: `./checkpoints`).
- `--arch`: Model architecture (default: `resnet50`).
- `--learning-rate`: Learning rate for training (default: `0.01`).
- `--hidden-units`: Number of hidden units in the classifier (default: `512`).
- `--no-classes`: Number of classes in the dataset (default: `102`).
- `--epochs`: Number of epochs for training (default: `10`).
- `--gpu`: Use GPU for training.
- `--early-stopping`: Enable early stopping.
- `--patience`: Patience for early stopping (default: `3`).
- `--monitor`: Metric to monitor for early stopping (choices: `validation_loss`, `validation_accuracy`, default: `validation_loss`).
- `--restore-best-weights`: Restore best weights at the end of training.

### Predicting with the Model

To make predictions with the trained model, use the `predict.py` script. Here is an example:

```sh
python predict.py input_image_path checkpoint_path --top-k 5 --category-names cat_to_name.json --gpu
```

Arguments:
- `input` (required): Path to the input image.
- `checkpoint` (required): Path to the checkpoint file.
- `--top-k`: Return top K most likely classes (default: `5`).
- `--category-names`: Path to a JSON file mapping categories to real names.
- `--gpu`: Use GPU for inference.

## Project Structure

```bash
image-classifier/
├── modules/
│   ├── data.py             # Data loading and preprocessing
│   ├── utils.py            # Utility functions
│   ├── vision_model.py     # Model definition and training logic
├── train.py                # Script for training the model
├── predict.py              # Script for making predictions
├── requirements.txt        # Python package dependencies
└── README.md               # Project documentation
```


## Contributing

Contributions are welcome! If you have suggestions or bug fixes, please open an issue or create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
