import argparse
from modules.data import get_data
from modules.vision_model import ImageClassifier

def main() -> None:
    """
    Main function to train a new network on a dataset and save the model as a checkpoint.
    Parses command line arguments and uses them to configure the training process.
    """
    parser = argparse.ArgumentParser(description='Train a new network on a dataset and save the model as a checkpoint')
    parser.add_argument('data_dir', type=str, help='Directory of the dataset')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='resnet50', help='Architecture of the model (All EfficientNet and ResNet variants are accepted)')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning rate for training')
    parser.add_argument('--hidden-units', type=int, default=512, help='Number of hidden units in the classifier (Only applies on EfficientNet architectures)')
    parser.add_argument('--no-classes', type=int, default=102, help='Number of classes in the dataset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--early-stopping', action='store_true', help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=3, help='Patience for early stopping')
    parser.add_argument('--monitor', type=str, default='validation_loss', choices=['validation_loss', 'validation_accuracy'], help='Metric to monitor for early stopping')
    parser.add_argument('--restore-best-weights', action='store_true', help='Restore best weights at the end of training (requires --early-stopping)')
    args = parser.parse_args()

    # Get the datasets and dataloaders
    datasets, dataloaders = get_data(args.data_dir)

    # Initialize the ImageClassifier
    classifier = ImageClassifier(
        arch=args.arch,
        hidden_units=args.hidden_units,
        learning_rate=args.learning_rate,
        no_classes=args.no_classes,
        gpu=args.gpu
    )

    # Train the classifier
    classifier.train(
        datasets,
        dataloaders,
        epochs=args.epochs,
        early_stopping=args.early_stopping,
        patience=args.patience,
        monitor=args.monitor,
        restore_best_weights=args.restore_best_weights
    )

    # Save the checkpoint
    classifier.save_checkpoint(args.save_dir, classifier.epochs)

if __name__ == "__main__":
    main()
