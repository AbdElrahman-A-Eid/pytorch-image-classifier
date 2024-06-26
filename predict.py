import argparse
import json
import torch
from modules.utils import predict
from modules.vision_model import ImageClassifier

def main() -> None:
    """
    Main function to predict flower name from an image along with the probability of that name.
    Parses command line arguments and uses them to configure the prediction process.
    """
    parser = argparse.ArgumentParser(description='Predict flower name from an image along with the probability of that name')
    parser.add_argument('input', type=str, help='Path to input image')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('--top-k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category-names', type=str, help='Path to a JSON file mapping categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    args = parser.parse_args()

    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    classifier = ImageClassifier.load_checkpoint(args.checkpoint)
    classifier.model.to(device)
    print(f'Predicting using {"GPU" if device.type == "cuda" else "CPU"}...')

    top_p, top_class = predict(args.input, classifier.model, device, args.top_k)
    top_class = [classifier.idx_to_class[idx] for idx in top_class]

    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        top_class = [cat_to_name[str(cls)] for cls in top_class]

    print(f"Predicted classes: {top_class}")
    print(f"Class probabilities: {(top_p*100.).tolist()}")

if __name__ == "__main__":
    main()
