import torch
from PIL import Image
from torchvision import transforms
from torch import nn
from typing import Tuple, List

def process_image(image_path: str) -> torch.Tensor:
    """
    Process an image file into a tensor suitable for model input.

    Args:
        image_path (str): Path to the image file.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    img = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = preprocess(img)
    return img_tensor

def predict(image_path: str, model: nn.Module, device: torch.device, topk: int = 5) -> Tuple[List[float], List[int]]:
    """
    Predict the top K classes for a given image using a trained model.

    Args:
        image_path (str): Path to the image file.
        model (nn.Module): Trained model for prediction.
        device (torch.device): Device to run the prediction on.
        topk (int, optional): Number of top classes to return. Defaults to 5.

    Returns:
        Tuple[List[float], List[int]]: Probabilities and class indices of the top K predictions.
    """
    model.eval()
    img = process_image(image_path).unsqueeze(0)
    with torch.no_grad():
        output = model(img.to(device))
    ps = torch.exp(output)
    top_p, top_class = ps.topk(topk, dim=1)
    return top_p.cpu().numpy()[0], top_class.cpu().numpy()[0]
