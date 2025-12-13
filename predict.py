import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as F

from model_utils import load_trained_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument("--weights", required=True, help="Path to trained weights.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument(
        "--score-threshold", type=float, default=0.5, help="Minimum score to keep boxes."
    )
    parser.add_argument(
        "--save-path",
        help="Optional path to save an image with predicted boxes drawn.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = load_trained_model(args.weights, device=device, num_classes=2)

    image = Image.open(args.image).convert("RGB")
    tensor = F.to_tensor(image).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model([tensor])[0]

    boxes = outputs["boxes"].cpu()
    scores = outputs["scores"].cpu()
    keep = [i for i, score in enumerate(scores) if score >= args.score_threshold]

    if not keep:
        print("No detections above threshold.")
    else:
        for i in keep:
            box = boxes[i].tolist()
            score = scores[i].item()
            print(f"Box: {box} | score: {score:.3f}")

    if args.save_path:
        draw = ImageDraw.Draw(image)
        for i in keep:
            box = boxes[i].tolist()
            draw.rectangle(box, outline="red", width=3)
            draw.text((box[0], box[1]), f"{scores[i]:.2f}", fill="red")
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(args.save_path)
        print(f"Saved visualization to {args.save_path}")


if __name__ == "__main__":
    main()
