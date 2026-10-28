"""Export the prototype image embedding model as an external ONNX model pack."""

import argparse
import json
from pathlib import Path

import torch
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    ResNet18_Weights,
    mobilenet_v3_large,
    resnet18,
)


PROFILE_CONFIGS = {
    "speed": {
        "model_dir": "mobilenet_v3_large_embedding_v1",
        "model_id": "mobilenet-v3-large-imagenet-embedding-v1",
        "dimensions": 960,
    },
    "balanced": {
        "model_dir": "resnet18_embedding_v1",
        "model_id": "resnet18-imagenet-embedding-v1",
        "dimensions": 512,
    },
    "accuracy": {
        "model_dir": "dinov2_vits14_embedding_v1",
        "model_id": "dinov2-vits14-embedding-v1",
        "dimensions": 384,
    },
}


class DINOEmbeddingWrapper(torch.nn.Module):
    """Expose only the image tensor so optional DINO training masks stay internal."""

    def __init__(self, backbone: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the AI embedding model pack")
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_CONFIGS),
        default="balanced",
        help="Built-in model profile to export",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def create_embedding_model(profile: str):
    """Create one pretrained encoder with a normalized embedding output."""
    if profile == "speed":
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        model.classifier = torch.nn.Identity()
        return model
    if profile == "accuracy":
        backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        return DINOEmbeddingWrapper(backbone)
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    return model


def main() -> None:
    args = parse_args()
    profile_config = PROFILE_CONFIGS[args.profile]
    output_dir = args.output_dir or (
        Path.home()
        / "image_classifier"
        / "ai_models"
        / profile_config["model_dir"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.onnx"

    model = create_embedding_model(args.profile)
    model.eval()
    example = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    torch.onnx.export(
        model,
        example,
        model_path,
        input_names=["images"],
        output_names=["embeddings"],
        dynamic_axes={
            "images": {0: "batch"},
            "embeddings": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    manifest = {
        "schema_version": 1,
        "model_id": profile_config["model_id"],
        "profile": args.profile,
        "task": "image_embedding",
        "runtime": "onnxruntime" if args.profile == "accuracy" else "opencv",
        "model_file": model_path.name,
        "input": {
            "name": "images",
            "width": 224,
            "height": 224,
            "layout": "NCHW",
            "color": "RGB",
            "dtype": "float32",
            "resize": "letterbox",
            "padding_rgb": [124, 116, 104],
            "scale": 0.00392156862745098,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "output": {
            "name": "embeddings",
            "dimensions": profile_config["dimensions"],
            "normalize": "l2",
        },
        "classifier": {
            "type": "balanced_knn",
            "neighbors_per_class": 5,
            "spatial_color_weight": 0.6,
            "uncertain_margin": 0.03,
            "removal_uncertain_margin": 0.05,
            "removal_min_similarity": 0.65,
            "minimum_samples_per_class": 1,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"模型: {model_path} ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"清单: {manifest_path}")


if __name__ == "__main__":
    main()
