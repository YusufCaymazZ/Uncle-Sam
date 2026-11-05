
import torch
from transformers import SegformerForSemanticSegmentation


MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"


def create_model(num_classes: int = 7):
    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )

    return model


if __name__ == "__main__":
    model = create_model()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)

    print("Model:", MODEL_NAME)
    print("Device:", device)
    print("Classes:", model.config.num_labels)
    print(
        "Parameters:",
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    dummy_input = torch.randn(1, 3, 512, 512, device=device)

    with torch.no_grad():
        output = model(pixel_values=dummy_input)

    print("Output shape:", output.logits.shape)
