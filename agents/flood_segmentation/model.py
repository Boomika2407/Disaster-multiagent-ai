import torch
import segmentation_models_pytorch as smp

def build_model(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=10):
    """
    Builds a U-Net model with the specified encoder using segmentation_models_pytorch.
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )
    return model

if __name__ == "__main__":
    # Smoke test the model creation
    model = build_model()
    print("Model built successfully.")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 512, 512)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
