import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import segmentation_models_pytorch as smp

# Function to load and preprocess image
def load_image(image_path):
    image = Image.open(image_path).convert("L")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return input_tensor

# Function to save output image
def save_output(output_tensor, output_path):
    output_tensor = output_tensor.squeeze().cpu().numpy()  # Remove batch dimension and convert to numpy array
    output_image = (output_tensor > 0.5).astype(np.uint8) * 255  # Threshold to binary values and scale to 0-255
    output_image = Image.fromarray(output_image)
    output_image.save(output_path)  # Save output image

# Main function
def main():
    image_path = 'data/2DTooth/A-PXI/Unlabeled/Image/A_U_0002.png'  # Replace with your image path
    checkpoint_path = 'checkpoints/best_model.pth'  # Replace with your checkpoint path
    output_path = 'output_image.jpg'  # Path to save the output image

    # Load and preprocess image
    input_tensor = load_image(image_path)

    # Load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = smp.UnetPlusPlus(encoder_name="resnet34",
                             encoder_weights=None,
                             encoder_depth=4,
                             decoder_channels=[512, 256, 128, 64],
                             in_channels=1,
                             classes=1
                             ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Run inference
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Save output image
    save_output(output_tensor, output_path)
    print(f"Output image saved to {output_path}")

if __name__ == "__main__":
    main()