import torch
from torchvision.transforms import v2


transform_landmarks= v2.Compose([
    v2.ToImage(),
    # v2.Grayscale(num_output_channels=1),
    v2.Resize((224,224)),
    # v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
    # v2.GaussianBlur(3, sigma=(0.1, 1.5)),

    v2.ToDtype(torch.float32,scale=True)

])

