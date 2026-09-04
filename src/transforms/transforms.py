import torch
from torchvision.transforms import v2


transform_distance= v2.Compose([
    v2.ToImage(),
    v2.Grayscale(num_output_channels=1),
    v2.Resize((224,224)),
    v2.RandomRotation(degrees=50),
    v2.ToDtype(torch.float32,scale=True)

])
transform_rotation= v2.Compose([
    v2.ToImage(),
    v2.Grayscale(num_output_channels=1),
    v2.Resize((224,224)),
    v2.ToDtype(torch.float32,scale=True)

])

