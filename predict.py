import torch
from PIL import Image
import torchvision.transforms as transforms
from main import CIFAR_CNN

classes = (
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
)

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

model = CIFAR_CNN()

model.load_state_dict(
    torch.load("models/cifar_cnn.pth")
)

model.eval()

image_path = "sample.jpg"

image = Image.open(image_path).convert("RGB")

image = transform(image)

image = image.unsqueeze(0)

with torch.no_grad():

    outputs = model(image)

    _, predicted = torch.max(outputs, 1)

print("Predicted Class:",
      classes[predicted.item()])
