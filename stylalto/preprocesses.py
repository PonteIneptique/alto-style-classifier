import torchvision.transforms as transforms


center = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]
)

resize = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]
)

resize_center = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(34),  # Larger side resized to 56, likely height
        transforms.CenterCrop((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]
)

PREPROCESSES = {
    "center": center,
    "resize": resize,
    "resize-center": resize_center
}
