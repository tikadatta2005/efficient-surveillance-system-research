from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from Architecture import Architecture
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import io


# Loss (not required for inference, kept if needed)
criterion = nn.CrossEntropyLoss()


def add_conv_layers(model, layers=1, skip_pool=0):

    in_channels = 3
    out_channels = 8

    size = 64

    skip_pool = skip_pool + 1

    total_conv_params = 0

    for layer in range(layers):

        model.add(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        total_conv_params += (
            ((3 * 3 * in_channels) + 1) * out_channels
        ) + (2 * out_channels)


        if (layer + 1) % skip_pool == 0:
            model.add(
                nn.MaxPool2d(2, 2)
            )
            size = size // 2


        if layer < layers - 1:
            in_channels = out_channels
            out_channels = out_channels * 2


    return total_conv_params, out_channels, size



app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Load architecture
model = Architecture()

add_conv_layers(
    model,
    layers=8,
    skip_pool=2
)


# Load trained weights
checkpoint = torch.load(
    "./exp/e8_8_ext_e7_momentum/models/epoch_94.pt",
    map_location=device
)

model.load_state_dict(checkpoint)


model = model.to(device)

model.eval()


# Same preprocessing as training
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


classes = {
    0: "Armed",
    1: "Fight",
    2: "Other"
}



@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    image_bytes = await file.read()


    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")


    image = transform(image)

    image = image.unsqueeze(0).to(device)


    with torch.no_grad():

        output = model(image)

        prediction = torch.argmax(
            output,
            dim=1
        ).item()


    print(
        "Prediction:",
        prediction,
        classes[prediction]
    )


    return {
        "prediction": classes[prediction],
        "class_id": prediction
    }