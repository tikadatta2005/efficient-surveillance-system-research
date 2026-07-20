from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from Architecture import Architecture
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import io


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


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def add_conv_layers(model, layers=1, skip_pool=0):

    in_channels = 3
    out_channels = 8
    size = 64

    skip_pool = skip_pool + 1

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

        if (layer + 1) % skip_pool == 0:
            model.add(
                nn.MaxPool2d(2,2)
            )
            size //= 2


        if layer < layers - 1:
            in_channels = out_channels
            out_channels *= 2


    return out_channels, size

model = Architecture()


out_channels, size = add_conv_layers(
    model,
    layers=8,
    skip_pool=1     # IMPORTANT: same as training
)


n_in = out_channels * size * size


model.add(
    nn.Flatten(),

    nn.Linear(n_in,128),

    nn.ReLU(),

    nn.Linear(128,3)
)


checkpoint = torch.load(
    "./exp/e8_8_ext_e7_momentum/models/epoch_94.pt",
    map_location=device
)


model.load_state_dict(
    checkpoint,
    strict=True
)


model = model.to(device)

model.eval()


transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])



classes = {
    0:"Armed",
    1:"Fight",
    2:"Other"
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

        pred = torch.argmax(
            output,
            dim=1
        ).item()


    result = classes[pred]


    print("Prediction:", result)


    return {
        "prediction": result,
        "class_id": pred
    }