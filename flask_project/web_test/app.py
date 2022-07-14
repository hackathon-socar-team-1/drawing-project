import io
from flask import render_template
import pix2pix

import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request
from torchvision.utils import save_image


app = Flask(__name__)
model = pix2pix.GeneratorUNet().to('cpu')
weights = torch.load('./model/weights_gen.pt', torch.device('cpu'))
model.load_state_dict(weights)
model.eval()


def transform_image(image_bytes):
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
                    transforms.Resize((256,256))
])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    output = model(tensor)
    return output

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        fake_img = get_prediction(img_bytes).squeeze()
        save_image(0.5*fake_img+0.5, './static/fake_img.png')
        return render_template("img.html")

if __name__ == '__main__':
    app.run()