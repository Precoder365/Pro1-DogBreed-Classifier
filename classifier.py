import ast
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
from torch import __version__
import openai
import os
import base64
import requests

openai.api_key = os.getenv("OPENAI_API_KEY")

resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

models = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16}

# Obtain ImageNet labels
with open('imagenet1000_clsid_to_human.txt') as imagenet_classes_file:
    imagenet_classes_dict = ast.literal_eval(imagenet_classes_file.read())

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def classifier(img_path, model_name):
    if model_name == "openai":
        base64_image = encode_image(img_path)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
        }

        prompt = f"Given an image, identify if it is a dog. If it is, return the breed name according to the ImageNet dataset labels. If it's not a dog, return the most likely category from the ImageNet labels. Only the name of the breed or category should be returned.:\n![image](data:image/jpeg;base64,{base64_image})"

        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        if response.status_code == 200:
            openai_pred = response.json()["choices"][0]["message"]["content"].strip()
            return openai_pred
        else:
            return f"Error: {response.status_code}, {response.text}"
    else:
        # Use pretrained models (resnet, alexnet, vgg)
        # Load the image
        img_pil = Image.open(img_path)

        # Define transforms
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Preprocess the image
        img_tensor = preprocess(img_pil)
        
        # Resize the tensor (add dimension for batch)
        img_tensor.unsqueeze_(0)
        
        # Wrap input in variable - no longer needed for v 0.4 & higher
        pytorch_ver = __version__.split('.')
        
        # PyTorch versions 0.4 & higher - Variable deprecated
        if int(pytorch_ver[0]) > 0 or int(pytorch_ver[1]) >= 4:
            img_tensor.requires_grad_(False)
        
        # PyTorch versions less than 0.4 - uses Variable because not-deprecated
        else:
            data = Variable(img_tensor, volatile=True) 

        # Apply model to input
        model = models[model_name]

        # Puts model in evaluation mode
        model = model.eval()
        
        # Apply data to model - adjusted based on version
        if int(pytorch_ver[0]) > 0 or int(pytorch_ver[1]) >= 4:
            output = model(img_tensor)
        else:
            output = model(data)

        # Return index corresponding to predicted class
        pred_idx = output.data.numpy().argmax()

        return imagenet_classes_dict[pred_idx]
