import sys
import time
import copy
import random
import pandas as pd
import torch
from torch.autograd import Variable
from densenet import densenet169
from torchvision import transforms
from torchvision.datasets.folder import pil_loader
from torch.autograd import Variable

def get_tensor_from_image(path, transform=None):
    image = pil_loader(img_path)
    if transform: image = transform(image)
    image = image.float()
    return Variable(image, requires_grad=True).unsqueeze(0).cuda()

assert len(sys.argv) > 1

img_path = sys.argv[1]

model = densenet169(pretrained=True)
model = model.cuda()

model.load_state_dict(torch.load("./models/model_XR_WRIST.pth"))
model.eval()

transformer = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

input_image = get_tensor_from_image(img_path, transformer)

skip = sorted(random.sample(range(36807),36807-100))
img_paths = pd.read_csv("~/mura-data/MURA-v1.1/train_image_paths.csv", skiprows=skip, header=None)
img_paths = "~/mura-data/" + img_paths
import pdb; pdb.set_trace()

output = model(input_image)

print(img_path, output.item())
