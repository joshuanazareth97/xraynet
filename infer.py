import sys
import time
import copy
import pandas as pd
import torch
from torch.autograd import Variable
from densenet import densenet169
from torchvision import transforms
from torchvision.datasets.folder import pil_loader
from torch.autograd import Variable

assert len(sys.argv) > 1

img_path = sys.argv[1]

model = densenet169(pretrained=True)
model = model.cuda()

model.load_state_dict(torch.load("./models/model_XR_WRIST.pth"))

transformer = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

image = pil_loader(img_path)

final_img = transformer(image).float()
# import pdb; pdb.set_trace()
final_img = Variable(final_img, requires_grad=True)

output = model(final_img.unsqueeze(0).cuda())

print(output)