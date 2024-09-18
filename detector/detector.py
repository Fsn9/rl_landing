from network_modules import VisionTransformer
import torch
import os
from torchvision.io import read_image
from torchvision import transforms
from time import time

""" Hyperparameters """
NUM_MODALITIES = 3
EMBED_DIM = 16
LEARNING_RATE = 3e-4
IMAGE_SIZE = 160

""" Device """
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)

""" Initialize model """
model = VisionTransformer(embed_dim=EMBED_DIM,
                          hidden_dim=512,
                          num_heads=8,
                          num_layers=6,
                          patch_size=16,
                          num_channels=1,
                          num_patches=256,
                          num_classes=40,
                          skip_mult=True,
                          dropout=0.2,
                          input_size=IMAGE_SIZE,
                          lr=LEARNING_RATE,
                          )
model_state_dict = model.state_dict()

""" Load model """
MODEL_PATH = './model.ckpt'
new_model = torch.load(MODEL_PATH)

for key1, key2 in zip(new_model['state_dict'],model.state_dict()): 
  model_state_dict[key2] = new_model['state_dict'][key1] # Assign parameters from new model to old model (contouring the Unexpected Key error)

model.load_state_dict(model_state_dict) # load new assigned state_dict
model.eval()
model.to(device)

""" Test image """
valid_imgs_path = './valid'
test_imgs_path ='./test'
valid_imgs = list(filter(lambda x: 'png' in x, [os.path.join(valid_imgs_path, img_name) for img_name in os.listdir(valid_imgs_path)]))
test_imgs = list(filter(lambda x: 'png' in x, [os.path.join(test_imgs_path, img_name) for img_name in os.listdir(test_imgs_path)]))

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

SHOW_IMAGE = False

inference_times = {}

for idx, img_name in enumerate(test_imgs):

  img_tensor = read_image(img_name) # read image and convert to tensor

  if SHOW_IMAGE:
    img_pil = to_pil(img_tensor)
    img_pil.show()

  img_tensor = img_tensor.float()/255 # normalize
  img_tensor = img_tensor.unsqueeze(0) # add batch dimension
  img_tensor = img_tensor.to(device) # send to cuda if available

  with torch.inference_mode():

    st = time()

    bbox, _, objectness = model(img_tensor) # infer

    inference_times[img_name] = (time() - st)

    objectness = torch.sigmoid(objectness) # convert to sigmoid because objectness is linear

    print(f'[{idx}] bbox,obj: {bbox.cpu().numpy()}, {objectness.cpu().numpy()} \n')

print('Inference times results')

for img_name in inference_times:
  print(f'{round(inference_times[img_name] * 1000, 2)} ms ({img_name})')
