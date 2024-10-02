#!/usr/bin/python

from rl_landing.detector import network_modules
from rl_landing.detector import squeeze_exciter
from rl_landing.controller import DQN, RL, Lander
import torch
import os
from torchvision.io import read_image
from torchvision import transforms
from time import time
import rclpy

""" Device """
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)

def main():
    rclpy.init(args=None)
    class_test = Lander()
    

    model_state_dict = class_test.detector.state_dict()

    """ Load model """
    MODEL_PATH = 'model.ckpt'
    new_model = torch.load(MODEL_PATH)

    for key1, key2 in zip(new_model['state_dict'],class_test.detector.state_dict()): 
        model_state_dict[key2] = new_model['state_dict'][key1] # Assign parameters from new model to old model (contouring the Unexpected Key error)

    class_test.detector.load_state_dict(model_state_dict) # load new assigned state_dict
    class_test.detector.eval()
    class_test.detector.to(device)

    """ Test image """
    valid_imgs_path = './detector/valid'
    test_imgs_path ='./detector/test'
    valid_imgs = list(filter(lambda x: 'png' in x, [os.path.join(valid_imgs_path, img_name) for img_name in os.listdir(valid_imgs_path)]))
    test_imgs = list(filter(lambda x: 'png' in x, [os.path.join(test_imgs_path, img_name) for img_name in os.listdir(test_imgs_path)]))

    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    SHOW_IMAGE = False

    for idx, img_name in enumerate(test_imgs):

        img_tensor = read_image(img_name) # read image and convert to tensor

        if SHOW_IMAGE:
            img_pil = to_pil(img_tensor)
            img_pil.show()

        img_tensor = img_tensor.float()/255 # normalize
        img_tensor = img_tensor.unsqueeze(0) # add batch dimension
        img_tensor = img_tensor.to(device) # send to cuda if available

        with torch.inference_mode():

            class_test.connect_networks(img_tensor) # infer
            print(class_test.agent.state)
            break

if __name__ == '__main__':
    main()
