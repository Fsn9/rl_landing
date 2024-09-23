#!/usr/bin/python

from detector import network_modules
from detector import squeeze_exciter
from controller import DQN
import torch
import os
from torchvision.io import read_image
from torchvision import transforms
from time import time
import rclpy

""" Hyperparameters """
UM_MODALITIES = 3
EMBED_DIM = 16
LEARNING_RATE = 3e-4
IMAGE_SIZE = 160

class Lander():
    """
    
    """
    def __init__(self):
        
        self.detector = network_modules.VisionTransformer(embed_dim=EMBED_DIM,
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
        

        self.agent = DQN(controller_name='ros2_controller', model='ros2_controller', input_size=self.detector.flatten_size, train=True, test=False, resume=False)

    def connect_networks(self, x):
        #dois modos: congelar detetor, ou treinar detetor


        self.x = self.detector.forward(x, Lander_Class=True)

        res = self.agent.train(self.x.flatten())
        print(res)

    #metodo para teste inferencia



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
