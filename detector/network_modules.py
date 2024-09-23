import torch
import torch.nn as nn
from detector.squeeze_exciter import SE
from detector.cbam import CBAM
HEAD_TYPE = 'i'

class ConvStem(nn.Module):
  """
  Feature extraction block applied to the input image
  """
  def __init__(self,
               in_channels=1,
               base_channels=64,
               out_cnn_channel=256,
               embed_dim=256,
               img_size=[512,128],
               pretrained=True,
               flatten=True):
    super().__init__()
    
    self.img_size = [int(img_size[0] * 0.5), int(img_size[1] * 0.5)] # Resize to half size
    self.blockinput = nn.Sequential(nn.Conv2d(in_channels, base_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                                nn.BatchNorm2d(base_channels),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(base_channels, 2*base_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                                nn.BatchNorm2d(2*base_channels))                
    self.resconv0 = nn.Conv2d(in_channels, 2*base_channels, kernel_size=(1,1), stride=(1,1), padding=(0,0))
    self.pooling0 = nn.AdaptiveMaxPool2d((self.img_size[0], self.img_size[1]))
    
    self.img_size = [int(self.img_size[0] * 0.5), int(self.img_size[1] * 0.5)] # Resize to half size
    self.block1 = nn.Sequential(nn.Conv2d(2*base_channels, 4*base_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                                nn.BatchNorm2d(4*base_channels),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(4*base_channels, 8*base_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                                nn.BatchNorm2d(8*base_channels))                
    self.resconv1 = nn.Conv2d(2*base_channels, 8*base_channels, kernel_size=(1,1), stride=(1,1), padding=(0,0))                                
    self.pooling1 = nn.AdaptiveMaxPool2d((self.img_size[0], self.img_size[1]))
    
    # self.img_size = [int(self.img_size[0] * 0.5), int(self.img_size[1] * 0.5)] # Resize to half size
    # self.block2 = nn.Sequential(nn.Conv2d(2*base_channels, 4*base_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
    #                             nn.BatchNorm2d(4*base_channels),
    #                             nn.ReLU(inplace=True),
    #                             nn.Conv2d(4*base_channels, 4*base_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
    #                             nn.BatchNorm2d(4*base_channels))                    
    # self.resconv2 = nn.Conv2d(2*base_channels, 4*base_channels, kernel_size=(1,1), stride=(1,1), padding=(0,0))                               
    # self.pooling2 = nn.AdaptiveMaxPool2d((self.img_size[0], self.img_size[1]))                     

    self.blockout = nn.Conv2d(in_channels=8*base_channels, out_channels=embed_dim, kernel_size=(3,3), stride=(1,1), padding=(1,1)) # stride para (1,1) e embed_dim/3
    
    self.activation = nn.ReLU(inplace=True)
    
    self.patch_size = self.img_size
    #print(f'patch_size: {self.patch_size}')
    self.grid_size = (self.patch_size[0], self.patch_size[1])
    self.num_patches = self.grid_size[0] * self.grid_size[1]
    
    self.flatten = flatten

  def forward(self, x):
    B, C, H, W = x.shape  # B, in_channels, image_size[0], image_size[1]

    residual = torch.clone(x) # residual version of x
    out = self.blockinput(x) # (conv2d -> bn -> relu -> conv2d -> bn)(x)
    out += self.resconv0(residual) # out = blockinput(x) + resconv0(x)
    self.activation(out) # out = relu(out)
    out = self.pooling0(out) # out = maxpool(out)
     
    residual = torch.clone(out)
    out = self.block1(out)    
    out += self.resconv1(residual)
    self.activation(out)  
    out = self.pooling1(out)
    
    # residual = torch.clone(out)
    # out = self.block2(out)
    # out += self.resconv2(residual)
    # self.activation(out)  
    # out = self.pooling2(out)
    
    out = self.blockout(out)
    
    if self.flatten:
      out = out.flatten(2).transpose(1, 2)  # BCHW -> BNC
    
    return out

class VisionTransformer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, patch_size, num_patches, skip_mult=True, dropout=0.0, input_size = 160, lr = 1e-3, num_modalities = 3):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels - Number of channels of the input (3 for RGB)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of encoder blocks to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            num_patches - Maximum number of patches an image can have
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        global args # get reference to global args varriable

        self.num_modalities = num_modalities

        """ Fixed parameters for this jetson test """
        self.active_modalities = [0,1,2] # lidar (0), thermal (1) and visual (3) sensors are active
        self.base_channels = 16

        self.embed_dim = embed_dim

        self.input_size = input_size

        self.num_patches = num_patches

        # Size of each patch
        self.patch_size = patch_size
        
        # Skip connection element wise multiplication or cat
        self.skip_mult = skip_mult

        """Conv stems (backbones)"""
        # Linear Layer for Input x projection
        #self.frac_embed_dim = embed_dim // self.num_modalities
        if 0 in self.active_modalities:
          self.input_layer0 = ConvStem(in_channels=1, base_channels=self.base_channels, out_cnn_channel=512, embed_dim=self.embed_dim, img_size=[self.input_size, self.input_size], pretrained=True, flatten=False) # flatten a False. só depois de concatenar
          self.actual_patch_size = self.input_layer0.patch_size
          self.actual_num_patches = self.actual_patch_size[0] * self.actual_patch_size[1]
        if 1 in self.active_modalities:
          self.input_layer1 = ConvStem(in_channels=1, base_channels=self.base_channels, out_cnn_channel=512, embed_dim=self.embed_dim, img_size=[self.input_size, self.input_size], pretrained=True, flatten=False) # flatten a False. só depois de concatenar
          self.actual_patch_size = self.input_layer1.patch_size
          self.actual_num_patches = self.actual_patch_size[0] * self.actual_patch_size[1]
        if 2 in self.active_modalities:
          self.input_layer2 = ConvStem(in_channels=1, base_channels=self.base_channels, out_cnn_channel=512, embed_dim=self.embed_dim, img_size=[self.input_size, self.input_size], pretrained=True, flatten=False) # flatten a False. só depois de concatenar
          self.actual_patch_size = self.input_layer2.patch_size
          self.actual_num_patches = self.actual_patch_size[0] * self.actual_patch_size[1]
        
        """ Print net info """
        print('[Network info]')
        print('patch size: ', self.actual_patch_size)
        print('num patches: ', self.actual_num_patches)
        print('embed dim: ', self.embed_dim)

        """ Dimensionality reduction """
        self.dim_reduction_factor = 2
        self.max_pool_dim_reduction = nn.AdaptiveMaxPool2d((self.actual_patch_size[0] // self.dim_reduction_factor, self.actual_patch_size[1] // self.dim_reduction_factor))

        """ Flatten size """
        self.flatten_size = (self.actual_patch_size[0] // self.dim_reduction_factor)**2 * len(self.active_modalities) * self.embed_dim # p**2 * N * e_dim

        """Transformer encoder"""
        """
        DETR has:
        d_model = 512
        nhead = 8
        num_encoder_layers = 6
        dim_feedforward = 2048
        dropout = 0.1
        activation = 'relu'
        normalize_before = False
        """
        #encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, activation='gelu', batch_first=False, norm_first=True)
        #self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(embed_dim))

        """ Attention blocks """
        self.SE = SE(embed_dim*self.num_modalities)
        self.CBAM = CBAM(embed_dim*self.num_modalities)

        """Regression mlps (heads)"""
        # Flattened encoded sizes depending on head type
        if HEAD_TYPE == "a": # [cls_token] for mlp_conf and [reg_vector] for mlp_bbox
          self.mlp_bbox_input_size = self.actual_num_patches * self.embed_dim * self.num_modalities
          self.mlp_conf_input_size = embed_dim
        elif HEAD_TYPE == "b": # [cls_token, reg_vector] for both
          self.mlp_bbox_input_size = self.actual_num_patches * self.embed_dim * self.num_modalities + self.embed_dim * self.num_modalities
          self.mlp_conf_input_size = self.mlp_bbox_input_size
        elif HEAD_TYPE == "c": # [reg_vector] for both
          self.mlp_bbox_input_size = self.actual_num_patches * self.embed_dim * self.num_modalities
          self.mlp_conf_input_size = self.mlp_bbox_input_size
        elif HEAD_TYPE == "d": # [cls_token0] for mlp_conf and [cls_token1] for mlp_bbox
          self.mlp_bbox_input_size = embed_dim
          self.mlp_conf_input_size = self.mlp_bbox_input_size
        elif HEAD_TYPE == "e":
          #self.mlp_bbox_input_size = embed_dim # TODO: go back to this
          self.mlp_bbox_input_size = self.actual_num_patches * self.embed_dim * self.num_modalities + self.embed_dim * self.num_modalities
        elif HEAD_TYPE == "f":
          self.mlp_objectness_input_size = self.embed_dim * self.num_modalities
          #self.mlp_bbox_input_size = self.actual_num_patches * self.embed_dim * self.num_modalities + self.embed_dim * self.num_modalities # a) reg vector ou b) com reg_vector+cls_token, default: b
          self.mlp_bbox_input_size = self.actual_num_patches * self.embed_dim * self.num_modalities # a) reg vector ou b) com reg_vector+cls_token, default: b
        elif HEAD_TYPE == "g" or HEAD_TYPE == "h": # both heads receive cls_token
          self.mlp_objectness_input_size = self.embed_dim * self.num_modalities
          self.mlp_bbox_input_size = self.embed_dim * self.num_modalities
        elif HEAD_TYPE == "i":
          self.mlp_objectness_input_size = self.mlp_bbox_input_size = self.flatten_size # both heads receive flatten size input

        print(f'Head type is {HEAD_TYPE}')
        print(f'mlp_bbox_size: {self.mlp_bbox_input_size}')
        if HEAD_TYPE != 'e' and HEAD_TYPE != 'f' and HEAD_TYPE != 'g' and HEAD_TYPE != 'h' and HEAD_TYPE != 'i': print(f'mlp_conf_size: {self.mlp_conf_input_size}')
        if HEAD_TYPE == 'f' or HEAD_TYPE == 'g' or HEAD_TYPE == 'h' or HEAD_TYPE == 'i': print(f'mlp_objectness_size: {self.mlp_objectness_input_size}')

        # MLP head for bbox prediction
        self.box_size = 4
        self.mlp_head_bbox = nn.Sequential(
            nn.LayerNorm(self.mlp_bbox_input_size),
            nn.Linear(self.mlp_bbox_input_size, self.embed_dim * self.num_modalities),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.embed_dim * self.num_modalities),
            nn.Linear(self.embed_dim * self.num_modalities, self.box_size),
        )

        if HEAD_TYPE == 'f' or HEAD_TYPE == 'g' or HEAD_TYPE == 'h' or HEAD_TYPE == 'i':
          self.objectness_size = 1
          self.mlp_head_objectness = nn.Sequential(
              nn.LayerNorm(self.mlp_objectness_input_size),
              nn.Linear(self.mlp_objectness_input_size, self.embed_dim * self.num_modalities),
              nn.GELU(),
              nn.Dropout(dropout),
              nn.LayerNorm(self.embed_dim * self.num_modalities),
              nn.Linear(self.embed_dim * self.num_modalities, self.objectness_size),
          )

        # MLP head for confidence prediction
        if HEAD_TYPE != 'e' and HEAD_TYPE != 'f' and HEAD_TYPE != 'g' and HEAD_TYPE != 'h' and HEAD_TYPE != 'i':
          self.confidence_size = 1
          self.mlp_head_conf = nn.Sequential(
              nn.LayerNorm(self.mlp_conf_input_size),
              nn.Linear(self.mlp_conf_input_size, self.embed_dim * self.num_modalities),
              nn.GELU(),
              nn.Dropout(dropout),
              nn.LayerNorm(self.embed_dim * self.num_modalities),
              nn.Linear(self.embed_dim * self.num_modalities, self.confidence_size),
          )

        self.dropout = nn.Dropout(dropout) # Dropout layer used in forward

        ''' Parameters/Embeddings '''
        # Classification token parameter
        self.cls_token_confidence = nn.Parameter(torch.randn(1,1,embed_dim*self.num_modalities))
        self.cls_token_bbox = nn.Parameter(torch.randn(1,1,embed_dim*self.num_modalities)) # for the case of head type 'd'
        self.cls_token_objectness = nn.Parameter(torch.randn(1,1,embed_dim*self.num_modalities)) # for the case of head type 'f'

        self.n_tokens = 2 if HEAD_TYPE == 'd' or HEAD_TYPE == 'h' else 1 # Define positional embedding size depending on the number of cls tokens

        # Polsitional embedding parameter
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_tokens + self.actual_num_patches, embed_dim*self.num_modalities))

    # Forward do ViT
    def forward(self, x, Lander_Class=False): #to active/deactive final MLPs blocks
        """ [Stage 1] """
        """Split channels"""
        modalities = {}
        for mod_idx in self.active_modalities:
           modalities[mod_idx] = x[:,mod_idx,:,:].unsqueeze(dim=1)

        """ Apply stem """
        stemmed = []
        for mod_idx in modalities:
           if mod_idx == 0: stemmed.append(self.input_layer0(modalities[mod_idx]))
           elif mod_idx == 1: stemmed.append(self.input_layer1(modalities[mod_idx]))
           elif mod_idx == 2: stemmed.append(self.input_layer2(modalities[mod_idx]))

        #print('x.shape ',x.shape, 'after stems')
        """ Fusion """
        x = torch.concat(stemmed, dim=1) # Concat along embed_dim axis
        #print('x.shape ',x.shape, 'after concat')
        # TODO: em vez de concat (sum ou mul)
        x = self.CBAM(x)
        #print('x.shape ',x.shape, 'after SE')

        if HEAD_TYPE != 'i':
          x = x.flatten(2).transpose(1, 2)
          B, T, _ = x.shape
        else:
          x = self.max_pool_dim_reduction(x) # apply maxpool to reduce dimensionality
          x = x.flatten(1)
          B, T = x.shape
        #print('x.shape ',x.shape, 'after flatten')
        #print('shape:', x.shape, '\ttranspose')

        """ [Stage 2] """
        """ Add CLS (classification) token on top """
        cls_token_confidence = self.cls_token_confidence.repeat(B, 1, 1)
        cls_token_objectness = self.cls_token_objectness.repeat(B, 1, 1)
        cls_token_bbox = self.cls_token_bbox.repeat(B, 1, 1)
        if HEAD_TYPE == "d":
          inp = torch.cat([cls_token_bbox, x], dim=1)
          inp = torch.cat([cls_token_confidence, inp], dim=1)
        elif HEAD_TYPE == "e":
          inp = torch.cat([cls_token_bbox, x], dim=1)
        elif HEAD_TYPE == "f":
          inp = torch.cat([cls_token_objectness, x], dim=1)
        elif HEAD_TYPE == "g": # cls token bbox for both heads
          inp = torch.cat([cls_token_bbox, x], dim=1)
        elif HEAD_TYPE == "h": # cls_token0 for bbox head and cls_token1 for objectness head
          inp = torch.cat([cls_token_bbox, x], dim=1)
          inp = torch.cat([cls_token_objectness, inp], dim=1)
        #else:
        #  inp = torch.cat([cls_token_confidence, x], dim=1) 

        """ [Stage 3] """
        """ Add positional encoding """
        if HEAD_TYPE != 'i':
          x = inp + self.pos_embedding[:,:T+self.n_tokens]
        
        """ [Stage 4] - **Transforrmer** Encoder Head """
        """ [Stage 5] - Linear projection of the Transformer encoder """
        #print('x shape: ', x.shape, 'before transformer')
        if HEAD_TYPE != 'i':
          x = self.dropout(x)
          x = x.transpose(0, 1)
          x = self.transformer(x)
        
        if HEAD_TYPE != 'i':
          inp = inp.transpose(0, 1)
        else:
          x = x.transpose(0,1)

        if HEAD_TYPE != 'i':
          if self.skip_mult: # residual mas com multiplicacao
            x = torch.mul(x, inp)
          else:	# residual (soma)
            x = x + inp

        x = x.transpose(0, 1) if HEAD_TYPE != 'd' and HEAD_TYPE != 'g' and HEAD_TYPE != 'h' else x
        #return x for Detector+DQN
        if Lander_Class:
          return x

        """ [Stage 6] """
        """ Perform prediction (depends on head type) """
        x_conf_head = []
        x_bbox_head = []

        #print('x shape: ', x.shape, 'after transformer')

        if HEAD_TYPE == "a": # [cls_token] for one head and [reg_vector] for the other
          x_bbox_head = x[:,1:,:]
          x_bbox_head = torch.flatten(input=x_bbox_head, start_dim=1, end_dim=-1)
          x_conf_head = x[:,0,:]
          x_conf_head = torch.flatten(input=x_conf_head, start_dim=1, end_dim=-1)
        elif HEAD_TYPE == "b": # [cls_token, reg_vector] for both
          x_bbox_head = torch.flatten(input=x, start_dim=1, end_dim=-1)
          x_conf_head = x_bbox_head
        elif HEAD_TYPE == "c": # [reg_vector] for both
          x_bbox_head = x[:,1:,:] 
          x_bbox_head = torch.flatten(input=x_bbox_head, start_dim=1, end_dim=-1)
          x_conf_head = x_bbox_head
        elif HEAD_TYPE == "d": # each head receives its cls_token
          x_conf_head = x[0]
          x_bbox_head = x[1]
        elif HEAD_TYPE == "e":
          #x_bbox_head = x[:,0,:] @ TODO: go back to this
          x_bbox_head = torch.flatten(input=x, start_dim=1, end_dim=-1)
        elif HEAD_TYPE == "f":
          x_objectness_head = x[:,0,:]
          x_objectness_head = torch.flatten(input=x_objectness_head, start_dim=1, end_dim=-1)
          #x_bbox_head = x # or x[:,1,:]
          x_bbox_head = x[:,1:,:] # or x[:,1,:]
          x_bbox_head = torch.flatten(input=x_bbox_head, start_dim=1, end_dim=-1) # só (a) reg vector ou (b) todo o x?
        elif HEAD_TYPE == "g": # each head same cls_token
          x_objectness_head = x[0]
          x_bbox_head = x[0]
        elif HEAD_TYPE == "h": # cl# s_token0 for bbox head and cls_token1 for objectness head
          x_objectness_head = x[0]
          x_bbox_head = x[1]
        elif HEAD_TYPE == "i":
          x_objectness_head = x
          x_bbox_head = x
        else: # same as "a"
          x_bbox_head = x[:,1:,:]
          x_bbox_head = torch.flatten(input=x_bbox_head, start_dim=1, end_dim=-1)
          x_conf_head = x[:,0,:]
          x_conf_head = torch.flatten(input=x_conf_head, start_dim=1, end_dim=-1)
        
        #print('x shape', x_bbox_head.shape, 'bbox')
        #print('x shape', x_objectness_head.shape, 'objectness')

        bbox = self.mlp_head_bbox(x_bbox_head)
        conf = self.mlp_head_conf(x_conf_head) if HEAD_TYPE != 'e' and HEAD_TYPE != 'f' and HEAD_TYPE != 'g' and HEAD_TYPE != 'h' and HEAD_TYPE != 'i' else []
        objectness = self.mlp_head_objectness(x_objectness_head) if HEAD_TYPE == 'f' or HEAD_TYPE == 'g' or HEAD_TYPE == 'h' or HEAD_TYPE == 'i' else []

        return bbox, conf, objectness
    
        [(1,4),(1,1),(1,1)]

# class ViT(pl.LightningModule):
#     #the logger to be used (in this case tensorboard)
#     n_step     = 1
#     train_loss = 0
#     train_acc  = 0
#     val_loss   = 0
#     val_acc    = 0
#     test_loss  = 0
#     test_acc   = 0

#     def __init__(self, model_kwargs):
#         super().__init__()
#         self.save_hyperparameters()
#         self.model = VisionTransformer(**model_kwargs) # Create model
#         self.example_input_array = next(iter(train_loader))[0]
#         self.test_pred = []
#         self.test_label = []
#         self.test_imgs = []
#         self.n_step_train = 0
#         self.n_step_val = 0
#         self.n_step_test = 0

#         """ Auxiliar variables """
#         self.last_val_batch = []
#         self.img2area_interval = {} # dict that creates pair between test img and corresponding area interval

#     def forward(self, x):
#         return self.model(x)

#     def configure_optimizers(self):
#         optimizer = optim.AdamW(self.parameters(), lr=self.hparams["model_kwargs"]["lr"])
#         lr_scheduler ={
#                        'scheduler':  optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, min_lr=1e-6, threshold=0.05),
#                         'monitor': 'val_loss', # Default: val_loss
#                         'interval': 'epoch',
#                         'frequency': 1
#                        }
#         return [optimizer], [lr_scheduler]