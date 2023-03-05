# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 23:22:03 2023

Implements the neural network's architecture as described in the AlphaGo Zero paper

@author: Ninniri Matteo. Student ID: 543873
"""

import torch

"""
Resnet block according to the specifics of the AlphaGo Zero paper
"""
class ResNetBlock(torch.nn.Module):
    def __init__(self, 
                 k_sz         : int = 3,            #kernel size
                 mid_ch       : int = 16,           #number of channels in the resnet blocks
                 device             = 'cuda'
                 ):
        """
        ResNetBlock Constructor

        Parameters
        ----------
        k_sz : int, optional
            Kernel size. The default is 3.
        mid_ch : int, optional
            number of channels in the resnet blocks. The default is 16.
        device : TYPE, optional
            'cpu' for CPU, 'cuda' for GPU. The default is 'cuda'.

        Returns
        -------
        None.

        """
        super().__init__()
        
        self.k_sz   = k_sz
        self.mid_ch = mid_ch
        self.device = device
        
        self.conv1 = torch.nn.Conv2d(in_channels = mid_ch, out_channels = mid_ch, 
                                     kernel_size = k_sz, padding = 'same').to(self.device)
        self.batch1 = torch.nn.BatchNorm2d(mid_ch).to(self.device)
        
        self.conv2 = torch.nn.Conv2d(in_channels = mid_ch, out_channels = mid_ch, 
                                     kernel_size = k_sz, padding = 'same').to(self.device)
        self.batch2 = torch.nn.BatchNorm2d(mid_ch).to(self.device)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.batch1(x1)
        x1 = torch.nn.functional.relu(x1)
        
        x1 = self.conv2(x1)
        x1 = self.batch2(x1)
        x1 = x + x1
        x1 = torch.nn.functional.relu(x1)
        
        return x1

"""the neural network's architecture as described in the AlphaGo Zero paper"""
class Net(torch.nn.Module):
    def __init__(self,
                 x_sz       : int = 6,    #number of rows
                 y_sz       : int = 7,    #number of cols
                 n_ch       : int = 2,    #input channels (board, turn)
                 k_sz       : int = 3,    #kernel size
                 
                 n_resnet   : int = 2,    #number of common layers
                 common_ch  : int = 32,
                 p_ch       : int = 32,
                 n_moves    : int = 7,
                 device           = 'cuda'
                 ):   
        """
        Constructor

        Parameters
        ----------
        x_sz : int, optional
            number of rows. The default is 6.
        y_sz : int, optional
            number of columns. The default is 7.
        n_ch : int, optional
            input channels. The default is 2.
        k_sz : int, optional
            kernel size. The default is 3.
        n_resnet : int, optional
            number of resnet layers. The default is 2.
        common_ch : int, optional
            channels in the resnet blocks. The default is 32.
        p_ch : int, optional
            channels in the CNN of the prior's head. The default is 32.
        n_moves : int, optional
            number of possible moves. The default is 7.
        device : TYPE, optional
            'cpu' for CPU, 'cuda' for GPU. The default is 'cuda'.

        Returns
        -------
        None.

        """
        super().__init__()
        
        self.x_sz      = x_sz
        self.y_sz      = y_sz
        self.n_ch      = n_ch
        self.k_sz      = k_sz
        
        self.n_resnet  = n_resnet
        self.common_ch = common_ch
        
        self.p_ch      = p_ch
        self.n_moves   = n_moves
        
        self.device    = device
        
        self.resnets   = []
        
        self.conv1     = torch.nn.Conv2d(in_channels = n_ch, out_channels = common_ch, 
                                         kernel_size = k_sz, padding      = 'same').to(self.device)
        self.batch1    = torch.nn.BatchNorm2d(common_ch).to(self.device)
        self.resnets.append(self.conv1)
        for i in range(1, n_resnet):
            self.resnets.append(ResNetBlock(self.k_sz, common_ch, self.device))
        
        #output layer's two heads (one is p, the softmax of the moves, and the other is the value net's head)
        self.out_p_cn  = torch.nn.Conv2d(in_channels = common_ch, out_channels = 2, kernel_size = 1, padding = 'same').to(self.device)
        self.out_p_bn  = torch.nn.BatchNorm2d(2).to(self.device)
        self.out_p_ln  = torch.nn.Linear(in_features = x_sz*y_sz*2, out_features = n_moves).to(self.device)
        
        self.out_z_cn  = torch.nn.Conv2d(in_channels = common_ch, out_channels = 1, kernel_size = 1, padding = 'same').to(self.device)
        self.out_z_bn  = torch.nn.BatchNorm2d(1).to(self.device)
        self.out_z_ln1 = torch.nn.Linear(in_features = x_sz*y_sz, out_features = p_ch).to(self.device)
        self.out_z_ln2 = torch.nn.Linear(in_features = p_ch, out_features = 1).to(self.device)
        
        self.modules = torch.nn.ModuleList(self.resnets + [self.out_p_cn, self.out_p_ln,  \
                                           self.out_z_cn, self.out_z_ln1, self.out_z_ln2, \
                                           self.batch1, self.out_p_bn, self.out_z_bn])
    
    def forward(self, x):
        # print(x.size())
        
        x1 = self.resnets[0](x)
        x1 = self.batch1(x1)
        x1 = torch.nn.functional.relu(x1)
        
        for i in range(1, len(self.resnets)):
            x1 = self.resnets[i](x1)
        
        #p
        p = self.out_p_cn(x1)
        p = self.out_p_bn(p)
        p = torch.nn.functional.relu(p)
        p = torch.flatten(p, start_dim = 1)
        p = self.out_p_ln(p)
        
        #moves from [batch_size, 1, #moves] to [batch_size, #moves]
        p = torch.reshape(p, (-1, self.n_moves)) 
        
        #z
        z = self.out_z_cn(x1)
        z = self.out_z_bn(z)
        z = torch.nn.functional.relu(z)
        z = torch.flatten(z, start_dim = 1)
        z = self.out_z_ln1(z)
        z = torch.nn.functional.relu(z)
        z = self.out_z_ln2(z)
        z = torch.tanh(z)
        
        
        return p, z
    
    def SaveModel(self, path : str):
        """
        Saves the model weights

        Parameters
        ----------
        path : str
            file path where we should save the model.

        Returns
        -------
        None.

        """
        torch.save(self.state_dict(), path)
    
    def LoadModel(self, path : str):
        """
        Loads the model weights

        Parameters
        ----------
        path : str
            file path containing the model's weights

        Returns
        -------
        None.

        """
        self.load_state_dict(torch.load(path))