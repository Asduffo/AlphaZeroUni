# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 00:35:34 2023

Trains the Connect 4 agent

@author: Ninniri Matteo. Student ID: 543873
"""

from Board import ConnectFourBoard
from AlphaZeroAgent import AlphaZeroAgent
from AlphaZeroTrainer import AlphaZeroTrainer, OptimizerData, SchedulerData
from Net import Net
from DataAugmenter import C4DataAugmenter
import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

torch.manual_seed(0)

nn = Net(x_sz     = 6,
        y_sz      = 7,
        n_ch      = 2,
        k_sz      = 3,
        n_resnet  = 8,
        common_ch = 128,
        p_ch      = 128,
        n_moves   = 7)
nn.LoadModel('Connect4_final.pth') #comment this line out if you want to start from zero

print("There are", count_parameters(nn), "parameters in the network")
agent = AlphaZeroAgent(nn, n_iters_per_move = 300, turns_before_ann = 6, epsilon = .1)

k = 200
trainer = AlphaZeroTrainer(BoardClass = ConnectFourBoard, 
                            augmenter   = C4DataAugmenter(),
                            opt_data    = OptimizerData(torch.optim.AdamW, lr = .0001, weight_decay = .0001),
                            sched_data  = SchedulerData(torch.optim.lr_scheduler.MultiStepLR, milestones=[2000], gamma = 1),
                            )
trainer.fit(agent,
            n_iters              =   k,
            dataset_sz_threshold = 128,
            batch_sz             =  16,
            n_eval_games         =   0,
            train_epochs         =   1,
            )

nn.SaveModel('Connect4_10.pth')