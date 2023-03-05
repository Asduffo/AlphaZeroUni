# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 00:35:34 2023

Trains the Tic-Tac-Toe agent

@author: Ninniri Matteo. Student ID: 543873
"""

from Board import TicTacToeBoard
from AlphaZeroAgent import AlphaZeroAgent
from AlphaZeroTrainer import AlphaZeroTrainer, OptimizerData, SchedulerData
from Net import Net
from DataAugmenter import TTTDataAugmenter
import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

torch.manual_seed(0)

nn = Net(x_sz      = 3,
         y_sz      = 3,
         n_ch      = 2,
         k_sz      = 3,
         n_resnet  = 2,
         common_ch = 16,
         p_ch      = 16,
         n_moves   = 9)
print("There are", count_parameters(nn), "parameters in the network")
agent = AlphaZeroAgent(nn, n_iters_per_move = 100, turns_before_ann = 2, alpha = .03, epsilon = .1)

k = 500
trainer = AlphaZeroTrainer(BoardClass = TicTacToeBoard, 
                          augmenter = TTTDataAugmenter(),
                          opt_data  = OptimizerData(torch.optim.Adam, lr = .001),
                          sched_data = SchedulerData(torch.optim.lr_scheduler.MultiStepLR, milestones=[1000], gamma = 1),
                          )
trainer.fit(agent,
            n_iters              =    k,
            dataset_sz_threshold =   16,
            batch_sz             =    4,
            n_eval_games         =    0,
            train_epochs         =    1,
            )

agent.SaveModel('TicTacToe_100_searches_500_epochs.pth')