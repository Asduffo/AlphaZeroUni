# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 20:37:33 2023

This script allows you to play against the Connect 4 agent

@author: Ninniri Matteo. Student ID: 543873
"""

from Board import ConnectFourBoard
from AlphaZeroAgent import AlphaZeroAgent
from Net import Net
from Game import Game

nn1 = Net(x_sz     = 6,
         y_sz      = 7,
         n_ch      = 2,
         k_sz      = 3,
         n_resnet  = 8,
         common_ch = 128,
         p_ch      = 128,
         n_moves   = 7)
nn1.LoadModel('Connect4_final.pth')
agent1 = AlphaZeroAgent(nn1, turns_before_ann = 6, n_iters_per_move = 300, epsilon = .1)

legal_choices = [1, 2]
turn = 0
print("Do you want to play as player 1 (1) or player 2 (2)?")
turn = int(input("\nChoice: "))

while((turn in legal_choices) == False):
    print("Illegal. Please choose one from: ", legal_choices)
    turn = int(input("\nChoice: "))
    
b = ConnectFourBoard()
g = Game()

if(turn == 2):
    g.Play(b, agent1, None, verbose = 3)
else:
    g.Play(b, None, agent1, verbose = 3)