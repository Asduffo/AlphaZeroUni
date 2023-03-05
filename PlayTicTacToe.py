1# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 20:40:52 2023

This script allows you to play against the Tic-Tac-Toe agent

@author: Ninniri Matteo. Student ID: 543873
"""

from Board import TicTacToeBoard
from AlphaZeroAgent import AlphaZeroAgent
from Net import Net
from Game import Game


nn100 = Net(x_sz     = 3,
           y_sz      = 3,
           n_ch      = 2,
           k_sz      = 3,
           n_resnet  = 2,
           common_ch = 16,
           p_ch      = 16,
           n_moves   = 9)
agent100 = AlphaZeroAgent(nn100, n_iters_per_move = 100, turns_before_ann = 2, alpha = .03, epsilon = .1)
agent100.LoadModel('TicTacToe_100_searches_500_epochs.pth') #change this to whatever you want

legal_choices = [1, 2]
turn = 0
print("Do you want to play as player 1 (1) or player 2 (2)?")
turn = int(input("\nChoice: "))

while((turn in legal_choices) == False):
    print("Illegal. Please choose one from: ", legal_choices)
    turn = int(input("\nChoice: "))

b = TicTacToeBoard()
g = Game()

if(turn == 2):
    g.Play(b, agent100, None, verbose = 3)
else:
    g.Play(b, None, agent100, verbose = 3)