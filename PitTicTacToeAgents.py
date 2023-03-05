# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 16:23:14 2023

This script makes two TicTacToe agents play against each other for k games.  
The relevant statistics are then printed

@author: Ninniri Matteo. Student ID: 543873
"""

from Board import TicTacToeBoard
from AlphaZeroAgent import AlphaZeroAgent
from Net import Net
from Game import Game
import torch

torch.manual_seed(0)

nn100 = Net(x_sz     = 3,
           y_sz      = 3,
           n_ch      = 2,
           k_sz      = 3,
           n_resnet  = 2,
           common_ch = 16,
           p_ch      = 16,
           n_moves   = 9)
agent100 = AlphaZeroAgent(nn100, n_iters_per_move = 10, turns_before_ann = 2, alpha = .03, epsilon = .1)
agent100.LoadModel('TicTacToe_100_searches_500_epochs.pth')

nn10 = Net(x_sz     = 3,
          y_sz      = 3,
          n_ch      = 2,
          k_sz      = 3,
          n_resnet  = 2,
          common_ch = 16,
          p_ch      = 16,
          n_moves   = 9)
agent10 = AlphaZeroAgent(nn10, n_iters_per_move = 10, turns_before_ann = 2, alpha = .03, epsilon = .1)
agent10.LoadModel('TicTacToe_10_searches_500_epochs.pth')

k = 50
won_as_player1   = 0
drawn_as_player1 = 0
lost_as_player1  = 0
won_as_player2   = 0
drawn_as_player2 = 0
lost_as_player2  = 0
for i in range(k):
    b = TicTacToeBoard()
    g = Game()
    result = g.Play(b, agent100, agent10, verbose = 0)
    
    if(result == 0):
        drawn_as_player1 += 1
    elif(result == 1):
        won_as_player1 += 1
    else:
        lost_as_player1 += 1
    
    ###########################################################################
    
    b = TicTacToeBoard()
    g = Game()
    result = g.Play(b, agent10, agent100, verbose = 0)
    
    if(result == 0):
        drawn_as_player2 += 1
    elif(result == -1):
        won_as_player2 += 1
    else:
        lost_as_player2 += 1
    
    print("Iteration", (i+1))
    print("Percentage of games the agent with 100 searches won AS PLAYER 1:", 
          won_as_player1/(i+1), ", drawn =", drawn_as_player1/(i+1), ", lost =", lost_as_player1/(i+1))
    print("Percentage of games the agent with 100 searches won AS PLAYER 2:", 
          won_as_player2/(i+1), ", drawn =", drawn_as_player2/(i+1), ", lost =", lost_as_player2/(i+1))
    print("======================================================================")
    
print("======================================================================")
print("Final score:") 
print("Percentage of games the agent with 100 searches won AS PLAYER 1:", 
      won_as_player1/k, ", drawn =", drawn_as_player1/k, ", lost =", lost_as_player1/k)
print("Percentage of games the agent with 100 searches won AS PLAYER 2:", 
      won_as_player2/k, ", drawn =", drawn_as_player2/k, ", lost =", lost_as_player2/k)