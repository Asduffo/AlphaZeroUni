# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 17:40:59 2023

This script makes two Connect 4 agents play against each other for k games. 
The relevant statistics are then printed

@author: Ninniri Matteo. Student ID: 543873
"""

from Board import ConnectFourBoard
from AlphaZeroAgent import AlphaZeroAgent
from Net import Net
from Game import Game
import torch



nn1 = Net(x_sz     = 6,
         y_sz      = 7,
         n_ch      = 2,
         k_sz      = 3,
         n_resnet  = 8,
         common_ch = 128,
         p_ch      = 128,
         n_moves   = 7)
nn1.LoadModel('Connect4_final.pth')

nn2 = Net(x_sz     = 6,
         y_sz      = 7,
         n_ch      = 2,
         k_sz      = 3,
         n_resnet  = 8,
         common_ch = 128,
         p_ch      = 128,
         n_moves   = 7)
nn2.LoadModel('Connect4_midpoint.pth')

k = 10
won_as_player1   = 0
drawn_as_player1 = 0
lost_as_player1  = 0
won_as_player2   = 0
drawn_as_player2 = 0
lost_as_player2  = 0
for i in range(k):
    torch.manual_seed(i)
    
    agent1 = AlphaZeroAgent(nn1, turns_before_ann = 6, n_iters_per_move = 300, epsilon = .1)
    agent2 = AlphaZeroAgent(nn2, turns_before_ann = 6, n_iters_per_move = 300, epsilon = .1)
    b = ConnectFourBoard()
    g = Game()
    result = g.Play(b, agent1, agent2, verbose = 0)
    
    if(result == 0):
        drawn_as_player1 += 1
    elif(result == 1):
        won_as_player1 += 1
    else:
        lost_as_player1 += 1
    
    ###########################################################################
    agent1 = AlphaZeroAgent(nn1, turns_before_ann = 6, n_iters_per_move = 300, epsilon = .1)
    agent2 = AlphaZeroAgent(nn2, turns_before_ann = 6, n_iters_per_move = 300, epsilon = .1)
    b = ConnectFourBoard()
    g = Game()
    result = g.Play(b, agent2, agent1, verbose = 0)
    
    if(result == 0):
        drawn_as_player2 += 1
    elif(result == -1):
        won_as_player2 += 1
    else:
        lost_as_player2 += 1
    
    print("Iteration", (i+1))
    print("Percentage of games agent 1 won AS PLAYER 1:", 
          won_as_player1/(i+1), ", drawn =", drawn_as_player1/(i+1), ", lost =", lost_as_player1/(i+1))
    print("Percentage of games agent 1 won AS PLAYER 2:", 
          won_as_player2/(i+1), ", drawn =", drawn_as_player2/(i+1), ", lost =", lost_as_player2/(i+1))
    print("======================================================================")
    
print("======================================================================")
print("Final score:") 
print("Percentage of games agent 1 won AS PLAYER 1:", 
      won_as_player1/k, ", drawn =", drawn_as_player1/k, ", lost =", lost_as_player1/k)
print("Percentage of games agent 1 won AS PLAYER 2:", 
      won_as_player2/k, ", drawn =", drawn_as_player2/k, ", lost =", lost_as_player2/k)