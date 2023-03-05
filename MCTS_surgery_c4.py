# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 16:02:48 2023

Performs the experiment described in the report for which we test, for a game 
played by a perfect Connect 4 solver, how many moves does our agent guess correctly

the game in question is represented by the series
[3,3,3,3,3,2,2,2,2,0,2,3,5,6,2,5,6,6,6,0,0,6,0,0,5,0,6,1,1,1,1,1,1,5,5,5,4,4,4,4,4]
(each number is the column where we place the stone at each move)

@author: Ninniri Matteo. Student ID: 543873
"""

from Board import ConnectFourBoard
from AlphaZeroAgent import AlphaZeroAgent
from Net import Net
import torch

torch.manual_seed(0)

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
nn.LoadModel('Connect4_final.pth') #change this to whatever you want
agent = AlphaZeroAgent(nn, turns_before_ann = 6, n_iters_per_move = 300, epsilon = .1)


b = ConnectFourBoard()

moves = [3,3,3,3,3,2,2,2,2,0,2,3,5,6,2,5,6,6,6,0,0,6,0,0,5,0,6,1,1,1,1,1,1,5,5,5,4,4,4,4,4]
correctly_guessed = 0

for action in moves:
    print("TARGET ACTION: ", action)
    if(not b.IsTerminal()):
        agent_p, _, agent_action = agent.mcts.SearchMove(b, 0, False)
        print("AGENT ACTION = ", agent_action)
        print("AGENT p = ", agent_p)
        b.ForceAction(action)
        
        if(agent_action == action): correctly_guessed += 1
    
    print(b.GetRoot().GetBoardMatrix())
    print("==================================================================")
    
print("The agent guessed", correctly_guessed, "out of", len(moves), "moves")