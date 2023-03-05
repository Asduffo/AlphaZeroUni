# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 00:22:38 2023

An alphazero agent. It is just a wrapper for an MCTS and a network.

@author: Ninniri Matteo. Student ID: 543873
"""

import torch

from MCTS import MCTS

class AlphaZeroAgent():
    def __init__(self,
                 f_0          : torch.nn.Module = None,
                 weights_path : str             = None,
                 
                 cpuct            : float       = 1,
                 n_iters_per_move : int         = 10,
                 turns_before_ann : int         = 5,
                 alpha            : float       = .03,
                 epsilon          : float       = .25,
                 
                 device                         = 'cuda'
                 ):
        """
        Initializes the agent

        Parameters
        ----------
        f_0 : torch.nn.Module, optional
            The initial neural network. The default is None.
        weights_path : str, optional
            if not None, loads the neural network weights from the file pointed by this path. The default is None.
        cpuct : float, optional
            \c_{puct} in the AlphaGo Zero paper. The default is 1.
        n_iters_per_move : int, optional
            Number of root-leaf expansions for each move search. The default is 10.
        turns_before_ann : int, optional
            For how many turns does the MCTS has to perform annealing. The default is 5.
        alpha : float
            alpha value used by the MCTS (see AlphaGo Zero's paper for more details)
        epsilon : float
            epsilon value used by the MCTS (see AlphaGo Zero's paper for more details)
        device : TYPE, optional
            'cpu' for CPU training. 'cuda' for GPU. The default is 'cuda'.

        Returns
        -------
        None.

        """
        self.cpuct            = cpuct
        self.n_iters_per_move = n_iters_per_move
        self.turns_before_ann = turns_before_ann
        self.alpha            = alpha
        self.epsilon          = epsilon
        self.device           = device
        
        """
        loads the weights, if any.
        """
        if(weights_path is None):
            self.f_0 = f_0
        else:
            self.LoadModel(weights_path)
        
        self.mcts = MCTS(self.f_0, cpuct, n_iters_per_move, turns_before_ann, alpha, epsilon, device)
        
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
        self.f_0.SaveModel(path)
        
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
        self.f_0.LoadModel(path)