# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 19:19:19 2023

Contains the data augmentation classes

@author: Ninniri Matteo. Student ID: 543873
"""

import torch

"""
Data augmenter. AlphaZero doesn't use it but that's because it tries to
pass as a "generalized" algorithm and games such as chess do not have stuff such
as rotations et cetera. But I have noticed that it speeds up training a bit
as each rotation of a whole game's data equals a game implicitly played
and hence k times the data at k times less the cost of playing them one by one.
"""
class DataAugmenter():
    def __init__(self):
        pass
    
    def Augment(self, data):
        """
        Given a dataset of training samples, returns an array containing, for
        each input sample, the legal rotation of the sample.

        Parameters
        ----------
        dataset : list
            a list of triples. Each triple is a training sample containing:
                the input board tensor
                the target probability 
                the current player

        Returns
        -------
        a list containing all the legal rotations of each sample.
        """
        return []
    
class C4DataAugmenter():
    def __init__(self):
        super().__init__()
        
    def Augment(self, dataset : list):
        """
        Given a dataset of training samples, returns an array containing, for
        each input sample, the mirror of the sample.
    
        Parameters
        ----------
        dataset : list
            a list of triples. Each triple is a training sample containing:
                the input board tensor
                the target probability 
                the current player
    
        Returns
        -------
        a list containing all the legal rotations of each sample.
        """
        
        # print("==============================================================")
        # print(dataset)
        # print("==============================================================")

        # print("len(dataset)", len(dataset))
        to_return = []
        for i in range(len(dataset)):
            
            new_board  = dataset[i][0].flip(dims  = [-1])
            new_policy = dataset[i][1].flip(dims  = [-1])
            new_legal  = dataset[i][3].flip(dims  = [-1])
            
            # print("==============================================================")
            # print("boardTensor",  dataset[i][0],  "\nnew_board",  new_board)
            # print("policyTensor", dataset[i][1], "\nnew_policy", new_policy)
            # print("legalTensor",  dataset[i][3],  "\nnew_legal",  new_legal)
            # print("==============================================================")
            
            new_sample = [new_board, new_policy, dataset[i][2], new_legal]
            
            to_return.append(new_sample)
            
        return to_return
            
class TTTDataAugmenter():
    def __init__(self):
        super().__init__()
        
    def Augment(self, dataset : list):
        """
        Given a dataset of training samples, returns an array containing, for
        each input sample, the legal rotation of the sample, the mirror, and
        the rotations of the mirror.

        Parameters
        ----------
        dataset : list
            a list of triples. Each triple is a training sample containing:
                the input board tensor
                the target probability 
                the current player

        Returns
        -------
        a list containing all the legal rotations of each sample.
        """
        
        to_return = []
        for sample in dataset:
            for i in range(1, 4):
                new_board  = torch.rot90(sample[0].clone().detach(), k = i, dims = [1, 2])
                
                
                policy_matrix = sample[1].clone().detach().view((3, 3))
                new_policy = torch.rot90(policy_matrix, k = i).reshape((-1))
                
                legal_matrix = sample[3].clone().detach().view((3, 3))
                new_legal  = torch.rot90(legal_matrix, k = i).reshape((-1))
                
                new_sample = [new_board, new_policy, sample[2], new_legal]
                
                to_return.append(new_sample)
            
            #creates a copy of the tensors and flips it like on a mirror
            boardTensor    = torch.fliplr(sample[0])
            policyTensor   = torch.fliplr(sample[1].reshape(1, -1))
            legalTensor    = torch.fliplr(sample[3].reshape(1, -1))
            
            #0 and not 1 since we want to add the flipped board as well => k = 0
            #basically saves the flipped board
            for i in range(0, 4):
                new_board  = torch.rot90(boardTensor.clone().detach(),  k = i, dims = [1, 2])
                
                policy_matrix = policyTensor.clone().detach().view((3, 3))
                new_policy = torch.rot90(policy_matrix, k = i).reshape((-1))
                
                legal_matrix = legalTensor.clone().detach().view((3, 3))
                new_legal  = torch.rot90(legal_matrix, k = i).reshape((-1))
                
                new_sample = [new_board, new_policy, sample[2], new_legal]
                
                to_return.append(new_sample)
                
        return to_return
    