# -*- coding: utf-8 -*-
"""
Created on Thu Dec 8 12:58:42 2022

This class implements a game board

@author: Ninniri Matteo. Student ID: 543873
"""

#UNCOMMENT graphviz if you want to use it (and the related methods below as well)
# import graphviz
# from graphviz import Source

import torch
import numpy as np

from Node import *

class Board():
    def __init__(self,
                 root : Node = None):
        """
        Constructor method

        Parameters
        ----------
        root : Node, optional
            The root node of the board (hence, the initial state of the board). 
            The default is None.

        Returns
        -------
        None.

        """
        if(root is None):
            self.root = self.CreateRoot()
        else:
            self.root = root
    
    #ended up unused
    def CreateRoot(self):
        pass
    
    def ForceAction(self, action : int):
        """
        Checks in the current root whether the parameter is a valid action, in which
        case it performs it.

        Parameters
        ----------
        action : int
            the integer of the action to perform. The action is assumed to be in
            the space of the illegal actions (and as a result, it will be
            converted into the legal space).

        Returns
        -------
        bool
            True. If the action is not acceptable, it causes an exception.

        """
        if(not any(self.root.has_been_expanded_list)):
            #this if is entered only during a Versus game where the first player
            #is human (and thus there's no expansion of the nodes)
            self.root.CreateChilds()
        
        #action address in the legal space = action - [number of zeros from 0 to action - 1]
        legal_mask = self.GetLegalMovesMask()
        legal_index = np.count_nonzero(legal_mask[:(action + 1)]) - 1
        
        try:
            # print("action =", action, ", legal_mask =", legal_mask, ", legal_index =", legal_index, ", legal_mask[0:action + 1] =", legal_mask[0:action + 1])
            self.root = self.root.childs[legal_index]
        except:
            print("self.root.GetLegalMoves =", self.root.GetLegalMoves(),
                  ", legal_mask =", legal_mask,
                  ", legal_index =", legal_index,
                  ", action =", action)
            exit()
            
        del self.root.parent
        self.root.parent = None
        
        return True
         
    ###########################################################################
    #The following methods are just root shortcuts
    
    def GetRoot(self):
        return self.root
    
    def GetCurrentPlayerID(self):
        return self.root.GetCurrentPlayerID()
    
    def HasExceededDrawLimit(self):
        return self.root.HasExceededDrawLimit()
    
    def GetWinner(self):
        return self.root.GetWinner()
    
    def IsTerminal(self):
        return self.root.IsTerminal()
    
    def GetBoardMatrix(self):
        return self.root.GetBoardMatrix()
    
    def GetBoard(self, player : int = None):
        return self.root.GetBoard(player)
    
    def GetBoardAsTensor(self, player : int = None):
        return self.root.GetBoardAsTensor(player)
    
    def GetLegalMoves(self):
        return self.root.GetLegalMoves()
    
    def GetLegalMovesMask(self):
        return self.root.GetLegalMovesMask()
    
    def AddIllegals(self, p: torch.Tensor):
        return self.root.AddIllegals(p)
    
    def RemoveIllegals(self, p : torch.Tensor):
        return self.root.RemoveIllegals(p)
    
    def ZeroIllegalsAndNormalize(self, p : torch.Tensor):
        return self.root.ZeroIllegalsAndNormalize(p)
    
    def GetCurrentTurn(self):
        return self.root.GetCurrentTurn()
    
    ###########################################################################
    #uncomment if you want to use graphviz for plotting the search tree
    """
    def RenderBoard(self, player, name = 'test.gv', keepUnvisited = True):
        root = self.root
        graph = graphviz.Digraph(format = 'png')
        
        self.ExploreNode(graph, root, '0', player, keepUnvisited)
        
        return graph
    
    def ExploreNode(self,
                    graph         : graphviz.Digraph,
                    node          : Node, 
                    identifier    : str,
                    player        : int,
                    keepUnvisited : bool = True):
        graph.node(identifier, node.GetBoardAsString())
        
        if(node.childs is not None):
            for i in range(len(node.childs)):
                childID = identifier + str(i)
                child   = node.childs[i]
                
                
                if(len(node.childs) > 1):
                    if(node.N[player][i].item() > 0 or keepUnvisited):
                        self.ExploreNode(graph, child, childID, player, keepUnvisited)
                    
                        label = 'N = ' + "{:.5f}".format(node.N[player][i].item()) + "\n" +\
                                'Q = ' + "{:.5f}".format(node.Q[player][i].item()) + "\n" +\
                                'P = ' + "{:.5f}".format(node.P[player][i].item()) + "\n" +\
                                'a = ' + "{:.5f}".format(node.a[player][i].item()) + "\n"
                        graph.edge(identifier, childID, label=label)
                else:
                    if(node.N[player] > 0 or keepUnvisited):
                        self.ExploreNode(graph, child, childID, player, keepUnvisited)
                        
                        label = 'N = ' + "{:.5f}".format(node.N[player]) + "\n" +\
                                'Q = ' + "{:.5f}".format(node.Q[player]) + "\n" +\
                                'P = ' + "{:.5f}".format(node.P[player]) + "\n" +\
                                'a = ' + "{:.5f}".format(node.a[player]) + "\n"
                            
                        graph.edge(identifier, childID, label=label)
    """  
            
class ConnectFourBoard(Board):
    def __init__(self, root : ConnectFourNode = None):
        if(root is None):
            self.root = ConnectFourNode()
        else:
            self.root = root
            
class TicTacToeBoard(Board):
    def __init__(self, root : TicTacToeNode = None):
        if(root is None):
            self.root = TicTacToeNode()
        else:
            self.root = root
