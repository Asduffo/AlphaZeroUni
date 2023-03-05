# -*- coding: utf-8 -*-
"""
Created on Thu Dec 8 12:43:48 2022

An MCTS node

@author: Ninniri Matteo. Student ID: 543873"""

from __future__ import annotations

import torch

import numpy as np
import copy as cp

import gym
import gym_connect4
import tictactoe_gym

"""
This class implements a node of the MCTS. It represents a board at a specific
timestep.
"""
class Node():
    def __init__(self,
                 env            : gym.Env = None,
                 gym_identifier : str     = 'tictactoe-v0',
                 parent         : 'Node'  = None,
                 turn           : int     = 0,
                 device                   = 'cuda'
                 ):
        """
        Node constructor

        Parameters
        ----------
        env : gym.Env, optional
            Board's gym environment. The default is None.
        gym_identifier : str, optional
            Board's gym identifier. The default is 'tictactoe-v0'.
        parent : 'Node', optional
            Basically, this is the name of the class as a string. We need it to
            instantiate its childrens (which might belong to a child class).
            The default is None.
        turn : int, optional
            Board's turn. The default is 0.
        device : TYPE, optional
            'cpu' for CPU, 'cuda' for GPU. The default is 'cuda'.

        Returns
        -------
        None.

        """
        self.gym_identifier         = gym_identifier
        self.parent                 = parent
        self.device                 = device
        self.CreateEnv(env, turn)
        
        #initializes the various attributes
        n_players                   = self.GetNPlayers()
        self.N                      = [None  for i in range(n_players)]
        self.Q                      = [None  for i in range(n_players)]
        self.P                      = [None  for i in range(n_players)]
        self.a                      = [None  for i in range(n_players)]
        self.lastAction             = [None  for i in range(n_players)]
        self.has_been_expanded_list = [False for i in range(n_players)]
        self.childs                 = None
    
    def GetNPlayers(self):
        """
        Overridable. Returns the number of players in a game

        Returns
        -------
        int
            The number of players.

        """
        return 2
    
    def CreateEnv(self,
                  env        : gym.Env = None,
                  turn       : int     = 0
                  ):
        """
        Creates the environment

        Parameters
        ----------
        env : gym.Env, optional
            If passed, sets the node to the gym's board. The default is None.
        turn : int, optional
            Current turn. The default is 0.
        
        Returns
        -------
        None.

        """
        if(env is None):
            self.env  = gym.make(self.gym_identifier)
            self.env.reset()
            
            self.turn = 0
        else:
            self.env  = env
            self.turn = turn
    
    
    def Expand(self, 
               p : torch.Tensor, 
               v : float,
               player_id : int):
        """
        Expands the node (but only if we haven't done it already for the current player).

        Parameters
        ----------
        p : torch.Tensor
            actions prior as calculated by a neural network.
        v : float
            node value (ended up unused).          
        player_id : int
            Player's ID (0 for player 1, 1 for player 2).

        Returns
        -------
        None.

        """
        
        """
        If the node has never been expanded, regardless on whether it had been
        expanded by the current player or not, we initialize its childrens
        (which are common to all the players regardless)
        """
        if(not any(self.has_been_expanded_list)):
            #creates the legal actions and the childs
            self.CreateChilds()
        
        #if the node has not been expanded yet by the current player, 
        #we need to create the Q, N, a and P data
        if(not self.has_been_expanded_list[player_id]):
            self.has_been_expanded_list[player_id] = True
            
            #removes the illegal moves from p before saving it
            #QUICK NOTE: it is assumed that p has already zeroed the illegals
            #and the rest is normalized.
            p_legal = self.RemoveIllegals(p).squeeze().to(self.device)
            
            #Node values setup
            self.N[player_id] = torch.zeros_like(p_legal).to(self.device)
            self.Q[player_id] = torch.zeros_like(p_legal).to(self.device)
            self.a[player_id] = torch.zeros_like(p_legal).to(self.device)
            self.P[player_id] = torch.clone(p_legal).detach()
        
        return None
    
    def CreateChilds(self):
        """
        creates the childrens

        Returns
        -------
        None.

        """
        
        self.childs = [self.__class__(self.CloneEnv(),
                                      self.gym_identifier,
                                      self,
                                      self.turn+1,
                                      self.device) for i in range(len(self.GetLegalMoves()))]
        
        """
        gets one node at a time and assigns to each one of them a specific
        move from the list self.legal_moves
        """
        for i in range(len(self.childs)):
            _, _, _, _ = self.childs[i].EnvStep(self.GetLegalMoves()[i])
    
    def CloneEnv(self):
        """
        returns a clone of the environment (meant to be extendible)

        Returns
        -------
        Gym.env
            copy of the node's environment.

        """
        return cp.deepcopy(self.env)
    
    
    def EnvStep(self, action):
        """
        performs and environment step (meant to be extendible since, for example, 
        Connect4 uses lists as a parameter for step())

        Parameters
        ----------
        action : int
            action to execute.

        Returns
        -------
            a quadruple containing the result of the step

        """
        a, b, c, d = self.env.step(action)
        return a, b, c, d
    
    
    def GetCurrentPlayerID(self):
        """
        Returns the current player ID

        Returns
        -------
        int
            0 for player 1, 1 for player 2.

        """
        return self.turn % 2
    
    
    def HasExceededDrawLimit(self):
        """
        Returns true if the game has exceeded the turn count limit

        Returns
        -------
        Returns true if the game has exceeded the turn count limit

        """
        pass
    
    def GetWinner(self):
        """
        Returns the board's winner

        Returns
        -------
        0: draw. 1: player 1; -1: player 2.

        """
        pass
    
    
    def IsTerminal(self):
        """
        Returns True if the node is a terminal state

        Returns
        -------
        True if the node is a terminal state.

        """
        pass
    
    def GetBoard(self, player : int = None):
        """
        Returns the board as a numpy array

        Parameters
        ----------
        player : int, optional
            player's ID. The default is None.

        Returns
        -------
        Returns the board as a numpy array.

        """
        pass
    
    def GetBoardMatrix(self):
        pass
    
    def GetBoardAsTensor(self, player : int = None):
        """
        Returns the board as a torch tensor.

        Parameters
        ----------
        player : int, optional
            player's ID. The default is None.

        Returns
        -------
        torch.tensor
            Board as a tensor.

        """
        return torch.from_numpy(self.GetBoard(player)).type(torch.FloatTensor).to(self.device)
    
    def GetCurrentTurn(self):
        """
        Returns the current turn

        Returns
        -------
        int
            Returns the current turn.

        """
        return self.turn
    
    def GetLegalMoves(self):
        """
        Returns a ndarray containing ONLY the legal moves of the node. It's in the
        space of all possible moves (hence between 0 and the total number of moves,
        legal or not.)

        Returns
        -------
        to_return : ndarray
            Returns a ndarray containing ONLY the legal moves of the node. It's in the
            space of all possible moves (hence between 0 and the total number of moves,
            legal or not).
        """
        
        #prepares the return vector
        to_return = []
        
        #if the node is terminal, there are no legal actions
        if self.IsTerminal(): return to_return
        
        #gets the legal moves MASK
        mask = self.GetLegalMovesMask()
        
        #if the move is legal, it appends it to the return vector
        for i in range(len(mask)):
            if(mask[i] == 1): to_return.append(i)
        
        return to_return
    
    def GetLegalMovesMask(self):
        """
        Returns a mask with a size equal to the number of possible moves (legal 
        and illegal) where an item is set to 1 iff the associated move is legal. 
        0 otherwise.

        Returns
        -------
        mask: np.array
            Returns a mask with a size equal to the number of possible moves 
            (legal and illegal) where an item is set to 1 iff the associated
            move is legal. 0 otherwise.
        """
        pass
    
    def AddIllegals(self, p: torch.Tensor):
        """
        Given a tensor which contains only the legal moves probabilities, returns
        a new probability tensor which also adds the illegal actions, set to zero

        Parameters
        ----------
        p : torch.Tensor
            tensor in the legal moves space.

        Returns
        -------
        torch.tensor
            a new probability tensor which also adds the illegal actions, set to zero.

        """
        #gets the mask containing the legal moves
        mask = self.GetLegalMovesMask()
        
        #tensor to return
        to_return = torch.zeros(len(mask))
        
        #scans through the batch
        if(len(p.shape) > 0):
            c = 0
            for i in range(len(mask)):
                #if the i-th move is legal, we add it to the return vector
                if(mask[i] == 1):
                    to_return[i] = p[c]
                    c += 1
        else:
            #0D tensor means that there is just one legal move
            for i in range(len(mask)):
                if(mask[i] == 1):
                    to_return[i] = p.clone().detach()
        
        return to_return.detach()
    
    def RemoveIllegals(self, p : torch.Tensor):
        """
        Given a tensor which contains both legal and illegal probabilities, returns
        a new probability tensor which does not contain the illegal moves

        Parameters
        ----------
        p : torch.Tensor
            a tensor which contains both legal and illegal probabilities.

        Returns
        -------
        to_return : torch.tensor
            a new probability tensor which does not contain the illegal moves.

        """
        
        #if there are no legal moves, it returns None
        if(self.IsTerminal()): return None
        
        #gets the legal moves list
        legal_moves = self.GetLegalMoves()
        
        #prepares the return tensor
        to_return = torch.zeros((p.size(0), len(legal_moves)))
        
        #sets up the return tensor by placing the legal probabilities where it is legal
        for i in range(len(legal_moves)):
            to_return[:, i] = p[:, legal_moves[i]]
        
        return to_return
    
    def ZeroIllegalsAndSoftmax(self, p : torch.Tensor):
        """
        Given a tensor of the size of the number of  possible moves (legal and illegal),
        sets to zero the p of the illegal moves for the current actions and normalizes the rest
        so that the sum is equal to zero.
        
        Parameters
        ----------
        p : torch.Tensor
            a tensor of the size of the number of  possible moves (legal and illegal)
            ASSUMPTION: p is >= 0 (output of a softmax function).

        Returns
        -------
        to_return : torch.Tensor
            p masked and normalized.

        """
        mask = torch.from_numpy(self.GetLegalMovesMask()).type(torch.FloatTensor).to(self.device)
        
        #masks and normalizes
        exps = torch.exp(p).float()
        masked_exps = exps*mask.float()
        exps_sum = torch.sum(masked_exps)
        to_return = masked_exps/exps_sum
        
        return to_return
    
    def GetBoardAsString(self):
        """
        returns the board as a string (used only with graphviz debugging)

        Returns
        -------
        str
            the board as a string.

        """
        return 'TODO'

"""
Class containing a node for the ConnectFour game.
"""
class ConnectFourNode(Node):
    def __init__(self,
                 env            : gym.Env           = None,
                 gym_identifier : str               = 'Connect4Env-v0',
                 parent         : 'ConnectFourNode' = None,  #future typing because Node has note been defined yet
                 turn           : int               = 0,
                 device                             = 'cuda'
                 ):
        """
        Node constructor

        Parameters
        ----------
        env : gym.Env, optional
            Board's gym environment. The default is None.
        gym_identifier : str, optional
            Board's gym identifier. The default is 'tictactoe-v0'.
        parent : 'Node', optional
            Basically, this is the name of the class as a string. We need it to
            instantiate its childrens (which might belong to a child class).
            The default is None.
        turn : int, optional
            Board's turn. The default is 0.
        device : TYPE, optional
            'cpu' for CPU, 'cuda' for GPU. The default is 'cuda'.

        Returns
        -------
        None.

        """
        super().__init__(env, gym_identifier, parent, turn, device)
    
    def EnvStep(self, action):
        """
        performs and environment step (Connect4 uses lists as a parameter for step())
        Connect4's action structure is weird: it takes a two slot array where
        at each slot we have the action of the i-th player. since we are interested
        in the actions of the current player, we set only that one and keep the other blank.

        Parameters
        ----------
        action : int
            action to execute.

        Returns
        -------
            a quadruple containing the result of the step

        """
        arr         = [0, 0]
        player      = self.GetCurrentPlayerID() ^ 1
        arr[player] = action
        
        a, b, c, d = self.env.step(arr)
        return a, b, c, d
        
    def HasExceededDrawLimit(self):
        """
        Returns true if the game has exceeded the turn count limit

        Returns
        -------
        Returns true if the game has exceeded the turn count limit

        """
        return self.turn > 42
    
    def GetWinner(self):
        """
        Returns the board's winner

        Returns
        -------
        0: no winner. 1: player 1; -1: player 2.

        """
        if    self.env.game.is_draw():    return  0
        elif  self.env.game.is_winner(0): return  1
        else:                             return -1
    
    def IsTerminal(self):
        """
        Returns True if the node is a terminal state

        Returns
        -------
        True if the node is a terminal state.

        """
        return self.env.game.is_game_over()
    
    def GetBoard(self, player : int = None):
        """
        Returns the board as a numpy array

        Parameters
        ----------
        player : int, optional
            player's ID. The default is None.

        Returns
        -------
        Returns the board as a numpy array.

        """
        
        if(player == None): player = 0
        
        board     = self.env.get_state(player)
        to_return = np.zeros((2, 6, 7))
        
        # print(board)
        
        for i in range(6):
            for j in range(7):
                if(board[i,j] == 1): 
                    to_return[0, i, j] = 1
                elif(board[i,j] == 2): 
                    to_return[1, i, j] = 1
        
        return to_return
    
    def GetBoardMatrix(self):
        board = np.array([['-', '-', '-', '-', '-', '-', '-'],
                          ['-', '-', '-', '-', '-', '-', '-'],
                          ['-', '-', '-', '-', '-', '-', '-'],
                          ['-', '-', '-', '-', '-', '-', '-'],
                          ['-', '-', '-', '-', '-', '-', '-'],
                          ['-', '-', '-', '-', '-', '-', '-']], dtype = object)
        
        #-1 since the last element is actually the current player's ID
        model = self.GetBoard()
        
        for i in range(6):
            for j in range(7):
                if(model[0, i, j] == 1): board[i, j] = 'O'
                elif(model[1, i, j] == 1): board[i, j] = 'X'
        
        return board
    
    def GetLegalMovesMask(self):
        """
        Returns a mask with a size equal to the number of possible moves (legal 
        and illegal) where an item is set to 1 iff the associated move is legal. 
        0 otherwise.

        Returns
        -------
        mask: np.array
            Returns a mask with a size equal to the number of possible moves 
            (legal and illegal) where an item is set to 1 iff the associated
            move is legal. 0 otherwise.
        """
        #-1 since the last element of the mask is just the player
        to_return = self.env.get_action_mask(self.GetCurrentPlayerID())[:-1]
        return to_return
    
    def GetBoardAsString(self):
        """
        returns the board as a string (used only with graphviz debugging)

        Returns
        -------
        str
            the board as a string.

        """
        return 'TODO'
    
"""
Class implementing a node for the Tic-Tac-Toe game
"""
class TicTacToeNode(Node):
    def __init__(self,
                 env            : gym.Env         = None,
                 gym_identifier : str             = 'tictactoe-v0',
                 parent         : 'TicTacToeNode' = None, 
                 turn           : int             = 0,
                 device                           = 'cuda'
                 ):
        super().__init__(env, gym_identifier, parent, turn, device)
    
    def EnvStep(self, action):
        """
        performs and environment step

        Parameters
        ----------
        action : int
            action to execute.

        Returns
        -------
            a quadruple containing the result of the step

        """
        a, b, c, d, _ = self.env.step(action)
        return a, b, c, d
    
    def HasExceededDrawLimit(self):
        """
        Returns true if the game has exceeded the turn count limit

        Returns
        -------
        Returns true if the game has exceeded the turn count limit

        """
        return self.turn > 9
    
    def GetWinner(self):
        """
        Returns the board's winner

        Returns
        -------
        0: no winner. 1: player 1; -1: player 2.

        """
        if(not self.IsTerminal()): return 0
        else:                      return self.env.GetWinner()
    
    def IsTerminal(self):
        """
        Returns True if the node is a terminal state

        Returns
        -------
        True if the node is a terminal state.

        """
        return self.env.IsGameOver()
    
    def GetBoardMatrix(self):
        board = np.array([['0','1','2'],['3','4','5'],['6','7','8']], dtype = object)
        
        #-1 since the last element is actually the current player's ID
        model = self.GetBoard()
        
        for i in range(3):
            for j in range(3):
                if(  model[0, i, j] == 1): board[i, j] = 'O'
                elif(model[1, i, j] == 1): board[i, j] = 'X'
        
        return board
    
    def GetBoard(self, player : int = None):
        """
        Returns the board as a numpy array

        Parameters
        ----------
        player : int, optional
            player's ID. The default is None.

        Returns
        -------
        Returns the board as a numpy array.

        """
        if(player == None or player == 0): player =  1
        else:                              player = -1
        
        observation      = self.env.get_observation(player)
        
        to_return        = np.zeros((2,3,3))
        for i in range(3):
            for j in range(3):
                if(observation[i,j] == 1): 
                    to_return[0, i, j] = 1
                elif(observation[i,j] == -1): 
                    to_return[1, i, j] = 1
        
        return to_return
    
    def GetLegalMoves(self):
        """
        Returns a ndarray containing ONLY the legal moves of the node. It's in the
        space of all possible moves (hence between 0 and the total number of moves,
        legal or not.)

        Returns
        -------
        to_return : ndarray
            Returns a ndarray containing ONLY the legal moves of the node. It's in the
            space of all possible moves (hence between 0 and the total number of moves,
            legal or not).
        """
        return self.env.get_actions()    
    
    def GetLegalMovesMask(self):
        """
        Returns a mask with a size equal to the number of possible moves (legal 
        and illegal) where an item is set to 1 iff the associated move is legal. 
        0 otherwise.

        Returns
        -------
        mask: np.array
            Returns a mask with a size equal to the number of possible moves 
            (legal and illegal) where an item is set to 1 iff the associated
            move is legal. 0 otherwise.
        """
        actions = np.zeros(shape = 9)
        
        for i in range(9):
            if(self.env.Is_valid_action(i)):
                actions[i] = 1
        return actions
    
    def GetBoardAsString(self):
        """
        returns the board as a string (used only with graphviz debugging)

        Returns
        -------
        str
            the board as a string.

        """
        observation      = self.env.get_observation(1)
        to_return        = ""
        
        # print(observation)
        
        for i in range(3):
            for j in range(3):
                if(observation[i,j] == 1): 
                    to_return = to_return + "X"
                elif(observation[i,j] == -1): 
                    to_return = to_return + "O"
                else:
                    to_return = to_return + "-"
            to_return = to_return + "\n"
        return to_return