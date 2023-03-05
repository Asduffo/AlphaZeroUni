# -*- coding: utf-8 -*-
"""
Created on Fri Dec 9 19:44:02 2022

This file contains the MCTS class and its relative methods

@author: Ninniri Matteo. Student ID: 543873
"""

import torch

from Board import Board


class MCTS():
    """
    This class performs the MCTS, using the neural network nn passed as a parameter.
    
    Constructor parameters
    ----------------------
    nn : torch.nn.Module
        The neural network module
    cpuct : float
        c\_puct value used by the MCTS (see AlphaGo Zero's paper for more details)
    n_iters_per_move : int
        How many root->leaf expansions (hence, SearchSingleStep calls) should we 
        execute for each move search
    turns_before_ann : int
        for how many turns \pi should be proportional to N and not just an argmax
    alpha : float
        alpha value used by the MCTS (see AlphaGo Zero's paper for more details)
    epsilon : float
        epsilon value used by the MCTS (see AlphaGo Zero's paper for more details)
    device
        'cpu' for CPU, 'cuda' for GPU
    """
    
    def __init__(self,
                 nn               : torch.nn.Module,
                 cpuct            : float = 1.0,
                 n_iters_per_move : int   = 200,
                 turns_before_ann : int   = 5,
                 alpha            : float = .03,
                 epsilon          : float = .25,
                 device                   = 'cuda'
                 ):
        self.nn               = nn
        self.cpuct            = cpuct
        self.n_iters_per_move = n_iters_per_move
        self.turns_before_ann = turns_before_ann
        self.alpha            = alpha
        self.epsilon          = epsilon
        self.device           = device
        
    
    def SearchSingleStep(self,
                         board  : Board,
                         player : int,
                         train  : bool = False):
        """
        Performs one single root->leaf search. It then expands the leaf and 
        initializes its childrens (unless the node is a terminal state)
        
        Parameters
        ----------
        board  : Board
            An instance of the class Board for which we want to perform a 
            root->leaf expansion on
        player : int
            Player ID (used to access the correct node values)
        train  : bool, optional
            Ended up unused
            
        Returns
        -------
        None
        """
        
        #gets the board's root, containing the current board state
        currNode = board.GetRoot()
        
        #tree search until we reach an unexpanded leaf
        while(currNode.has_been_expanded_list[player]):
            #(state, action) priors (calculated when currNode is first expanded.
            #if we reach this line it means that it has already been expanded)
            p = currNode.P[player].clone().detach()
            
            #adds dirichlet noise to the root
            if(currNode == board.GetRoot() and len(p.shape) > 0 and self.alpha > 0 and self.epsilon > 0):
                rho_alpha = torch.distributions.Dirichlet(torch.ones_like(p)*self.alpha).sample().to(self.device)
                
                #applies the noise. There is no need to normalize since the
                #vector p is in the space of the legal moves
                p = (1 - self.epsilon)*p + self.epsilon*rho_alpha
                
                #normalizes p to proper probabilities
                p = p/torch.sum(p)
            
            #calculates U and a according to the paper
            frac = torch.sqrt(torch.sum(currNode.N[player]))/(1 + currNode.N[player])
            U = self.cpuct*p*frac
            currNode.a[player] = currNode.Q[player] + U
            
            #writing "a" is faster than "writing currNode.a[player]", isn't it?
            a = currNode.a[player]
            
            #calculates the action to take. The paper explicitly says to use 
            #argmax, so no sampling. HOWEVER, if two actions are equi-probable, 
            #some randomness might actually help
            
            #gets the elements amounting to the maximum a value
            max_u = (a == torch.max(a)).nonzero().view(-1)
            
            # if(currNode.GetCurrentTurn() == 1):
                # print("P = ", currNode.P[player])
                # print("U = ", U)
                # print("Q = ", currNode.Q[player])
                # print("a = ", a)
                # print("max_u ", max_u)
                
            """
            n_elements == 0 happens if there's 1 dominant action (no equi-probable). 
            When this happens, we just sample from argmax
            """
            n_elements = max_u.size(0)
            if(n_elements > 0):
                #samples one of the equi-probable moves.
                action = torch.randint(low = 0, high = n_elements, size = (1,1))[0].item()
                lastActionIndex = max_u[action]
            else:
                lastActionIndex = torch.argmax(a).item()
            
            # print("len(currNode.childs) = ", len(currNode.childs), ", len(a) = ", a.size(1), ", lastActionIndex = ", lastActionIndex)
            
            #saves the action inside the node's data in preparation to the backup
            #phase and performs the action.
            currNode.lastAction[player] = lastActionIndex
            currNode = currNode.childs[lastActionIndex]
        
        #evaluates the node (the current node is unexpanded for sure)
        p, v = self.nn(currNode.GetBoardAsTensor(currNode.GetCurrentPlayerID()).unsqueeze(0))
        
        
        #expand iff the node is not terminal
        if(not currNode.IsTerminal()):
            #zeroes the illegal entries in the new node's prior vector and 
            #re-normalizes the rest to sum 1 before expanding the node
            p = currNode.ZeroIllegalsAndSoftmax(p)
            currNode.Expand(p, v, player)
        else:
            #if we have encountered a terminal state, v might as well become the
            #reward of the game according to the current player viewpoint
            
            #gets the winner (-1, 0 or 1)
            winner = currNode.GetWinner()
            
            #0 or 1
            curr_player = currNode.GetCurrentPlayerID()
            
            # print("==========================================================")
            # print(currNode.GetBoardMatrix())
            # print("winner", winner)
            # print("curr_player", curr_player)
            
            if(winner == 0):
                #draw
                v = torch.tensor([0]).type(torch.FloatTensor).to(self.device)
            elif((winner ==  1. and curr_player == 1) or 
                 (winner == -1. and curr_player == 0)):
                #case in which the player who did the last move (and so, not the 
                #current player) WON (it will receive a positive reward))
                v = torch.tensor([-1]).type(torch.FloatTensor).to(self.device)
            else:
                #case in which the player who did the last move (and so, not the 
                #current player) LOST (it will receive a negative reward)
                v = torch.tensor([1]).type(torch.FloatTensor).to(self.device)
            
            # print("winner", winner, ", curr_player", currNode.GetBoardAsTensor())
        
        #After we have encountered a leaf and (possibly) expanded it, it performs
        #backtracking and updates the nodes' values
        while(currNode.parent is not None):
            #moves to the parent
            currNode = currNode.parent
            
            #since currNode = currNode.parent sent us to the node's parent,
            #we need to invert the board's value. So, 
            v = -v
            
            #saves the move's data
            if(len(currNode.N[player].shape) > 0):
                lastActionIndex = currNode.lastAction[player]
                
                #makes the formulas cleaner
                N = currNode.N[player][lastActionIndex]
                Q = currNode.Q[player][lastActionIndex]
                
                #proper update (this is an alternative formula, equivalent to the
                #one in the original paper, which does not use W (or better, it's implicit))
                currNode.Q[player][lastActionIndex] = (N*Q + v.item())/(N + 1)
                currNode.N[player][lastActionIndex] += 1
            else:
                #if we have only one legal move, we are dealing with a 1D tensor
                #and the shape management is different
                Q = currNode.Q[player]
                N = currNode.N[player]
                
                currNode.Q[player] = (N*Q + v.item())/(N + 1)
                currNode.N[player] += 1
            
    def SearchMove(self,
                   board  : Board,
                   player : int,
                   train  : bool = False):
        """
        Performs self.n_iters_per_move root->node expansions and determines
        which move is the best

        Parameters
        ----------
        board  : Board
            An instance of the class Board on which we want to perform a root->leaf
            expansion on
        player : int
            player ID (used to access the correct node values)
        train  : bool, optional
            if True, samples the action from p instead of extracting the argmax from it

        Returns
        -------
        pi : torch.Tensor
            raw \pi vector (used only for debugging purposes)
        sample : tuple
            a triple containing:
                a tensor of the board for which we want to execute the move
                the target probability 
                the current player
        finalAction : int
            index of the action in the FULL moves space.
        """
        
        #performs several root->leaf searches from the root and updates the nodes' data
        for i in range(self.n_iters_per_move):
            self.SearchSingleStep(board, player, train)
        
        # print("N after search is", board.GetRoot().N)
        
        #calculates the temperature parameter
        turn = board.GetCurrentTurn()
        temp = self.ScheduleTemperature(turn, train)
        
        #gets the root
        curr = board.GetRoot()
        
        #calculates the \pi vector in the space of the legal moves
        if(temp != 0):
            """
            if we're training and we're doing annealing, the action is sampled
            proportionally to N
            """
            curr_N_power = curr.N[player]**(1/temp)     #numerator
            curr_N_power_sum = torch.sum(curr_N_power)  #denominator
            
            # print("curr_N_power", curr_N_power, "curr_N_power_sum", curr_N_power_sum)
            
            #\pi in the space of the legal moves
            legal_pi = curr_N_power/curr_N_power_sum
        else:
            """
            if temp == 0, then curr.N[player]**(1/temp) = infinity and the
            highest N prevails. This is too numerically unstable to attempt, but
            it becomes equivalent to choosing max(N)
            """
            if(len(curr.N[player].shape) > 0):
                #the action is chosen as the 
                action_index = torch.argmax(curr.N[player]).item()
                legal_pi = torch.zeros_like(curr.N[player]).to(self.device)
                legal_pi[action_index] = 1
            else:
                """
                if N has only one element, the shape handling is different since
                we have only one element and we return just a tensor containing a 
                single 1
                """
                legal_pi = torch.ones_like(curr.N[player]).to(self.device)
        
        """
        adds the illegal moves (they are set to zero anyway), which makes the
        softmax compatible with the neural network's target
        """
        pi = board.AddIllegals(legal_pi)
        # print("legal_pi", legal_pi, "pi", pi)
        
        #samples the final action
        if(train):
            finalAction = torch.multinomial(pi, 1).item()
        else:
            finalAction = torch.argmax(pi).item()
        
        #target action vector
        sample_pi = torch.clone(pi).detach()
        # sample_pi = torch.zeros_like(pi).to(self.device)
        # sample_pi[finalAction] = 1
        
        """
        creates the (incomplete) training sample. curr_player_id will eventually
        get replaced by the game's winner which, of course, we cannot calculate
        in advance
        """
        curr_player_id = curr.GetCurrentPlayerID()
        sample = (curr.GetBoardAsTensor(curr_player_id), sample_pi, curr_player_id)
        
        return pi, sample, finalAction
    
    def SelfPlay(self, 
                 board   : Board,
                 train   : bool  = False,
                 verbose : int   = 0):
        """
        Given a board, the MCTS will proceed to use it to play against itself.

        Parameters
        ----------
        board  : Board
            An instance of the class Board over which we want to perform a root->leaf
            expansion on
        train  : bool, optional
            if True, samples the action from p instead of extracting the argmax from it
        verbose : int, optional
            if > 0, tells the debugger to log the relevant infos after each move.

        Returns
        -------
        gameDataset : list.
            a list containing the game's dataset. Each element contains:
                a tensor of the board at that state
                a tensor containing the policy target (move played at that turn)
                a tensor containing the value target (winner of the game from the player's viewpoint)
        """
        gameDataset = []
        i = 0
        
        #until it's not game over:
        while(not board.IsTerminal()):
            #starts searching for a good move
            pi, sample, finalAction = self.SearchMove(board, 0, train)
            
            """
            the legal moves mask is added to the training samples because I 
            wanted to test how well would the nn learn by masking the illegal 
            moves while calculating the loss. Although I ended up not using it,
            I decided to leave it in the case I want to do more experiments.
            """
            to_add = list(sample) + [torch.from_numpy(board.GetRoot().GetLegalMovesMask()).to(self.device)]
            gameDataset.append(to_add)
            
            if(verbose > 0):
                print("===========================")
                print(board.GetBoardMatrix())
                print("current player: ", board.GetCurrentPlayerID())
                print("Is terminal = ", board.IsTerminal())
                print("pi = ", pi)
                print("finalAction ", finalAction)
                print(i)
            
            #performs the action
            board.ForceAction(finalAction)
            
            i+=1
        
        winner = board.GetWinner()
        
        if(verbose > 0):
            print("===========================")
            print(board.GetBoardMatrix())
            print("current player: ", board.GetCurrentPlayerID())
            print("Is terminal = ", board.IsTerminal())
            print("Winner: ", winner)
        
        #saves the dataset
        for i in range(len(gameDataset)):
            #0 = player 1; 1 = player 2
            curr_player = gameDataset[i][2]
            
            if (winner == 0): gameDataset[i][2] = 0   #draw case
            elif(winner ==  1 and curr_player == 0 or #if the sample board's turn is player 0 and he won
                 winner == -1 and curr_player == 1):  #OR the sample board's turn is player 1 and he won
                gameDataset[i][2] = 1
            else: gameDataset[i][2] = -1              #case where the current player lost
        
        return gameDataset
    
    def ScheduleTemperature(self, 
                            turn  : int, 
                            train : bool = False):
        """
        Given the current turn, returns the temperature

        Parameters
        ----------
        turn : int
            Current turn.
        train : bool, optional
            Wether we are training the network or not. The default is False.

        Returns
        -------
        int
            the temperature value.

        """
        # if(train and self.turns_before_ann > turn): 
        if(self.turns_before_ann > turn): 
            return 1
        else:
            return 0