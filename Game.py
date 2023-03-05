# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 23:31:27 2023

Class which moderates a game between two players. The players can be humans or 
Agents or both.

@author: Ninniri Matteo. Student ID: 543873
"""


from AlphaZeroAgent import AlphaZeroAgent
from Board import Board

"""
Class which moderates a game between two players. The players can be humans or 
Agents or both.
"""
class Game():
    def __init__(self):
        pass
    
    def Play(self,
            board   : Board,                 
            player1 : AlphaZeroAgent = None, 
            player2 : AlphaZeroAgent = None,
            verbose : int            = 0,
            ):
        """
        Given a board, starts a new game from that board

        Parameters
        ----------
        board : Board
            The initial game's board.
        player1 : AlphaZeroAgent, optional
            Player 1. None == human player. The default is None.
        player2 : AlphaZeroAgent, optional
            Player 2. None == human player. The default is None.
        verbose : int, optional
            If > 1, displays the board and everything after each move 
            (you might not want if two agents are playing and you only care about the result)
            The default is 0.

        Returns
        -------
        int
            an integer representing the winner.

        """
        playerList = [player1, player2]
        
        if(verbose > 1):
            print(board.GetBoardMatrix())
            print("================")
        
        while not board.IsTerminal():
            currentPlayerID = board.GetCurrentPlayerID()
            self.PerformMove(board, currentPlayerID, playerList[currentPlayerID], verbose)
            
            if(verbose > 1):
                print(board.GetBoardMatrix())
                print("================")
        
        if(verbose > 0):
            print("winner: ", board.GetWinner())
        return board.GetWinner()
    
    def PerformMove(self,
                    board        : Board,
                    currPlayerID : int,
                    player       : AlphaZeroAgent = None,
                    verbose      : int = 0,
                    ):
        """
        Performs a move on the board

        Parameters
        ----------
        board : Board
            The current game's board.
        currPlayerID : int
            current player's ID (useful only if it's an agent).
        player : AlphaZeroAgent, optional
            the current player. None == human player. The default is None.
        verbose : int, optional
            If > 2, shows the action and the pi vector after an agent's move.

        Returns
        -------
        None.

        """
        if(player == None):                                 #human player
            legal_moves = board.GetLegalMoves()
            
            print("Make a move. Legal choices: ", legal_moves)
            move = int(input("\nChoice: "))
            
            while((move in legal_moves) == False):
                print("Illegal. Please choose one from: ", legal_moves)
                move = int(input("\nChoice: "))
            
            board.ForceAction(move)
            
        else:
            pi, _, action = player.mcts.SearchMove(board, currPlayerID, False)
            if(verbose > 2):
                print("Action: ", action, "pi = ", pi)
            board.ForceAction(action)