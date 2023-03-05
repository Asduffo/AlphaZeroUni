# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 16:58:39 2023

The alphazero training algorithm.

Putting the training code here avoids any circular dependencies between Game 
and the AlphaZeroAgent itself which is just a container with an MCTS and a 
neural network.

@author: Ninniri Matteo. Student ID: 543873
"""

from matplotlib import pyplot as plt
import torch
import copy as cp
from tqdm import tqdm

from BatchGenerator import BatchGenerator
from Board import ConnectFourBoard, Board
from Game import Game
from AlphaZeroAgent import AlphaZeroAgent

from DataAugmenter import DataAugmenter

"""
A wrapper for the optimizer. Useful if we want to instantiate it at the __init__() call
"""
class OptimizerData():
    def __init__(self,
                 optimizer_type : torch.optim,
                 **kwargs):
        self.optimizer_type = optimizer_type
        self.kwargs         = kwargs
     
"""
A wrapper for the scheduler. Useful if we want to instantiate it at the __init__() call
"""
class SchedulerData():
    def __init__(self,
                 scheduler_type : torch.optim.lr_scheduler,
                 **kwargs):
        self.scheduler_type = scheduler_type
        self.kwargs         = kwargs

class AlphaZeroTrainer():
    def __init__(self,
                BoardClass : Board         = ConnectFourBoard,
                opt_data  : OptimizerData  = OptimizerData(torch.optim.SGD, lr = 10e-2, weight_decay = 10e-4, momentum = .9),
                # opt_data   : OptimizerData = OptimizerData(torch.optim.Adam, lr = .01),
                
                sched_data : SchedulerData = SchedulerData(torch.optim.lr_scheduler.MultiStepLR, milestones=[1000, 2000], gamma = .1),
                augmenter  : DataAugmenter = None,
                device                     = 'cuda'
                ):
        """
        Constructor

        Parameters
        ----------
        BoardClass : Board, optional
            Class of the board relative to the game we're training the agent on. 
            The default is ConnectFourBoard.
        opt_data : OptimizerData, optional
            Optimizer's data. The default is OptimizerData(torch.optim.SGD, lr = 10e-2, weight_decay = 10e-4, momentum = .9).
        sched_data : SchedulerData, optional
            Scheduler's data. The default is SchedulerData(torch.optim.lr_scheduler.MultiStepLR, milestones=[1000, 2000], gamma = .1).
        augmenter : DataAugmenter, optional
            Data augmenter. None means that we do not want to perform augmentation. The default is None.
        device : TYPE, optional
            'cpu' for CPU training, 'cuda' for GPU training. The default is 'cuda'.

        Returns
        -------
        None.

        """
        super().__init__()
        
        self.BoardClass = BoardClass
        self.opt_data   = opt_data
        self.sched_data = sched_data
        self.augmenter  = augmenter
        self.device     = device
    
    def AlphaGoLoss(self, p_tgt, p_prd, v_tgt, v_prd, legals):
        """
        Calculates the loss for a specific batch of data samples. You might notice
        that it has a lot of commented out stuff which are alternative losses
        tested that didn't work as well as CrossEntropy

        Parameters
        ----------
        p_tgt : torch.Tensor
            Target prior probabilities.
        p_prd : torch.Tensor
            Predicted prior probabilities.
        v_tgt : torch.Tensor
            Target board value.
        v_prd : torch.Tensor
            Predicted board value.
        legals : torch.Tensor
            Unused. It contains a mask containing the legal moves.

        Returns
        -------
        loss : torch.Tensor
            the loss.

        """
        # print("p_tgt", p_tgt.size(),", p_prd", p_prd.size(),
        #       ", v_tgt", v_tgt.size(),", v_prd", v_prd.size())
        
        #the value target is just a standard sgd.
        term1 = ((v_tgt - v_prd)**2).reshape((v_tgt.size(0))).float()
        # return term1.mean()
        
        """NLL of the masked softmaxes (the Paper's loss)"""
        # exps = torch.exp(p_prd)
        # masked_exp = exps*legals
        # exps_sum = torch.sum(masked_exp, dim = 1)
        # masked_softmax = masked_exp/exps_sum[:, None] + .00001
        # term2 = p_tgt.float()*torch.log(masked_softmax.float())
        # term2 = torch.sum(term2, dim = 1)
        # loss = (term1 - term2).mean()
        
        """Scalar product of the target against the masked log softmaxes"""
        # logsoftmax = torch.nn.functional.log_softmax(p_prd, dim = 1)
        # term2 = torch.sum(p_tgt.float()*logsoftmax, dim = 1)
        # loss = (term1 - term2).mean()
        
        """Scalar product of the target against the masked softmaxes"""
        # exps = torch.exp(p_prd)
        # masked_exp = exps*legals
        # exps_sum = torch.sum(masked_exp, dim = 1)
        # masked_softmax = masked_exp/exps_sum[:, None]
        # term2 = p_tgt.float() * masked_softmax.float()
        # term2 = torch.sum(term2, dim = 1)
        # loss = (term1 + term2).mean()
        
        """masked MSE"""
        # exps = torch.exp(p_prd)
        # masked_exp = exps.float()*legals.float()
        # exps_sum = torch.sum(masked_exp, dim = 1)
        # masked_softmax = masked_exp/exps_sum[:, None]
        # term2 = (p_tgt.float() - masked_softmax)**2
        # term2 = torch.sum(term2, dim = 1)
        # loss = (term1 + term2).mean()
        
        """Crossentropy"""
        # p_tgt = torch.argmax(p_tgt, dim = 1)
        term2 = torch.nn.CrossEntropyLoss(reduction = 'none')(p_prd, p_tgt)
        loss = (term1 + term2).mean()
        # print(term1.mean(), ", ", term2.mean())
        
        """NLL but with torch's biltin method loss"""
        # logsoftmax = torch.nn.functional.log_softmax(p_prd, dim = 1)
        # p_tgt = torch.argmax(p_tgt, dim = 1)
        # term2 = torch.nn.NLLLoss(reduction = 'none')(logsoftmax, p_tgt)
        # loss = (term1 + term2).mean()
        
        # print("term1", term1.size(), "term2", term2.size())
        
        # return term2.mean()
        
        return loss
    
    def fit(self,
            agent                : AlphaZeroAgent, #the agent to train
            n_iters              : int =  64,   #number of games to play
            dataset_sz_threshold : int = 256,   #size the dataset must have before performing a weight update
            batch_sz             : int =   2,   #batch size
            n_eval_games         : int =   4,   #how many games f_n will play against f_0 at each evaluation
            train_epochs         : int =   1,
        ):
        """
        Plays n_iters games by self-play and trains the agent's nn using the data obtained.

        Parameters
        ----------
        agent : AlphaZeroAgent
            The agent we want to train.
        n_iters : int, optional
            Number of games to play. The default is 64.
        dataset_sz_threshold : int, optional
            size the dataset must have before performing a weight update. The default is 256.
        batch_sz : int, optional
            Size of the batches fed to the agent's NN. The default is 2.
        n_eval_games : int, optional
            how many games f_n will play against f_0 at each evaluation. The default is 4.
        train_epochs : int, optional
            PLEASE DO NOT TOUCH. If > 1, at each training step the
            trainer will feed the dataset train_epochs times to the network. 
            Increasing this is plain wrong and it was only used for debugging purposes
            The default is 1.

        Returns
        -------
        None.

        """
        
        #starts the optimizer and the scheduler
        optimizer = self.opt_data.optimizer_type(agent.f_0.parameters(), **(self.opt_data.kwargs))
        scheduler = self.sched_data.scheduler_type(optimizer, **(self.sched_data.kwargs))
        
        #dataset pool
        dataset = []
        
        #history
        self.loss = []
        
        #prograss bar
        pbar1 = tqdm(range(n_iters))
        for i in pbar1:
            #plays a bunch of games by itself. no need for gradients until well later
            with torch.no_grad():
                newBoard = self.BoardClass()
                gameData = agent.mcts.SelfPlay(newBoard, train = True)
                dataset  = dataset + gameData
                
                # print("======================================================")
                # print(gameData)
                # print("======================================================")

                #augmentation, if any
                if(self.augmenter is not None):
                    dataset = dataset + self.augmenter.Augment(gameData)
            
            #if we have enough data for a training epoch:
            if(len(dataset) >= dataset_sz_threshold):
                data_size = len(dataset)
                
                #in case of a 2d tensor as a board:
                if(len(dataset[0][0].shape) == 2):
                    #it turns it into a 3d tensor
                    X = torch.zeros(size = (data_size, 1, *dataset[0][0].shape)).to(self.device)
                else:
                    X = torch.zeros(size = (data_size, *dataset[0][0].shape)).to(self.device)
                
                p = torch.zeros(size = (data_size, 1, *dataset[0][1].shape)).to(self.device)
                v = torch.zeros(size = (data_size, 1, 1)).to(self.device)
                l = torch.zeros(size = (data_size, 1, *dataset[0][3].shape)).to(self.device)
                # print("X.size() = ", X.size(), ", p.size() = ", p.size(), ", v.size() = ", v.size())
                
                for j in range(data_size):
                    X[j, :] = dataset[j][0]
                    p[j, :] = dataset[j][1]
                    v[j, :] = dataset[j][2]
                    l[j, :] = dataset[j][3]
                
                # pbar_train = tqdm(b, leave=False)
                for k in range(train_epochs):
                    #splits the 3 datasets into batches
                    b = BatchGenerator(data_size, batch_sz, random_state = None).get_batches()
                    
                    iteration = 1
                    tr_loss = 0
                    
                    #iterates through the batches
                    for batch in b:
                        #extracts the elements
                        X_b = X[batch, :]
                        p_b = p[batch, :]
                        v_b = v[batch, :]
                        l_b = l[batch, :]
                        
                        pred, value = agent.f_0(X_b)
                        
                        p_b = p_b.reshape((p_b.size(0), -1))
                        v_b = v_b.reshape((v_b.size(0), -1))
                        l_b = l_b.reshape((l_b.size(0), -1))
                        
                        #loss calculation and backpropagation as usual
                        loss = self.AlphaGoLoss(p_b, pred, v_b, value, l_b)
                        
                        if(torch.isnan(loss).item()):
                            print("WARNING: nan Loss encountered")
                        
                        #pytorch's optimizer routine
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        #verbose stuff
                        loss_det = loss.clone().detach()
                        tr_loss += loss_det
                        
                        average_loss = (tr_loss/iteration).item()
                        # pbar_train.set_description("Average loss: %s" % average_loss.item())
                        pbar1.set_description("Average loss: %s" % average_loss)
                        
                        # print("pred", torch.nn.functional.softmax(pred, dim = 1), 
                        #       "p_b, ", p_b, 
                        #       ", value", value.view(-1), 
                        #       ", v_b", v_b.view(-1))
                        
                        iteration += 1
                
                self.loss.append(average_loss)
                
                #now we test the model against f_0, if we want to (unused in the final implementation)
                with torch.no_grad():
                    #used for evaluation
                    self.f_n = cp.deepcopy(agent.f_0)
                    
                    n_won = 0
                    n_drawn = 0
                    percentage_won = 0
                    
                    for j in range(n_eval_games):
                        g = Game()
                        
                        #wether f_n should be player 1 or player 2
                        f_n_player1 = j % 2 == 0
                        
                        eval_board = self.BoardClass()
                        
                        #dummy agent
                        generator_agent = AlphaZeroAgent(f_0              = self.f_n,
                                                           cpuct            = agent.cpuct,
                                                           n_iters_per_move = agent.n_iters_per_move,
                                                           device           = agent.device)
                        
                        #plays the game
                        if(f_n_player1): 
                            outcome = g.Play(eval_board, generator_agent, agent)
                            if(outcome == -1):
                                n_won += 1
                            elif(outcome == 0):
                                n_drawn += 1
                        else:
                            outcome = g.Play(eval_board, agent, generator_agent)
                            if(outcome == 1):
                                n_won += 1
                            elif(outcome == 0):
                                n_drawn += 1
                        
                        pbar1.set_description("Game %s moves: %s" % (j, eval_board.GetCurrentTurn()))
                    
                    #win rate
                    if n_eval_games > 0:
                        percentage_won = n_won/n_eval_games
                
                print("\niteration", i, "games won percentage =", percentage_won, 
                      "(won:", n_won, ", drawn = ", n_drawn, "). average_loss = ", average_loss)
                
                #discharges the data pool
                dataset = []
            
            scheduler.step()
            
    def PlotLoss(self, ylim = None):
        """ 
            Plots the loss graph
        """
        plt.figure(dpi=500)
        plt.xlabel('Weight update iteration')
        plt.ylabel("Loss")
        
        plt.plot(self.loss, 'r')
        
        plt.show()