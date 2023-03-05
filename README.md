# AlphaZero
An AlphaZero implementation

## Project description
The files

 - AlphaZeroAgent.py
 - AlphaZeroTrainer.py
 - BatchGenerator.py
 - Board.py
 - NeuralNet.py
 - DataAugmenter.py
 - Game.py
 - MCTS.py
 - Node.py

belongs to the actual framework. The purposes of the various components are described in detail in the report, while the others are testing/training scripts (see the sections below for further info).

## Training AlphaZero
The files

 - TrainingConnect4.py
 - TrainingTicTacToe.py

trains, respectively, the Connect 4 and the Tic-Tac-Toe agents. Keep in mind that the Connect 4 agent has actually been trained by running its associated script several times, each time loading the previous session's checkpoint (the commands are left commented in the script). See the report for further details. Finally, keep in mind that the scripts have been tested using Kaggle's cloud computing services because of their prohibitive computational costs and, as a result, the final weights might differ from the ones you will obtain by running it on your machine.

The project's folder structure is such that they are ready to be executed on the project folder itself as long as the requirements have been installed.

## Checkpoints
Connect4_final.pth is the final model, obtained after playing 2500 games.
Connect4_midway.pth is an intermediary checkpoint, obtained after playing 1200 games.
TicTacToe_100_searches_500_epochs.pth is the Tic-Tac-Toe agent's weights and TicTacToe_10_searches_500_epochs.pth is the agent used for the comparison described in section 4.2.1 of the report.

## Agents comparison
The tests which compared the various intermediate checkpoints against each other can be executed by running the PitConnect4Agents.py and PitTicTacToeAgents.py scripts.

## Playing against AlphaZero
Just run the PlayConnectFour.py or PlayTicTacToe.py scripts.

## Additional files
MCTS_surgery_c4.py: used to prove how, despite being unable to defeat a specialized Connect 4 solver, the agent is still capable of guessing correctly 31 of the 41 moves performed by an unbeatable Connect 4 solver on a game it played against itself.

## Package requirements
This project uses
 - OpenAi gym (version 0.26.2. Keep in mind that gym's core methods have changed a lot through the years and the project might not work with previous versions!)
 - Pytorch (I used version 1.12.1+cu116, but other versions might work as well)
 - numpy
 - tqdm
 
There are also two extra packages which are not mandatory, but might be useful in case you want to run some diagnostics:
 - matplotlib: used to plot the training loss
 - graphviz: the Tic-Tac-Toe node contains, commented out, some methods useful to generate a graphviz representation of a board's search tree in a string format. It can then be pasted to websites such as https://dreampuf.github.io/GraphvizOnline/ to plot it.

## Credits
The gym environments for Tic-Tac-Toe and Connect 4 comes from two separate repositories (I have modified them myself to make them compatible with the latest gym version):
 - https://github.com/alfiebeard/tictactoe-gym
 - https://github.com/davidcotton/gym-connect4
