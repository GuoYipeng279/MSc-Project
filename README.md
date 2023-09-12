# myProject

An reinforcement learning extension built on HuMoR model. 

![Navigation](55.png)

Model structure
![Model](f7.png)

This repo contains the code used in the project. The repo should be placed under the HuMoR repo folder. 
The environment is also build on HuMoR environment. 
Required packages are in requirements-rl.txt, which includes the HuMoR requirements. 
```
pip install -r requirements-rl.txt
```
All manipulation and commands are under the HuMoR repo folder, on a GPU divice, activating the environment. 
To a test time sample run:
Download HuMoR model.
Activate the environment.
Install the packages.
Then cd to humor repo (not myProject repo).
Copy humor.sh, to under the HuMoR folder.
Replace the file humor/models/humor_model.py to myProject/humor_model.py. 
Connect to a GPU server, for example:
```
ssh gpucluster2.doc.ic.ac.uk
```
Then cd to humor repo (not myProject repo), on the server.
Then run:
```
sbatch humor.sh 4
```
Every running will be slightly different beacuse of randomness of the model. 
A simulation result start with 'run_his_...' will appear under the HuMoR folder.
Put this file name into the .blend file visualize.blend line 12 to visualize(Please delete previous skeletons in the scene, if any). 
In line 13 there is a parameter control the sample interval, usually around 30, can float +-1~2.
There are .npy result files under the results folder for quick visualization.

For training:
```
sbatch humor.sh 0
```
A trained model start with 'agent...' will appear under the HuMoR folder.

On a GPU avaliable machine, the file main.py can be run with argument 0 for training, 4 for testing directly.