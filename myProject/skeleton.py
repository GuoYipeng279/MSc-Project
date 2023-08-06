import importlib
import os
import sys
from typing import Tuple
from gym.spaces import Box
from gym import Env
import numpy as np
import torch
from torch import tensor
from torch.utils.data import DataLoader
from humor.models.humor_model import HumorModel
from torch.distributions import Normal, kl_divergence
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation
NUM_WORKERS = 0

from humor.utils.torch import get_device, load_state
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

from humor.utils.logging import Logger, class_name_to_file_name, cp_files, mkdir

class Skeleton(Env):
    '''
    Idea: the state space is the observable space(339D), the action space is some increment of the state space
    for each step, use the humor model to provide a default next state, the actor also provide a small increment, according to this default state.
    Add the humor increment and actor increment together to get the next state.
    To evaluate the generated state, the reward includes:
    input the previous state and the generated state to the encoder of humor, compare similarity of the latent distribution to the prior distribution.
    deduct the physics loss of the generated state, these forms the critic part. 
    '''

    def __init__(self, args_obj, config_file, init_pose, obstacles):

        def test(args_obj) -> Tuple[HumorModel, DataLoader]:

            # set up output
            args = args_obj.base
            mkdir(args.out)

            # create logging system
            test_log_path = os.path.join(args.out, 'test.log')
            Logger.init(test_log_path)

            # save arguments used
            Logger.log('Base args: ' + str(args))
            Logger.log('Model args: ' + str(args_obj.model))
            Logger.log('Dataset args: ' + str(args_obj.dataset))
            Logger.log('Loss args: ' + str(args_obj.loss))

            # save training script/model/dataset/config used
            test_scripts_path = os.path.join(args.out, 'test_scripts')
            mkdir(test_scripts_path)
            # pkg_root = os.path.join(cur_file_path, '..')
            dataset_file = class_name_to_file_name(args.dataset)
            # dataset_file_path = os.path.join(pkg_root, 'datasets/' + dataset_file + '.py')
            model_file = class_name_to_file_name(args.model)
            loss_file = class_name_to_file_name(args.loss)
            # model_file_path = os.path.join(pkg_root, 'models/' + model_file + '.py')
            # cp_files(test_scripts_path, [model_file_path, dataset_file_path, config_file])

            # load model class and instantiate
            model_class = importlib.import_module('models.' + model_file)
            Model = getattr(model_class, args.model)
            model = Model(**args_obj.model_dict,
                            model_smpl_batch_size=args.batch_size) # assumes model is HumorModel

            # load loss class and instantiate
            loss_class = importlib.import_module('losses.' + loss_file)
            Loss = getattr(loss_class, args.loss)
            loss_func = Loss(**args_obj.loss_dict,
                            smpl_batch_size=args.batch_size*args_obj.dataset.sample_num_frames) # assumes loss is HumorLoss

            device = get_device(args.gpu)
            model.to(device)
            loss_func.to(device)

            print(model)

            # count params
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            Logger.log('Num model params: ' + str(params))

            # freeze params in loss
            for param in loss_func.parameters():
                param.requires_grad = False

            # load in pretrained weights if given
            if args.ckpt is not None:
                start_epoch, min_val_loss, min_train_loss = load_state(args.ckpt, model, optimizer=None, map_location=device, ignore_keys=model.ignore_keys)
                Logger.log('Successfully loaded saved weights...')
                Logger.log('Saved checkpoint is from epoch idx %d with min val loss %.6f...' % (start_epoch, min_val_loss))
            else:
                Logger.log('ERROR: No weight specified to load!!')
                # return

            # load dataset class and instantiate training and validation set
            if args.test_on_train:
                Logger.log('WARNING: running evaluation on TRAINING data as requested...should only be used for debugging!')
            elif args.test_on_val:
                Logger.log('WARNING: running evaluation on VALIDATION data as requested...should only be used for debugging!')
            Dataset = getattr(importlib.import_module('datasets.' + dataset_file), args.dataset)
            split = 'test'
            if args.test_on_train:
                split = 'train'
            elif args.test_on_val:
                split = 'val'
            test_dataset = Dataset(split=split, **args_obj.dataset_dict)
            # create loaders
            test_loader = DataLoader(test_dataset, 
                                    batch_size=args.batch_size,
                                    shuffle=args.shuffle_test, 
                                    num_workers=NUM_WORKERS,
                                    pin_memory=True,
                                    drop_last=False,
                                    worker_init_fn=lambda _: np.random.seed())

            test_dataset.return_global = True
            model.dataset = test_dataset

            return model, test_loader

        self.HuMoR, self.test_loader = test(args_obj)
        self.action_space = Box(low=-np.ones(339),high=np.ones(339)) # the action, is a disturb of the humor z
        self.observation_space = Box(low=-10*np.ones(339),high=10*np.ones(339)) # agent move in latent space
        self.init_pose = init_pose # initial pose, in world coo
        self.state = self.init_pose # initialize the agent state
        print('Visualizing INITIAL!')
        # print('66INIT',joints)
        self.max_simu = 10
        self.walking_length = self.max_simu
        self.obstacles = obstacles # obstacles in the environment, in format of triangles; 3x3xn
        self.fig, ax = plt.subplots()
        self.sc = ax.scatter([], [])  # Empty scatter plot
        self.scatter_plots = []
        # print("INIT INI", self.init_pose)

    def init(self):
        self.sc.set_offsets([])  # Clear the scatter plot
        return (self.sc,)
    
    def update(self, frame):
        data = self.scatter_plots[frame]  # Get the data for the current frame
        self.sc.set_offsets(data)  # Set the new data for the scatter plot
        return (self.sc,)

    def physics_loss(self, past_coo, cur_coo):
        '''
        Compute the physics loss, for example penetration loss, joints beneath ground loss
        Also the reward when the agent get close to target
        This loss enable the agent to avoid obstacles in its way, and move towards the target
        past_coo, current_coo need to extract from world coordinates, represent the joints positions
        '''
        # for penetration, see 322 rasterized line 151
        # for each bone AB to CD, check AB, CD, AC, BD with all triangles
        # old, new = self.HuMoR.split_output(past_coo)['joints'], self.HuMoR.split_output(cur_coo)['joints']
        # old, new are 66D vectors, indicate the joint position. 
        return 0
    
    def HuMoR_loss(self, past_coo, cur_coo):
        '''
        Evaluate how the generated current state match the HuMoR opinion, by compare the posterior distribution to the prior. 
        After the edition form RL side, the state should not be too distorted. 
        If no addition from RL side, this loss should be close to 0. 
        '''
        prior, var1 = self.HuMoR.prior(past_coo)
        posterior, var2 = self.HuMoR.posterior(past_coo, cur_coo)
        var = torch.max(var1,var2)
        # print('COMPARE', prior, posterior, var1, var2)
        diss = kl_divergence(Normal(prior, var), Normal(posterior, var))
        # print('COMPARE', torch.cat((diss, prior, posterior, var1, var2)).T)
        # x,y = prior.cpu().detach().numpy().squeeze(), posterior.cpu().detach().numpy().squeeze()
        # xv,yv = var.cpu().detach().numpy().squeeze(), var.cpu().detach().numpy().squeeze()
        # plt.axis('equal')
        # plt.scatter(x,y)
        # min_val = min(min(x), min(y))
        # max_val = max(max(x), max(y))
        # plt.plot([min_val, max_val], [min_val, max_val], color='red', label='y=x line')
        # for i, txt in enumerate(diss.cpu().detach().numpy().squeeze()):
        #     plt.annotate(txt, (x[i], y[i]))
        #     ellipse = Ellipse((x[i], y[i]), xv[i], yv[i], edgecolor='red', facecolor='none')
        #     plt.gca().add_patch(ellipse)
        # plt.show()
        diss = diss.mean(1)
        # print('DISS VALUE', diss)
        # raise NotImplementedError
        # return 0
        return diss
        # mean, var = self.HuMoR.prior(past_coo) # get the humor prior, according to past coo
        # mean, var = mean.cpu().detach().numpy().squeeze(), var.cpu().detach().numpy().squeeze()
        # # print(mean.shape, var.shape)
        # normal = multivariate_normal(mean, var)
        # return normal.pdf(cur_sta.cpu().detach().numpy().squeeze()) # probability of getting to this position in latent space, according to humor

    def step(self, action):
        '''
        A step in this environment: according to previous latent state, the humor model take its default action,
        then the agent add on a trained increment of the default humor action, base on the current state given by humor. 
        '''
        # print('DO STEP')
        self.walking_length -= 1
        tensor_state = tensor(self.state, device='cuda:0')
        prior, var1 = self.HuMoR.prior(tensor_state)
        z = self.HuMoR.rsample(prior, var1)
        decoded = self.HuMoR.decode(z, tensor_state)[:,:339]
        # print('DECODE',decoded)
        new_state = decoded.cpu().detach().numpy()# + action # add the action increment onto the default humor model
        tensor_new_state = tensor(new_state, device='cuda:0')
        # self.simple_vis(tensor_new_state, tensor_state)
        reward = -self.HuMoR_loss(tensor_state, tensor_new_state)-self.physics_loss(tensor_state, tensor_new_state)
        # print('REWARD SHAPE', reward.shape)
        self.state = new_state
        done = self.walking_length <= 0
        # reward = 0
        info = {'world':0}
        self.render()

        # print('STEP OVER')
        return self.state, reward, done, info

    def render(self, mode='human'):
        # Your visualization code here
        # For example, create a scatter plot of the environment state
        self.vid_vis(tensor(self.state))

    def reset(self):
        # print('DO RESET')
        self.state = self.init_pose
        self.walking_length = self.max_simu
        self.scatter_plots = []
        self.ani = FuncAnimation(self.fig, self.update, frames=len(self.scatter_plots), init_func=self.init, blit=True)
        plt.show()
        # print('RESET OVER')
        return self.state

    def simple_vis(self, pose, old=None):
        x_pred_dict = self.HuMoR.split_output(pose)
        joints = x_pred_dict['joints'].reshape((22,3)).cpu().numpy().T
        # print('66INIT',joints)
        plt.axis('equal')
        plt.scatter(joints[0], joints[2])
        if old is not None:
            x_old_dict = self.HuMoR.split_output(old)
            old_joints = x_old_dict['joints'].reshape((22,3)).cpu().numpy().T
            plt.scatter(old_joints[0], old_joints[2], c='red')
        plt.show()

    def vid_vis(self, pose):
        x_pred_dict = self.HuMoR.split_output(pose)
        joints = x_pred_dict['joints'].reshape((22,3)).cpu().numpy().T
        self.scatter_plots.append((joints[0], joints[2]))
