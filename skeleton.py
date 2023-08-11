import importlib
import os
import sys
from typing import Tuple
from gym.spaces import Box
from gym import Env
import numpy as np
import torch
from torch import tensor, Tensor
from torch.utils.data import DataLoader
from humor.body_model.utils import SMPL_JOINTS
from humor.models.humor_model import HumorModel
from torch.distributions import Normal, kl_divergence
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation
from humor.train.train_state_prior import build_pytorch_gmm, load_gmm_results

from humor.utils.transforms import compute_world2aligned_mat, rotation_matrix_to_angle_axis
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

    def __init__(self, args_obj, init_pose, obstacles):

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
        self.action_space = Box(low=-np.ones(48),high=np.ones(48)) # the action, is a disturb of the humor z
        self.observation_space = Box(low=-10*np.ones(141),high=10*np.ones(141)) # agent move in real world space
        self.init_pose = init_pose # initial pose, in world coo
        self.observation = self.init_pose # initialize the agent state
        print('Visualizing INITIAL!')
        # print('66INIT',joints)
        self.max_simu = 300
        self.walking_length = self.max_simu
        self.obstacles = obstacles # obstacles in the environment, in format of triangles; 3x3xn
        self.scatter_plots = []
        self.loss = []
        self.epi = 0
        gmm_out_path = os.path.join('checkpoints/init_state_prior_gmm/prior_gmm.npz')
        self.gmm(gmm_out_path)
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)
        self.fig.suptitle('Learning supervisor')


        pose = tensor(self.init_pose, device='cuda:0').reshape(1,1,-1)
        init_dict = self.HuMoR.split_output(pose)
        del init_dict['contacts']
        self.state = self.VAE339toGMM138(init_dict) # initialize the agent state
        self.roll_out_init(pose, init_dict)
        # print("INIT INI", self.init_pose)
    
    def update(self, frame):
        # plt.cla()  # Clear the previous plot
        self.ax1.cla()
        self.ax2.cla()
        self.ax1.plot([-0.7, -0.7], [-1, 2], color='red', label='y=x line')
        self.ax1.plot([0.7, 0.7], [-1, 2], color='red', label='y=x line')
        self.ax1.scatter(self.scatter_plots[frame][:,0], self.scatter_plots[frame][:,2], c='b', marker='o')  # Create the scatter plot
        if self.loss: self.ax1.text(0.1,0.1,'phy:'+str(self.loss[frame][2]))
        self.ax1.set_xlim([-1, 1])
        self.ax1.set_ylim([-1, 2])
        self.ax2.scatter(self.scatter_plots[frame][:,1], self.scatter_plots[frame][:,2], c='b', marker='o')  # Create the scatter plot
        self.ax2.set_xlim([-1, 1])
        self.ax2.set_ylim([-1, 2])

    def physics_loss(self):
        '''
        Compute the physics loss, for example penetration loss, joints beneath ground loss
        Also the reward when the agent get close to target
        This loss enable the agent to avoid obstacles in its way, and move towards the target
        past_coo, current_coo need to extract from world coordinates, represent the joints positions
        '''
        # for penetration, see 322 rasterized line 151
        # return whether the current config is in restricted region
        trans = self.x_pred_dict['trans']
        # all_pos = joints+trans
        # if torch.max(torch.abs(trans[:,:,0])) > 0.7:
        #     return 100
        return torch.max(torch.abs(trans[:,:,0]))
    
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

    def VAE339toGMM138(self, state, trans=True):
        '''
        transform the VAE 339D state, to GMM 138D data.
        '''
        B = state['joints'].size(0)
        # print(batch_in['joints'].shape, batch_in['joints_vel'].shape, batch_in['trans_vel'].shape, batch_in['root_orient_vel'].shape)
        _trans = state['trans'].reshape((1, -1))
        joints = state['joints'].reshape((1, -1))
        joints_vel = state['joints_vel'].reshape((1, -1))
        trans_vel = state['trans_vel'].reshape((1, -1))
        root_orient_vel = state['root_orient_vel'].reshape((1, -1))
        data = [joints, joints_vel, trans_vel, root_orient_vel]
        if trans: data.append(_trans)
        cur_state = torch.cat(data, dim=-1)
        return cur_state.cpu().detach()

    def GMM_reward(self, batch_in):
        '''
        This loss is trying to keep the transitioned state still on the feasible manifold.
        '''
        cur_state = self.VAE339toGMM138(batch_in, False)
        # print(cur_state.device)

        # eval likelihood
        test_logprob = self.gmm_distrib.log_prob(cur_state)
        mean_logprob = test_logprob.mean()

        # print('Mean test logprob: %f' % (mean_logprob.item()))
        # the more unlikely current state on the GMM model, the more penalty it suffers
        return mean_logprob

    def step(self, action):
        '''
        A step in this environment: according to previous latent state, the humor model take its default action,
        then the agent add on a trained increment of the default humor action, base on the current state given by humor. 
        '''
        # print('DO STEP')
        self.walking_length -= 1
        tensor_state = tensor(self.observation, device='cuda:0')
        action = tensor(action*1e-2, device='cuda:0')
        decoded = self.roll_out_step(action=action)
        # print('DECODE',decoded)
        new_state = decoded.cpu().detach().numpy()# + action*3e-3 # add the action increment onto the default humor model
        tensor_new_state = tensor(new_state, device='cuda:0')
        # self.simple_vis(tensor_new_state, tensor_state)
        self.vid_vis(tensor_state) # CAUTION: this will not show global position
        gmm_reward = self.GMM_reward(self.x_pred_dict).item()*0
        humor_loss = self.HuMoR_loss(tensor_state, tensor_new_state).item()
        physics_loss = self.physics_loss().item()
        self.loss.append((gmm_reward,humor_loss,physics_loss))
        print('GMM:', gmm_reward, 'humor:', humor_loss, 'phy:', physics_loss)
        reward = gmm_reward-humor_loss-physics_loss
        # print('REWARD SHAPE', reward.shape)
        self.observation = new_state
        self.state = self.VAE339toGMM138(self.x_pred_dict)
        done = self.walking_length <= 0
        # reward = 0
        info = {'world':0}
        self.render()

        # print('STEP OVER')
        return self.state, reward, done, info

    def roll_out_init(self, x_past, init_input_dict):
        '''
        Given input for first step, roll out using own output the entire time by sampling from the prior.
        Returns the global trajectory.

        Input:
        - x_past (B x steps_in x D_in)
        - initial_input_dict : dictionary of each initial state (B x steps_in x D), rotations should be matrices
                                (assumes initial state is already in its local coordinate system (translation at [0,0,z] and aligned))
        Returns: 
        - x_pred - dict of (B x num_steps x D_out) for each value. Rotations are all matrices.
        '''
        print('ROLLOUT', x_past.shape, {k:init_input_dict[k].shape if type(init_input_dict[k]) == torch.Tensor else init_input_dict[k] for k in init_input_dict})
        J = len(SMPL_JOINTS)
        self.cur_input_dict = init_input_dict

        # need to transform init input to local frame
        self.world2aligned_rot = self.world2aligned_trans = None
        # check to make sure we have enough input steps, if not, pad
        pad_x_past = x_past is not None and x_past.size(1) < self.HuMoR.steps_in
        pad_in_dict = self.cur_input_dict[list(self.cur_input_dict.keys())[0]].size(1) < self.HuMoR.steps_in
        if pad_x_past:
            num_pad_steps = self.HuMoR.steps_in -  x_past.size(1)
            cur_padding = torch.zeros((x_past.size(0), num_pad_steps, x_past.size(2))).to(x_past) # assuming all data is B x T x D
            x_past = torch.cat([cur_padding, x_past], axis=1)
        if pad_in_dict:
            for k in self.cur_input_dict.keys():
                cur_in_dat = self.cur_input_dict[k]
                num_pad_steps = self.HuMoR.steps_in - cur_in_dat.size(1)
                cur_padding = torch.zeros((cur_in_dat.size(0), num_pad_steps, cur_in_dat.size(2))).to(cur_in_dat) # assuming all data is B x T x D
                padded_in_dat = torch.cat([cur_padding, cur_in_dat], axis=1)
                self.cur_input_dict[k] = padded_in_dat
        
        B, S, D = x_past.size()
        self.past_in = x_past.reshape((B, -1))

        self.global_world2local_rot = torch.eye(3).reshape((1, 1, 3, 3)).expand((B, 1, 3, 3)).to(x_past)
        self.global_world2local_trans = torch.zeros((B, 1, 3)).to(x_past)
        self.trans2joint = torch.zeros((B,1,1,3)).to(x_past)
        if self.HuMoR.need_trans2joint:
            self.trans2joint = -torch.cat([self.cur_input_dict['joints'][:,-1,:2], torch.zeros((B, 1)).to(x_past)], axis=1).reshape((B,1,1,3)) # same for whole sequence
        self.pred_local_seq = []
        self.pred_global_seq = []

        self.x_past = x_past
        # for t in range(num_steps):
        #     pass

    def roll_out_step(self, use_mean=False, action=None):
        '''
        This is splited from the HuMoR model function roll_out, and transfered to the version to fit in the step function of RL framework. 
        it can take in action that act as offset in latent space
        '''
        J = len(SMPL_JOINTS)
        B, S, D = self.x_past.size()

        x_pred_dict = None
        # print('self.x_past', self.x_past)
        # sample next step, provide offset, from the action RL take
        sample_out = self.HuMoR.sample_step(self.past_in, use_mean=use_mean, offset=action)
        decoder_out = sample_out['decoder_out']

        # split output predictions and transform out rotations to matrices
        x_pred_dict = self.HuMoR.split_output(decoder_out, convert_rots=True)
        # print('sample_out', sample_out, self.x_pred_dict)
        if self.HuMoR.steps_out > 1:
            for k in x_pred_dict.keys():
                # only want immediate next frame prediction
                x_pred_dict[k] = x_pred_dict[k][:,0:1,:]

        self.pred_local_seq.append(x_pred_dict)

        # output is the actual regressed joints, but input to next step can use smpl joints
        x_pred_smpl_joints = None

        # prepare input to next step
        # update input dict with new frame
        del_keys = []
        for k in self.cur_input_dict.keys():
            if k in x_pred_dict:
                # drop oldest frame and add new prediction
                keep_frames = self.cur_input_dict[k][:,1:,:]
                # print(keep_frames.size())

                if k == 'joints' and self.HuMoR.use_smpl_joint_inputs and x_pred_smpl_joints is not None:
                    self.cur_input_dict[k] = torch.cat([keep_frames, x_pred_smpl_joints], axis=1)
                else:
                    self.cur_input_dict[k] = torch.cat([keep_frames, x_pred_dict[k]], axis=1)
            else:
                del_keys.append(k)
        for k in del_keys:
            del self.cur_input_dict[k]

        # get world2aligned rot and translation
        root_orient_mat = x_pred_dict['root_orient'][:,0,:].reshape((B, 3, 3))
        world2aligned_rot = compute_world2aligned_mat(root_orient_mat)
        world2aligned_trans = torch.cat([-x_pred_dict['trans'][:,0,:2], torch.zeros((B,1)).to(self.x_past)], axis=1)

        #
        # transform inputs to this local frame (body pose is not affected) for next step
        #
        self.cur_input_dict = self.HuMoR.apply_world2local_trans(world2aligned_trans, world2aligned_rot, self.trans2joint, self.cur_input_dict, self.cur_input_dict, invert=False)

        # convert rots to correct input format
        if self.HuMoR.in_rot_rep == 'aa':
            if 'root_orient' in self.HuMoR.data_names:
                self.cur_input_dict['root_orient'] = rotation_matrix_to_angle_axis(self.cur_input_dict['root_orient'].reshape((B*S,3,3))).reshape((B, S, 3))
            if 'pose_body' in self.HuMoR.data_names:
                self.cur_input_dict['pose_body'] = rotation_matrix_to_angle_axis(self.cur_input_dict['pose_body'].reshape((B*S*(J-1),3,3))).reshape((B, S, (J-1)*3))
        elif self.HuMoR.in_rot_rep == '6d':
            if 'root_orient' in self.HuMoR.data_names:
                self.cur_input_dict['root_orient'] = self.cur_input_dict['root_orient'][:,:,:6]
            if 'pose_body' in self.HuMoR.data_names:
                self.cur_input_dict['pose_body'] = self.cur_input_dict['pose_body'].reshape((B, S, J-1, 9))[:,:,:,:6].reshape((B, S, (J-1)*6))

        #
        # compute current world output and update world2local transform
        #
        cur_world_dict = dict()
        cur_world_dict = self.HuMoR.apply_world2local_trans(self.global_world2local_trans, self.global_world2local_rot, self.trans2joint, x_pred_dict, cur_world_dict, invert=True)
        #
        # update world2local transform
        # Disable this can remain the agent in the center, but not recommanded
        self.global_world2local_trans = torch.cat([-cur_world_dict['trans'][:,0:1,:2], torch.zeros((B, 1, 1)).to(self.x_past)], axis=2)
        # print(world2aligned_rot)
        self.global_world2local_rot = torch.matmul(self.global_world2local_rot, world2aligned_rot.reshape((B, 1, 3, 3)))

        self.pred_global_seq.append(cur_world_dict)

        # cat all inputs together to form past_in
        in_data_list = []
        for k in self.HuMoR.data_names:
            in_data_list.append(self.cur_input_dict[k])


        # prepare for next frame
        self.past_in = torch.cat(in_data_list, axis=2)
        self.past_in = self.past_in.reshape((B, -1))
        self.x_pred_dict = x_pred_dict
        # print('PAST_IN', self.past_in.shape)
        return self.past_in

    def roll_out_after(self):

        # aggregate global pred_seq
        pred_seq_out = dict()
        for k in self.pred_global_seq[0].keys():
            pred_seq_out[k] = torch.cat([self.pred_global_seq[i][k] for i in range(len(self.pred_global_seq))], axis=1)
        
        return pred_seq_out
        
    def render(self, mode='human'):
        # Your visualization code here
        # For example, create a scatter plot of the environment state
        pass

    def reset(self):
        self.observation = self.init_pose
        self.walking_length = self.max_simu
        self.epi += 1
        if self.scatter_plots:
            # fig, ax = plt.subplots()
            ans = self.roll_out_after() # this zanshi over write the vid_vis in step function
            self.scatter_plots = ans['joints'].reshape((300,22,3)).cpu().detach().numpy()
            ani = FuncAnimation(self.fig, self.update, frames=len(self.scatter_plots))
            saving = 'animation{}.mp4'.format(self.epi)
            ani.save(saving, writer='ffmpeg', fps=30)
            print('SAVING', saving)
        self.scatter_plots = []
        self.loss = []
        # same as in __init__
        pose = tensor(self.init_pose, device='cuda:0').reshape(1,1,-1)
        init_dict = self.HuMoR.split_output(pose)
        del init_dict['contacts']
        self.state = self.VAE339toGMM138(init_dict) # initialize the agent state
        self.roll_out_init(pose, init_dict)
        return self.state

    def to_plot_pose(self, pose):
        x_pred_dict = self.HuMoR.split_output(pose)
        joints = x_pred_dict['joints'].reshape((22,3)).cpu().detach().numpy().T
        return joints[0], joints[2]

    def simple_vis(self, pose, old=None):
        x, y = self.to_plot_pose(pose)
        # print('66INIT',joints)
        plt.axis('equal')
        plt.scatter(x,y)
        if old is not None:
            x_old_dict = self.HuMoR.split_output(old)
            old_joints = x_old_dict['joints'].reshape((22,3)).cpu().numpy().T
            plt.scatter(old_joints[0], old_joints[2], c='red')
        plt.show()

    def vid_vis(self, pose):
        x_pred_dict = self.HuMoR.split_output(pose)
        joints = x_pred_dict['joints'].reshape((22,3)).cpu().numpy()
        self.scatter_plots.append(joints)

    def default_roll_out(self):
        pose = tensor(self.init_pose, device='cuda:0').reshape(1,1,-1)
        init_dict = self.HuMoR.split_output(pose)
        del init_dict['contacts']
        ans = self.HuMoR.roll_out(pose, init_dict, 300)
        print('ROLLING', {k:ans[k].shape if type(ans[k])== Tensor else ans for k in ans})
        self.scatter_plots = ans['joints'].reshape((300,22,3)).cpu().detach().numpy()
        fig, ax = plt.subplots()
        ani = FuncAnimation(self.fig, self.update, frames=self.scatter_plots.shape[0])
        saving = 'roll_out.mp4'
        ani.save(saving, writer='ffmpeg', fps=30)
        print('SAVING ROLL', saving)

    def default_roll_out_split(self):
        pose = tensor(self.init_pose, device='cuda:0').reshape(1,1,-1)
        init_dict = self.HuMoR.split_output(pose)
        del init_dict['contacts']
        self.roll_out_init(pose, init_dict)
        for t in range(self.max_simu): self.roll_out_step()
        ans = self.roll_out_after()
        # ans = self.HuMoR.roll_out(pose, init_dict, 300)
        print('ROLLING', {k:ans[k].shape if type(ans[k])== Tensor else ans for k in ans})
        self.scatter_plots = ans['joints'].reshape((300,22,3)).cpu().detach().numpy()
        # fig, ax = plt.subplots()
        ani = FuncAnimation(self.fig, self.update, frames=self.scatter_plots.shape[0])
        saving = 'roll_out.mp4'
        ani.save(saving, writer='ffmpeg', fps=30)
        print('SAVING ROLL', saving)

    def gmm(self, gmm_path):
        #
        # Evaluate likelihood of test data
        #

        # load in GMM result
        gmm_weights, gmm_means, gmm_covs = load_gmm_results(gmm_path)

        # build pytorch distrib
        self.gmm_distrib = build_pytorch_gmm(gmm_weights, gmm_means, gmm_covs)
