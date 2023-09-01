import importlib
import os
import sys
from typing import Tuple
from gym.spaces import Box, Discrete
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

navigation_controller = ['forward','right','left']

class Skeleton(Env):

    def __init__(self, args_obj, init_pose, controller=None, representer="VAE339toGMM138"):
        '''
        Initialize the training environment.
        For controller(list), list which action dimensions are opened to the agent.
        For representer(str), method to encode the states from VAE output.
        '''

        def test(args_obj) -> Tuple[HumorModel, DataLoader]:
            '''
            Borrowed form humor code, to load in humor archit, and trained model.
            '''
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
        self.action_space = Box(low=-2*np.ones(48),high=2*np.ones(48)) # the action, is a disturb of the humor z
        self.controller = controller
        if controller is not None:
            # set up the action space like the shape of the controller.
            # self.action_space = Box(low=-1.1*np.ones_like(controller),high=-0.9*np.ones_like(controller)) # the action, is a disturb of the humor z
            self.action_space = Discrete(len(controller)) # action selector
        else:
            self.action_space = Box(low=-2.*np.ones(48),high=2.*np.ones(48)) # the action, is a disturb of the humor z
        self.state_encoder = lambda x:x # not recommand, this results using the raw 339D data, too high.
        if representer == "VAE339toGMM138":
            self.observation_space = Box(low=-10*np.ones(141),high=10*np.ones(141)) # agent move in real world space
            self.state_encoder = self.VAE339toGMM138 # get down to 138/141D, the GMM uses
        if representer == "mean":
            self.observation_space = Box(low=-10*np.ones(6),high=10*np.ones(6)) # agent move in real world space
            self.state_encoder = self.VAE339toMEAN6 # to very low 6D mean position data, use when got a very clear view in advance
        if representer == "GLOBAL5":
            self.observation_space = Box(low=-10*np.ones(5),high=10*np.ones(5)) # agent move in real world space
            self.state_encoder = self.GLOBAL5 # to very low 6D mean position data, use when got a very clear view in advance
        if representer == "RELA5": # use this for this project
            # space: [x,y,dx,dy,cos(theta),sin(theta)], for x,y the position, theta the orient angle.
            self.observation_space = Box(low=-np.array([20,20,5,5,1,1]),high=np.array([20,20,5,5,1,1])) # agent move in real world space # free for all rotation
            self.state_encoder = self.RELA5 # to very low 6D mean position data, use when got a very clear view in advance
        if representer == "GLOBAL3":
            self.observation_space = Box(low=-np.array([10,10,0.5]),high=np.array([10,10,0.5])) # agent move in real world space
            self.state_encoder = self.GLOBAL3 # to very very low 3D mean position data, use when got a very clear view in advance


        self.init_pose = init_pose

        # self.init_pose = self.HuMoR.prepare_input(init_pose, 'cpu') # initial pose, in world coo
        self.init_tensor = tensor(self.init_pose, device='cuda:0') # initialize the agent state
        print('Visualizing INITIAL!')
        # print('66INIT',joints)
        self.strength = 64 # sample size for Actor.
        self.max_simu = 600
        self.walking_length = self.max_simu
        self.vel = torch.zeros((20,2))
        self.reward = 0
        self.scatter_plots = []
        self.loss = []
        self.eva = []
        self.epi = 0
        # load in GMM model
        gmm_out_path = os.path.join('checkpoints/init_state_prior_gmm/prior_gmm.npz')
        self.gmm(gmm_out_path)
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 3]}, figsize=(20,10))
        self.fig.suptitle('Learning supervisor')
        self.forward = None # the evaluation function, to be setup when A2C model initialized

        pose = self.init_tensor.reshape(1,1,-1)
        init_dict = self.HuMoR.split_output(pose)
        del init_dict['contacts']
        self.roll_out_init(pose, init_dict)
        self.init_matrix2d = torch.tensor([[0.,-1.],[1.,0.]])
        self.control = None
        self.control_timer = 0
        self.init_info = None
        distance = np.random.uniform(0,10)
        angle = np.random.uniform(0,2*np.pi)
        self.objective = np.cos(angle)*distance, np.sin(angle)*distance
        print('OBJECTIVE', self.objective)
        self.state = self.state_encoder(init_dict) # initialize the agent state
        # print("INIT INI", self.init_pose)

        # INFO_STORAGE_FOR_TRAINING_STATISTICS
        self.rew_critic_pair = []
    
    def update(self, frame):
        # plt.cla()  # Clear the previous plot
        self.ax1.cla()
        self.ax2.cla()
        self.ax1.plot([-0.7, -0.7], [-1, 2], color='red', label='y=x line')
        self.ax1.plot([0.7, 0.7], [-1, 2], color='red', label='y=x line')
        self.ax1.scatter(self.scatter_plots[frame][:,0], self.scatter_plots[frame][:,2], c='b', marker='o')  # Create the scatter plot
        if self.eva: self.ax1.text(0,-0.5,'eva:'+str(self.eva[frame]))
        if self.loss: self.ax2.text(0,-0.5,'strai:'+str(self.loss[frame][3]))
        self.ax1.set_xlim([-1, 1])
        self.ax1.set_ylim([-1, 2])
        self.ax2.scatter(self.scatter_plots[frame][:,1], self.scatter_plots[frame][:,2], c='b', marker='o')  # Create the scatter plot
        self.ax2.set_xlim([-1, 5])
        self.ax2.set_ylim([-1, 2])

    def GLOBAL3(self, state):
        '''
        Return a 3D vector, [[position:2],[angle:1]]
        This is the simplest possible state representation for the agent.
        '''
        orient = self.global_world2local_rot.reshape((3,3))[0,1].reshape((1,-1)) # sine value of the direction
        trans = state['trans'].reshape((1, -1))[0,:2].reshape((1,-1))
        data = [trans, orient]
        cur_state = torch.cat(data, dim=-1)
        ans = cur_state.cpu().detach()
        return ans

    def GLOBAL5(self, state):
        '''
        Return a 5D vector, [[position:2],[velocity:2],[angle:1]]
        This is the simple possible state representation for the agent.
        '''
        orient = self.global_world2local_rot.reshape((3,3))[0,1].reshape((1,-1)) # sine value of the direction
        trans = state['trans'].reshape((1, -1))[0,:2].reshape((1,-1))
        vel = state['trans_vel'].reshape((1, -1))[0,:2].reshape((1,-1))
        data = [trans, vel, orient]
        cur_state = torch.cat(data, dim=-1)
        ans = cur_state.cpu().detach()
        return ans

    def RELA5(self, state, global_world2local_rot=None, B=1):
        '''
        Return a 5D vector, [[position:2],[velocity:2],[angle:1]]
        This is the simple possible state representation for the agent.
        '''
        if global_world2local_rot is None: global_world2local_rot = self.global_world2local_rot
        orient = global_world2local_rot.reshape((B,3,3))[:,0,:2].reshape((B,-1)).cpu().detach() # cosine & sine value of the direction
        trans = state['trans'].reshape((B, -1))[:,:2].reshape((B,-1)).cpu().detach()
        vel = state['trans_vel'].reshape((B, -1))[:,:2].reshape((B,-1)).cpu().detach()
        # self.vel[(self.max_simu-self.walking_length)%20] = vel
        data = [trans, vel, orient]
        # data = [trans, torch.mean(self.vel, 0).reshape(1,-1), orient]
        cur_state = torch.cat(data, dim=-1)
        ans = cur_state
        ans[:,:2] = ans[:,:2]-tensor(self.objective).view(1,-1).expand(B,-1)
        return ans

    def navigator_reward(self):
        '''
        Reward function for navigation task
        '''
        global5 = self.RELA5(self.cur_world_dict)
        if global5[0,1].item()**2+global5[0,0].item()**2 < 1:
            return 10.
        return 0.

    def straight_line_reward(self, done):
        '''
        Reward function to train agent walking straight.
        Calculate the distance from starting point. 
        '''
        global5 = self.GLOBAL3(self.cur_world_dict)
        pose = self.init_tensor.reshape(1,1,-1)
        init_dict = self.HuMoR.split_output(pose)
        del init_dict['contacts']
        init_global5 = self.GLOBAL3(init_dict)
        relative = global5-init_global5
        if done % 10 == 0: print("RELATIVE",done, relative, global5,init_global5)
        # self.min_degree = min(global5[0,-1].item(), self.min_degree)
        if abs(global5[0,-1].item()) > 0.4: # if the abs sine value greater than 40%, we say it detoured or wavy too much
            print('PUNISHED', global5[0,-1].item(), np.arcsin(global5[0,-1].item())*180/np.pi)

            return relative[0,1].item()-abs(relative[0,0].item())*2, True # cut the path with direction accuracy (sin value) too big. 
            return torch.norm(relative[0,:2]).item(), True # cut the path with direction accuracy (sin value) too big. 
        if done <= 0:
            
            return relative[0,1].item()-abs(relative[0,0].item())*2, False # when done, calculate the distance the agent moved. 
            return torch.norm(relative[0,:2]).item(), False # when done, calculate the distance the agent moved. 
        return 0, False

    def physical_reward(self):
        '''
        Compute the physics loss, for example penetration loss, joints beneath ground loss
        Also the reward when the agent get close to target
        This loss enable the agent to avoid obstacles in its way, and move towards the target
        past_coo, current_coo need to extract from world coordinates, represent the joints positions
        '''
        # for penetration, see 322 rasterized line 151
        # return whether the current config is in restricted region
        trans = self.cur_world_dict['joints']
        # all_pos = joints+trans
        # if torch.max(torch.abs(trans[:,:,0])) > 0.7:
        #     return 100
        if torch.mean(torch.abs(trans[:,:,0])) > 2:
            print('DEAD -100')
            return -100.
        if torch.mean(trans[:,:,1]) > 3:
            print('YEAH 100')
            return 100.
        return -0.1
    
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
        diss = diss.mean(1)
        # print('DISS VALUE', diss)
        return diss
    
    def VAE339toMEAN6(self, state):
        '''
        transfer the skeleton information to translation information directly
        '''
        _trans = state['trans'].reshape((1, -1))
        trans_vel = state['trans_vel'].reshape((1, -1))
        data = [_trans, trans_vel]
        cur_state = torch.cat(data, dim=-1)
        ans = cur_state.cpu().detach()
        print('VAE339toMEAN6', ans)
        return ans

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

    def batched_simu_step(self, decoded: Tensor) -> Tensor:
        '''
        Calculate the observations for each output
        '''
        B = self.strength
        temp_dict = self.HuMoR.split_output(decoded)
        root_orient_mat = temp_dict['root_orient'][:,0,:].view((B, 3, 3))
        world2aligned_rot = compute_world2aligned_mat(root_orient_mat)
        # world2aligned_trans = torch.cat([-x_pred_dict['trans'][:,0,:2], torch.zeros((B,1)).to(self.x_past)], axis=1)
        cur_world_dict = dict()
        global_world2local_trans = self.global_world2local_trans.expand(B,-1,-1)
        global_world2local_rot = self.global_world2local_rot.expand(B,-1,-1,-1)
        trans2joint = self.trans2joint.expand(B,-1,-1,-1)
        cur_world_dict = self.HuMoR.apply_world2local_trans(
            global_world2local_trans, 
            global_world2local_rot, 
            trans2joint, 
            temp_dict, cur_world_dict, invert=True)
        # global_world2local_trans = torch.cat([-cur_world_dict['trans'][:,0:1,:2], torch.zeros((B, 1, 1)).to(self.x_past)], axis=2)
        global_world2local_rot = torch.matmul(global_world2local_rot, world2aligned_rot.view((B, 1, 3, 3)))
        states = self.state_encoder(cur_world_dict,global_world2local_rot,B)
        return states

    def moving(self, decoded: Tensor) -> Tensor:
        '''
        Make the agent move forward
        '''
        states = self.batched_simu_step(decoded)
        cosine = states[:,4]
        distance = states[:,1]
        selected = 0
        if torch.max(cosine) < 0.93:
            selected = torch.argmax(cosine)
        else:
            distance = torch.masked_fill(distance, cosine < 0.93, -np.inf)
            selected = torch.argmax(distance)
        return decoded[selected].view(1,-1)

    def moving_forward(self, decoded: Tensor) -> Tensor:
        '''
        Make the agent move forward
        '''
        states = self.batched_simu_step(decoded)
        init_dir = torch.matmul(self.init_info[4:6], self.init_matrix2d.T)
        rela = states[:,0:2]-self.init_info[0:2].expand(self.strength,-1)
        distance = torch.einsum('bi,bi->b', rela, init_dir.expand(self.strength,-1)) # distance moved along the desired direction
        cosine = torch.einsum('bi,bi->b', self.init_info[4:6].expand(self.strength,-1), states[:,4:6])
        selected = 0
        if torch.max(cosine) < 0.93:
            selected = torch.argmax(cosine)
        else:
            distance = torch.masked_fill(distance, cosine < 0.93, np.inf)
            selected = torch.argmax(distance)
        return decoded[selected].view(1,-1)
    
    def moving_toward(self, decoded: Tensor, target: Tensor) -> Tensor:
        '''
        Make the agent move towards some point, when the point is in the front. 
        '''
        states = self.batched_simu_step(decoded)
        diff = target.expand(self.strength,-1)-states[:,:2] # where is the relative coordinate
        distance = torch.norm(diff, dim=-1) # distance to the target
        velo = states[:,2:4]
        angle_dir = states[:,4:6] # face direction
        angle_dir = torch.matmul(angle_dir, self.init_matrix2d.T)
        cosine = torch.einsum('bi,bi->b', angle_dir, diff)/torch.norm(diff, dim=-1)
        selected = 0
        if torch.max(cosine) < 0.85:
            cosine = torch.einsum('bi,bi->b', angle_dir, velo)/torch.norm(velo, dim=-1)
            selected = torch.argmax(cosine)
            # somehow should use turning behavior, refuse to turn in this mode, force turning will mess up the skeleton
        elif torch.max(cosine) < 0.93:
            selected = torch.argmax(cosine) # small deviation, try go back to path by adjustment. 
        else:
            distance = torch.masked_fill(distance, cosine < 0.93, np.inf)
            selected = torch.argmin(distance)
        return decoded[selected].view(1,-1)
    
    def turning(self, decoded: Tensor, ang_velo: float) -> Tensor:
        '''
        Control turning by a angular velocity, this action is highly variable
        '''
        self.bad = 0
        turning_matrix = torch.tensor([[np.cos(ang_velo), -np.sin(ang_velo)],[np.sin(ang_velo), np.cos(ang_velo)]], dtype=torch.float32)
        states = self.batched_simu_step(decoded)
        now_velo = self.state[0,2:4] # current velo
        now_dir = now_velo/torch.norm(now_velo) # current velo dir
        hope_dir = torch.matmul(now_dir, turning_matrix.T).expand(self.strength,-1) # turn a small degree
        sam_velo = states[:,2:4] # sampled velo
        sam_speed = torch.norm(sam_velo, dim=-1) # sampled speeds
        init_speed = torch.norm(self.init_info[2:4]).expand(self.strength) # inital speed
        sam_dir = sam_velo/sam_speed.unsqueeze(1) # sampled dir
        cosine = torch.einsum('bi,bi->b', sam_dir, hope_dir)
        coscos = torch.einsum('bi,bi->b', sam_dir-now_dir, hope_dir-now_dir)
        diff = torch.abs(sam_speed-init_speed)
        m1 = torch.masked_fill(cosine, coscos < 0., -np.inf)
        selected = 0
        if m1.isfinite().sum().item() < 1:
            self.bad = 2
            selected = coscos.argmax()
            return decoded[selected].view(1,-1)
        else:
            cosine = m1
            # m2 = cosine.masked_fill(diff > 0.3, -np.inf)
            # if m2.isfinite().sum().item() < 1:
            #     self.bad = 1
            #     selected = diff.argmin()
            #     return decoded[selected].view(1,-1)
        selected = cosine.argmax()
        # print('SEL:', cosine[selected], coscos[selected], diff[selected], sam_dir[selected],now_dir, hope_dir[selected])
        return decoded[selected].view(1,-1)
            

    def turning1(self, decoded: Tensor, ang_velo: float) -> Tensor:
        '''
        Control the skeleton to turn around
        '''
        # ang_velo *= self.control_timer
        self.bad = 0
        turning_matrix = torch.tensor([[np.cos(ang_velo), -np.sin(ang_velo)],[np.sin(ang_velo), np.cos(ang_velo)]], dtype=torch.float32)
        now_angle = self.state[0,4:6]
        init_speed = torch.norm(self.init_info[2:4]).expand(self.strength)
        best_angle = torch.matmul(now_angle, turning_matrix.T).expand(self.strength, -1)
        states = self.batched_simu_step(decoded)
        angle_dir = states[:,4:6] # face direction
        now_speed = torch.norm(states[:,2:4], dim=-1)
        cosine = torch.einsum('bi,bi->b', angle_dir, best_angle)
        selected = 0
        now_angle = now_angle.expand(self.strength,-1)
        coscos = torch.einsum('bi,bi->b', angle_dir-now_angle, best_angle-now_angle)
        mask1 = torch.masked_fill(cosine, coscos < 0., -np.inf)
        m1 = m2 = None
        m1 = torch.isfinite(mask1).sum().item()
        if m1 < 1: print('COSCOS', coscos)
        if not torch.max(mask1) > -np.inf:
            selected = torch.argmax(cosine)
            self.bad = 2
            print("SELECT1",selected)
        else:
            cosine = mask1 # at least turn in the correct direction
            speed_diff = torch.abs(init_speed-now_speed)
            mask2 = torch.masked_fill(cosine, torch.abs(init_speed-now_speed) > 0.3, -np.inf)
            m2 = torch.isfinite(mask2).sum().item()
            if not torch.max(mask2) > -np.inf:
                selected = torch.argmin(speed_diff)
                self.bad = 1
                print("SELECT2",selected)
            else:
                cosine = mask2
                selected = torch.argmax(cosine)
                print("SELECT3",selected)
        # raise NotImplementedError
        print('M:',m1,m2,"TIMER", self.control_timer)
        self.control_timer += 1 # effective controlled
        return decoded[selected].view(1,-1)

    def stoping(self, decoded: Tensor) -> Tensor:
        '''
        Stop the skeleton
        '''
        states = self.batched_simu_step(decoded)
        speed = torch.norm(states[:,2:4], dim=-1)
        selected = 0
        selected = torch.argmin(speed)
        return decoded[selected].view(1,-1)

    def bounded_sample(self, bound=6.1) -> Tensor:
        B = self.strength
        offsets = torch.zeros(B,48)
        i = 0
        while i < B:
            candidate = torch.randn(48)
            if torch.norm(candidate) < bound: # bound, check chi table for this, 7 is ~57% most likely, 6 is ~10%
                offsets[i] = candidate
                i += 1
        offsets = offsets.to('cuda:0')
        return offsets

    def selection(self, control):
        '''
        Actor's selectors generating funtion, for navigation task
        '''
        self.bad = 0
        if self.control != control:
            self.control = control
            self.control_timer = 0
            self.init_info = self.state[0,:]
        if control == 0: return lambda x: self.moving_forward(x)
        if control == 1: return lambda x: self.turning(x, -np.pi/150) # right turn (anti-clock)
        if control == 2: return lambda x: self.turning(x, np.pi/150) # left turn
        if control == 3: return lambda x: self.stoping(x)
        if control == 4: return lambda x: self.moving_toward(x, torch.zeros(2))
        raise NotImplementedError

    def step(self, controller):
        '''
        A step in this environment: according to previous latent state, the humor model take its default action,
        then the agent add on a trained increment of the default humor action, base on the current state given by humor. 
        '''
        # print('DO STEP')
        self.walking_length -= 1
        if self.controller is not None:
            action = np.zeros(48) # this 48D is fixed
            # action[7] = -1.
            # use = np.argmax(controller)
            if controller < len(self.controller) or True:
                # action[self.controller[controller][0]] = self.controller[controller][1] # setup which dimensions to control, only the maximum take effect
                # action = tensor(action, device='cuda:0')
                selection = self.selection(controller)
                self.roll_out_step(True, self.bounded_sample(), selection)
            else:
                # do nothing, just roaming
                self.roll_out_step(False)
                # self.back_to_init()
            # action[self.controller] = controller # setup which dimensions to control, all element take effect
        else: # default 48
            action = controller
            action = tensor(action, device='cuda:0')
            self.roll_out_step(use_mean=True, action=action)
        # print('DECODE',decoded)
        # new_state = decoded.cpu().detach().numpy()
        # tensor_new_state = tensor(new_state, device='cuda:0')
        # self.simple_vis(tensor_new_state, tensor_state)
        # self.vid_vis(tensor_state) # CAUTION: this will not show global position
        # gmm_reward = self.GMM_reward(self.x_pred_dict).item()*0
        gmm_reward = 0
        humor_loss = 0
        physical_reward = 0.#self.physical_reward()#.item()*10
        navigation_reward = self.navigator_reward()
        # straight_line_reward, done1 = self.straight_line_reward(self.walking_length)
        # if self.walking_length % 30 == 0: print('GMM:', gmm_reward, 'humor:', humor_loss, 'phy:', physical_reward, 'stra:', straight_line_reward)
        reward = gmm_reward-humor_loss+navigation_reward #+physical_reward  # write desired reward functions here.
        self.loss.append((gmm_reward,humor_loss,physical_reward, navigation_reward,reward))
        # print('REWARD SHAPE', reward.shape)
        # self.observation = new_state
        self.state = self.state_encoder(self.cur_world_dict)
        # if self.walking_length % 100 == 0: print(self.state)
        done = self.walking_length <= 0 or abs(reward)>1.1 # here setting the environment stop when get success or agent killed
        if self.forward:
            actions, values, log_prob = self.forward(tensor(self.state, device='cuda:0'))
            self.eva.append(values.item())
            if self.walking_length % 30 or done:
                print('SEE CONTROLLER:',controller, 'state:', self.state, 'actions:', actions, 'values:', values.item())
            if done:
                view = (self.state.cpu().detach().numpy().tolist()[0], reward, values.item(), self.walking_length)
                print("SAVE EPISODE END", view)
                self.rew_critic_pair.append(view)
        # reward = 0
        if done:
            self.reward = reward
        info = {'world':0}
        self.render()

        # print('STEP OVER')
        return self.state, reward, done, info

    def step_fake(self, controller):
        '''
        this is a fake step function. For test the RL environment, without HuMoR
        '''
        self.walking_length -= 1
        if controller == 0:
            self.speed += 0.01
            self.speed = min(self.speed, 0.1)
        if controller == 1:
            self.angle += np.pi/12
        if controller == 2:
            self.angle -= np.pi/12
        self.state[0,2:4] = tensor((self.speed*self.state[0,-2],self.speed*self.state[0,-1]))
        self.state[0,-2] = np.cos(self.angle)
        self.state[0,-1] = np.sin(self.angle)
        self.state[0,:2] += self.state[0,2:4]
        reward = 0.
        if self.state[0,1].item()**2+self.state[0,0].item()**2 < 1:
            reward = 10.
        done = self.walking_length <= 0 or abs(reward)>1.1 # here setting the environment stop when get success or agent killed
        return self.state, reward, done, {}


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

        This is the first lines in humor rollout function, to setup the rollout process
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

    def roll_out_step(self, use_mean=False, action=None, select_batch=None):
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

        if decoder_out.shape[0] > 1 and select_batch is not None:
            decoder_out = select_batch(decoder_out)

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

        # prepare the state space
        self.cur_world_dict = cur_world_dict

        # prepare for next frame
        self.past_in = torch.cat(in_data_list, axis=2)
        self.past_in = self.past_in.reshape((B, -1))
        self.x_pred_dict = x_pred_dict
        # print('PAST_IN', self.past_in.shape)
        return self.past_in

    def roll_out_after(self):
        '''
        Final pipeline of humor rollout, to collect generated data
        '''
        # aggregate global pred_seq
        pred_seq_out = dict()
        for k in self.pred_global_seq[0].keys():
            pred_seq_out[k] = torch.cat([self.pred_global_seq[i][k] for i in range(len(self.pred_global_seq))], axis=1)
        
        return pred_seq_out
        
    def render(self, mode='human'):
        '''
        Not implemented, due to device limitations(no screen on GPU server), tend to save training results as video files
        '''
        pass

    def reset(self, reborn=None, save=True):
        self.vel = torch.zeros((20,2))
        self.angle = 0.
        self.speed = 0.
        self.walking_length = self.max_simu
        self.epi += 1
        if self.scatter_plots and self.reward > 1 and save:
            # fig, ax = plt.subplots()
            ans = self.roll_out_after() # this zanshi over write the vid_vis in step function
            self.scatter_plots = ans['joints'].reshape((-1,22,3)).cpu().detach().numpy()
            ani = FuncAnimation(self.fig, self.update, frames=len(self.scatter_plots))
            saving = 'myProject/outs1/zanimation{}.mp4'.format(self.epi)
            ani.save(saving, writer='ffmpeg', fps=30)
            print('SAVING', saving)
            plt.cla()
            fig, (ax1,ax2) = plt.subplots(1,2, gridspec_kw={'width_ratios': [1, 3]}, figsize=(30,10))
            ploting(self.scatter_plots,ax1,ax2)
            plt.savefig('myProject/outs1/zanimation{}.png'.format(self.epi))
        self.reward = 0
        self.scatter_plots = []
        self.loss = []
        self.eva = []
        # same as in __init__
        pose = self.init_tensor.reshape(1,1,-1)
        init_dict = self.HuMoR.split_output(pose)
        del init_dict['contacts']
        self.roll_out_init(pose, init_dict)
        self.control = None
        self.control_timer = 0
        self.init_info = None
        distance = np.random.uniform(0,10)
        angle = np.random.uniform(0,2*np.pi)
        self.objective = np.cos(angle)*distance, np.sin(angle)*distance
        if reborn is not None:
            self.objective = reborn
        self.state = self.state_encoder(init_dict) # initialize the agent state
        print('OBJECTIVE RES', self.objective)
        return self.state

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
        '''
        Simply rollout humor model to skeleton view, this function cannot accept settings
        '''
        pose = self.init_tensor.reshape(1,1,-1)
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

    def default_roll_out_novis(self, B):
        '''
        Simply rollout humor model to skeleton view, this function cannot accept settings
        '''
        pose = self.init_tensor.expand(B,-1,-1).view(B,1,-1)
        init_dict = self.HuMoR.split_output(pose)
        del init_dict['contacts']
        ans = self.HuMoR.roll_out(pose, init_dict, 60)


    def default_roll_out_split(self, use_mean=False, action=None, save_id=None, simu=None):
        '''
        Rolling out humor model, can accept action as z offset. 
        '''
        pose = self.init_tensor.reshape(1,1,-1)
        init_dict = self.HuMoR.split_output(pose)
        del init_dict['contacts']
        self.roll_out_init(pose, init_dict)
        for t in range(self.max_simu if simu is None else simu):
            selection = self.selection(4) if t < 100 else self.selection(2)
            self.roll_out_step(use_mean=use_mean, action=action, select_batch=selection)
            # self.VAE339toMEAN6(self.cur_world_dict)
            self.state = self.state_encoder(self.cur_world_dict)
            view = (self.state.cpu().detach().numpy().tolist()[0], t)
            # print("SAVE EPISODE END", view)
            self.rew_critic_pair.append(view)
            # self.straight_line_reward()
        ans = self.roll_out_after()
        if save_id is not None:
            # ans = self.HuMoR.roll_out(pose, init_dict, 300)
            print('ROLLING', {k:ans[k].shape if type(ans[k])== Tensor else ans for k in ans})
            self.scatter_plots = ans['joints'].reshape((self.max_simu,22,3)).cpu().detach().numpy()
            # fig, ax = plt.subplots()
            ani = FuncAnimation(self.fig, self.update, frames=self.scatter_plots.shape[0])
            saving = 'myProject/features/roll_out{}.mp4'.format(save_id)
            ani.save(saving, writer='ffmpeg', fps=30)
            plt.cla()
            fig, (ax1,ax2) = plt.subplots(1,2, gridspec_kw={'width_ratios': [1, 3]}, figsize=(30,10))
            ploting(self.scatter_plots,ax1,ax2)
            plt.savefig('myProject/features/roll_out{}.png'.format(save_id))
            print('SAVING ROLL', saving)
        return ans

    def gmm(self, gmm_path):
        #
        # Evaluate likelihood of test data
        #

        # load in GMM result
        gmm_weights, gmm_means, gmm_covs = load_gmm_results(gmm_path)

        # build pytorch distrib
        self.gmm_distrib = build_pytorch_gmm(gmm_weights, gmm_means, gmm_covs)


def ploting(ans,ax1,ax2):
    '''
    ans: skeleton animation information, a -1x22x3 array
    '''
    step = ans.shape[0]
    ax1.cla()
    ax2.cla()
    # ax1.set_xlim([-1, 1])
    # ax1.set_ylim([-1, 2])
    ax2.set_xlim([-1, 5])
    ax2.set_ylim([-1, 2])
    ax1.set_xlim([-2, 5])
    ax1.set_ylim([-2, 5])
    x,y = ans[0,:,0],ans[0,:,1]
    ax1.scatter(x,y,s=3) # plot the initial state
    x,y = ans[0,:,1],ans[0,:,2]
    ax2.scatter(x,y,s=3) # plot the initial state
    for f in range(1,step-1):
        # rule of roll out: the user offset is taken a multiplier of the std of z.
        x,y = ans[f,:,0],ans[f,:,1]
        ax1.scatter(x,y,c='green',alpha=np.ones(22)*f/step,s=1) # plot the intermid states
        x,y = ans[f,:,1],ans[f,:,2]
        ax2.scatter(x,y,c='green',alpha=np.ones(22)*f/step,s=1) # plot the intermid states
    x,y = ans[-1,:,0],ans[-1,:,1]
    ax1.scatter(x,y,c='red',s=7) # plot the final state
    x,y = ans[-1,:,1],ans[-1,:,2]
    ax2.scatter(x,y,c='red',s=7) # plot the final state
