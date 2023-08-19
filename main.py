
import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

import json
from humor.test.test_humor import parse_args
import numpy as np
from matplotlib import pyplot as plt
from torch import tensor
import torch
from stable_baselines3 import A2C
from skeleton import Skeleton, moving_forward_controller, left_turn_controller, right_turn_controller
init_pose = tensor([[ 0.0000e+00,  0.0000e+00,  8.0511e-01, -1.2322e-03, -7.0556e-01,
          8.5901e-02, -9.9506e-01, -9.9287e-02,  7.6269e-04, -3.7206e-04,
          1.1409e-02,  9.9993e-01, -9.9289e-02,  9.9499e-01, -1.1390e-02,
          2.9144e-01,  4.7586e-01, -5.8740e-01,  9.9901e-01, -3.8074e-02,
          2.2805e-02,  3.9895e-02,  9.9553e-01, -8.5599e-02, -1.9443e-02,
          8.6424e-02,  9.9607e-01,  9.9853e-01, -2.6929e-02,  4.7129e-02,
         -6.2903e-04,  8.6245e-01,  5.0613e-01, -5.4276e-02, -5.0542e-01,
          8.6117e-01,  9.9343e-01, -6.9779e-02, -9.0748e-02,  3.7885e-02,
          9.4848e-01, -3.1457e-01,  1.0802e-01,  3.0907e-01,  9.4489e-01,
          9.9278e-01, -1.7642e-02, -1.1868e-01, -2.7652e-02,  9.2887e-01,
         -3.6937e-01,  1.1676e-01,  3.6999e-01,  9.2167e-01,  9.8578e-01,
         -8.0672e-02, -1.4738e-01,  2.5950e-02,  9.3977e-01, -3.4083e-01,
          1.6600e-01,  3.3216e-01,  9.2850e-01,  9.9911e-01, -3.8829e-02,
         -1.6275e-02,  4.0699e-02,  9.8971e-01,  1.3720e-01,  1.0780e-02,
         -1.3774e-01,  9.9041e-01,  9.7321e-01, -1.9458e-01,  1.2245e-01,
          1.0677e-01,  8.5422e-01,  5.0884e-01, -2.0361e-01, -4.8213e-01,
          8.5211e-01,  9.9203e-01, -8.8782e-02, -8.9442e-02,  9.7384e-02,
          9.9052e-01,  9.6910e-02,  7.9989e-02, -1.0485e-01,  9.9127e-01,
          9.9856e-01, -3.4446e-02, -4.1027e-02,  3.3999e-02,  9.9936e-01,
         -1.1538e-02,  4.1398e-02,  1.0127e-02,  9.9909e-01,  9.9563e-01,
          9.1619e-02, -1.8330e-02, -9.2368e-02,  9.9469e-01, -4.5386e-02,
          1.4074e-02,  4.6880e-02,  9.9880e-01,  9.9996e-01, -7.9575e-03,
          3.1180e-03,  7.9595e-03,  9.9997e-01, -5.8146e-04, -3.1133e-03,
          6.0636e-04,  9.9999e-01,  9.9883e-01, -4.7046e-02, -1.1409e-02,
          4.8328e-02,  9.8276e-01,  1.7845e-01,  2.8171e-03, -1.7879e-01,
          9.8388e-01,  9.0024e-01,  4.3479e-01, -2.2876e-02, -4.3535e-01,
          8.9823e-01, -6.0333e-02, -5.6842e-03,  6.4274e-02,  9.9792e-01,
          8.8796e-01, -4.5840e-01,  3.7389e-02,  4.5983e-01,  8.8646e-01,
         -5.2376e-02, -9.1348e-03,  6.3700e-02,  9.9793e-01,  9.9239e-01,
          2.5738e-02, -1.2042e-01, -3.7416e-02,  9.9470e-01, -9.5754e-02,
          1.1731e-01,  9.9531e-02,  9.8810e-01,  6.2928e-01,  7.6551e-01,
         -1.3412e-01, -7.6734e-01,  6.3937e-01,  4.8976e-02,  1.2324e-01,
          7.2097e-02,  9.8975e-01,  6.7375e-01, -7.3795e-01,  3.8603e-02,
          7.3581e-01,  6.6513e-01, -1.2725e-01,  6.8232e-02,  1.1414e-01,
          9.9112e-01,  7.9180e-01, -2.5291e-01, -5.5597e-01,  2.2950e-01,
          9.6674e-01, -1.1291e-01,  5.6603e-01, -3.8194e-02,  8.2350e-01,
          8.2782e-01,  3.2728e-01,  4.5563e-01, -3.3243e-01,  9.4041e-01,
         -7.1502e-02, -4.5188e-01, -9.2277e-02,  8.8729e-01,  9.8036e-01,
         -1.4647e-01, -1.3204e-01,  1.3982e-01,  9.8845e-01, -5.8385e-02,
          1.3907e-01,  3.8777e-02,  9.8952e-01,  9.9407e-01,  3.2146e-02,
          1.0390e-01, -3.0610e-02,  9.9940e-01, -1.6347e-02, -1.0436e-01,
          1.3070e-02,  9.9445e-01,  1.6245e-03, -2.8482e-01,  8.8401e-01,
         -5.5182e-02, -2.1845e-01,  7.4918e-01,  7.4455e-02, -2.4010e-01,
          6.8994e-01, -1.3759e-02, -2.3928e-01,  8.9972e-01, -9.6250e-02,
         -2.7485e-01,  3.3965e-01,  1.6118e-01, -7.8171e-02,  4.6492e-01,
         -1.3543e-02, -1.8616e-01,  1.1338e+00, -1.8051e-02, -5.2053e-01,
          6.2422e-02,  1.1092e-01, -2.2801e-02,  7.2010e-02,  1.0378e-02,
         -1.4428e-01,  1.1281e+00, -8.9189e-02, -3.7981e-01,  4.0535e-02,
          1.5433e-01,  1.1472e-01,  2.3981e-02, -2.7197e-03, -2.1105e-01,
          1.3212e+00, -7.9487e-02, -1.4423e-01,  1.2728e+00,  3.4606e-02,
         -2.2412e-01,  1.1992e+00,  1.4428e-03, -1.1885e-01,  1.4380e+00,
         -1.7616e-01, -1.4660e-01,  1.2293e+00,  1.8533e-01, -2.2162e-01,
          1.1957e+00, -2.2553e-01, -2.3912e-01,  9.7636e-01,  1.9568e-01,
         -3.3831e-01,  1.0197e+00, -3.4166e-01, -4.3088e-02,  8.2902e-01,
          3.4399e-01, -2.6479e-01,  7.7882e-01,  1.0705e-02, -7.4640e-01,
          9.1069e-02, -1.3117e-02, -6.5188e-01,  1.0578e-01, -6.1951e-02,
         -6.8535e-01,  1.6696e-02,  6.1482e-02, -6.9239e-01,  8.9408e-02,
         -5.0143e-02, -4.3600e-01,  6.6703e-02, -1.8777e-02, -4.9920e-01,
          1.6411e-01,  5.2510e-02, -6.8932e-01, -3.4454e-02,  3.9516e-02,
         -6.8476e-02, -1.7581e-01,  8.2091e-02,  1.3037e-01,  1.6101e-01,
         -3.5180e-02, -6.6699e-01,  4.6575e-02,  1.2462e-02, -1.1470e-02,
         -7.7386e-02,  7.3329e-02,  1.2802e-01,  2.9324e-01, -9.5655e-02,
         -4.8894e-01,  3.6477e-02, -7.5001e-02, -5.6492e-01,  1.1714e-02,
         -3.2485e-02, -5.6993e-01,  4.5901e-02, -1.6097e-01, -4.3028e-01,
          2.1620e-03, -5.7818e-02, -6.4297e-01, -6.2898e-02, -5.7622e-02,
         -4.9302e-01,  1.4586e-01, -6.2222e-02, -6.4255e-01, -8.6503e-02,
         -1.8598e-02, -6.4377e-01,  1.6776e-01,  7.7387e-03, -5.7664e-01,
          1.2950e-02, -3.0246e-02, -7.0329e-01,  2.2927e-01]])


if __name__ == '__main__':
    print(sys.argv[1:])
    args = parse_args(['@./configs/test_humor_sampling.cfg'])
    env = Skeleton(args, init_pose, moving_forward_controller, "GLOBAL3")
    test = 0
    if test == 1:
        print("TEST MODE")
        for i in range(48*9):
            print("ENGINEERING:"+str(i))
            arr = np.zeros(48)
            # arr[7] = -1.
            posneg = (i%9)/2. - 2.
            arr[i//9] = posneg
            offset = tensor(arr, device='cuda:0')
            env.default_roll_out_split(True, offset, "fea"+str(i//9)+"_"+str(i%9))
        with open('features.json', 'w') as f:
            json.dump(env.rew_critic_pair, f)
        print("FEATURE ENGINEERING FINISHED!")
    elif test == 2:
        # Visualize the critic network, for 5D simple data
        model = A2C("MlpPolicy",env=env, verbose=1) #, learning_rate=0.01)
        env.forward = model.policy.forward
        useSaved = True
        # env.default_roll_out_split()
        if useSaved:
            try:
                model = A2C.load('agentWS', env=env)
                env.forward = model.policy.forward
                print('Agent Loaded')
            except:
                pass
        inpu = torch.cat([tensor([[0,y,0,0,x]], device='cuda:0') for x in np.arange(-0.4,0.4,0.01) for y in np.arange(-1,10,0.1)])
        print(inpu.shape)
        actions, value, _ = env.forward(inpu)
        print(value.shape)
        value = value.reshape((80,-1)).cpu().detach()
        plt.contour(value.T)
        plt.savefig("SHOWME.png")
        # plt.show()
    else:
        model = A2C("MlpPolicy",env=env, verbose=1) #, learning_rate=0.01)
        env.forward = model.policy.forward
        useSaved = False
        # env.default_roll_out_split()
        if useSaved:
            try:
                model = A2C.load('agentWS', env=env)
                env.forward = model.policy.forward
                print('Agent Loaded')
            except:
                pass
        print('START LEARNING')
        total_step = 100000
        model.learn(total_step)
        model.save('agent48WS'+str(total_step))
        value_pair = np.array(env.rew_critic_pair)
        # plt.cla()
        # plt.scatter(value_pair[:,0], value_pair[:,1], alpha=np.array(range(len(value_pair)))/len(value_pair))
        # plt.show()
        with open('zrew_critic_pair5.json', 'w') as f:
            json.dump(env.rew_critic_pair, f)
        