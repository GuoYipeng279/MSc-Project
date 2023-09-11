
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
from stable_baselines3.common.callbacks import EvalCallback

from skeleton import Skeleton, navigation_controller
init_pose = tensor([[ 0.0000e+00,  0.0000e+00,  9.4889e-01, -7.1368e-03,  1.5177e-02, # sta
          3.4314e-03, -9.9999e-01, -3.4565e-03, -2.2632e-04,  5.9605e-08,
         -6.5348e-02,  9.9786e-01, -3.4639e-03,  9.9786e-01,  6.5348e-02,
          7.1859e-02,  3.9588e-02, -7.0745e-03,  9.9737e-01, -5.5047e-02,
          4.7153e-02,  6.2416e-02,  9.8301e-01, -1.7263e-01, -3.6849e-02,
          1.7512e-01,  9.8386e-01,  9.9933e-01,  3.5936e-02, -6.6485e-03,
         -3.6530e-02,  9.8764e-01, -1.5245e-01,  1.0879e-03,  1.5259e-01,
          9.8829e-01,  9.9967e-01,  2.4189e-02,  8.5032e-03, -2.3731e-02,
          9.9845e-01, -5.0415e-02, -9.7095e-03,  5.0197e-02,  9.9869e-01,
          9.9747e-01,  6.4913e-02, -2.9150e-02, -6.2258e-02,  9.9450e-01,
          8.4241e-02,  3.4458e-02, -8.2213e-02,  9.9602e-01,  9.9729e-01,
         -6.7212e-02,  3.0009e-02,  6.4800e-02,  9.9506e-01,  7.5173e-02,
         -3.4913e-02, -7.3024e-02,  9.9672e-01,  9.9854e-01, -2.5540e-02,
         -4.7608e-02,  2.1353e-02,  9.9603e-01, -8.6456e-02,  4.9627e-02,
          8.5313e-02,  9.9512e-01,  9.8149e-01, -9.4049e-02,  1.6682e-01,
          7.0583e-02,  9.8743e-01,  1.4141e-01, -1.7802e-01, -1.2702e-01,
          9.7579e-01,  9.9285e-01, -1.0768e-01, -5.1514e-02,  1.0991e-01,
          9.9303e-01,  4.2621e-02,  4.6565e-02, -4.7978e-02,  9.9776e-01,
          9.9966e-01, -2.4926e-02, -7.9026e-03,  2.4795e-02,  9.9956e-01,
         -1.6243e-02,  8.3040e-03,  1.6041e-02,  9.9984e-01,  1.0000e+00,
          0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00,  0.0000e+00,
          0.0000e+00,  0.0000e+00,  1.0000e+00,  1.0000e+00,  0.0000e+00,
          0.0000e+00,  0.0000e+00,  1.0000e+00,  0.0000e+00,  0.0000e+00,
          0.0000e+00,  1.0000e+00,  9.9760e-01,  4.4789e-02,  5.2714e-02,
         -4.1330e-02,  9.9703e-01, -6.4972e-02, -5.5468e-02,  6.2637e-02,
          9.9649e-01,  9.1169e-01,  4.1084e-01,  5.4550e-03, -4.0843e-01,
          9.0763e-01, -9.6885e-02, -4.4755e-02,  8.6101e-02,  9.9528e-01,
          9.1036e-01, -4.0341e-01,  9.2180e-02,  4.0891e-01,  9.1116e-01,
         -5.0842e-02, -6.3480e-02,  8.3978e-02,  9.9444e-01,  9.9963e-01,
         -2.6861e-02, -4.3516e-03,  2.7036e-02,  9.9853e-01,  4.7077e-02,
          3.0807e-03, -4.7177e-02,  9.9888e-01,  4.6379e-01,  8.6277e-01,
         -2.0130e-01, -8.7908e-01,  4.7639e-01,  1.6425e-02,  1.1007e-01,
          1.6934e-01,  9.7939e-01,  4.7531e-01, -8.5143e-01,  2.2169e-01,
          8.7302e-01,  4.8769e-01,  1.2800e-03, -1.0920e-01,  1.9293e-01,
          9.7512e-01,  9.1373e-01, -2.1195e-01, -3.4666e-01,  1.6180e-01,
          9.7241e-01, -1.6807e-01,  3.7272e-01,  9.7477e-02,  9.2281e-01,
          9.3634e-01,  1.7886e-01,  3.0212e-01, -2.0098e-01,  9.7863e-01,
          4.3514e-02, -2.8788e-01, -1.0147e-01,  9.5227e-01,  9.4842e-01,
          1.7123e-01, -2.6680e-01, -1.4982e-01,  9.8377e-01,  9.8773e-02,
          2.7938e-01, -5.3706e-02,  9.5868e-01,  9.7790e-01, -1.3581e-01,
          1.5895e-01,  1.0321e-01,  9.7477e-01,  1.9791e-01, -1.8182e-01,
         -1.7713e-01,  9.6725e-01, -6.5476e-04, -2.7719e-01,  9.7751e-01,
         -6.4817e-02, -2.9085e-01,  8.8610e-01,  6.5753e-02, -2.8648e-01,
          8.7917e-01, -5.5242e-03, -3.2742e-01,  1.1111e+00, -1.2959e-01,
         -3.3006e-01,  4.9298e-01,  1.2626e-01, -3.2785e-01,  4.8493e-01,
         -1.5171e-02, -2.9772e-01,  1.2695e+00, -1.0887e-01, -3.8301e-01,
          5.5582e-02,  9.4879e-02, -3.7338e-01,  5.1982e-02, -1.2207e-02,
         -2.8808e-01,  1.3332e+00, -1.7842e-01, -2.5995e-01,  1.0331e-02,
          1.2858e-01, -2.3615e-01, -1.3632e-02,  5.0941e-03, -3.0843e-01,
          1.5693e+00, -8.6076e-02, -2.9671e-01,  1.4598e+00,  7.7965e-02,
         -3.1092e-01,  1.4556e+00, -7.2940e-03, -2.3549e-01,  1.6561e+00,
         -2.2404e-01, -3.1294e-01,  1.4574e+00,  2.0865e-01, -3.1496e-01,
          1.4501e+00, -2.3519e-01, -3.5737e-01,  1.1968e+00,  2.1620e-01,
         -3.5740e-01,  1.1827e+00, -2.9121e-01, -2.6774e-01,  9.4959e-01,
          2.7067e-01, -2.7838e-01,  9.2700e-01, -7.1368e-03,  1.5177e-02,
          3.4332e-03, -1.0856e-02,  2.2197e-02,  4.9818e-03, -1.1099e-02,
          2.1769e-02,  1.2875e-04, -2.1935e-03,  5.6052e-03,  2.8610e-05,
         -2.6585e-02, -1.1394e-02,  1.0897e-02, -1.5511e-03,  1.1510e-02,
          2.6643e-03, -2.5936e-03,  3.6356e-03,  3.7909e-04, -9.8714e-03,
         -2.1301e-03,  1.0564e-02, -1.4032e-02, -1.7806e-03,  4.9675e-03,
         -6.4958e-03,  3.3996e-03,  5.9366e-04,  1.0596e-02,  6.0318e-03,
          1.2481e-03, -1.2882e-02,  1.1297e-04,  9.5201e-03, -7.0149e-03,
          4.5360e-03,  7.2956e-04, -6.7759e-03,  3.8973e-03,  4.6492e-04,
         -6.7496e-03,  4.0976e-03,  9.0122e-04, -4.9023e-03,  1.0229e-02,
         -3.7479e-03, -8.5830e-03,  1.8756e-02,  3.7479e-03, -6.5959e-03,
          7.5026e-03,  2.1243e-03, -5.5007e-03,  7.2933e-03,  5.5647e-03,
         -9.7273e-03,  1.1092e-02,  1.4591e-03, -1.6355e-02, -8.4405e-04,
          5.0747e-03, -9.5994e-03,  8.3623e-03,  6.4731e-04]])
init_pose1 = tensor([[ 0.0000e+00,  0.0000e+00,  8.0511e-01, -1.2322e-03, -7.0556e-01, # old
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

graph = {0:[1,2,3],1:[4],2:[5],3:[6],4:[7],5:[8],6:[9],7:[10],8:[11],9:[12],10:[],
         11:[],12:[13,14,15],13:[16],14:[17],15:[],16:[18],17:[19],18:[20],19:[21],20:[],21:[]}

graph1 = {0:None,1:0,2:0,3:0,4:1,5:2,6:3,7:4,8:5,9:6,10:7,11:8,12:9,13:12,14:12,15:12,16:13,17:14,18:16,19:17,20:18,21:19}

def green_to_red(num_colors):

    # Define the RGB values for green and red
    green = np.array([0, 0, 255])
    red = np.array([255, 0, 0])

    # Generate a list of colors by linearly interpolating between green and red
    colors = []
    for i in range(num_colors):
        # Calculate the intermediate color by linear interpolation
        intermediate_color = (green * (num_colors - i - 1) + red * i) / (num_colors - 1)
        
        # Convert the RGB values to hexadecimal format
        hex_color = '#{:02X}{:02X}{:02X}'.format(int(intermediate_color[0]), int(intermediate_color[1]), int(intermediate_color[2]))
        
        colors.append(hex_color)

    return colors

if __name__ == '__main__':
    arg = sys.argv[1:]
    args = parse_args(['@./configs/test_humor_sampling.cfg'])
    env = Skeleton(args, init_pose, navigation_controller, "RELA5")
    test = 4 # choose function
    if arg: test = arg[0]
    if test == 1:
        # Check features
        print("TEST MODE")
        for i in range(1):
            print("ENGINEERING:"+str(i))
            arr = np.zeros(48)
            # arr[7] = -1.
            posneg = (i%9)/2. - 2.
            arr[i//9] = posneg
            # offset = tensor(arr, device='cuda:0')
            offset = env.bounded_sample()
            env.default_roll_out_split(True, offset, "fea_zheng"+str(i//9)+"_"+str(i%9))
        with open('features_zheng.json', 'w') as f:
            json.dump(env.rew_critic_pair, f)
        print("FEATURE ENGINEERING FINISHED!")
    elif test == 3:
        env.strength = 1
        arr = np.zeros(48)
        # arr[30] = -1.5
        offset = tensor(arr, device='cuda:0').reshape(1,48)
        env.default_roll_out_split(True, offset, "FIN30_15")
        # print("TEST MODE")
        # for i in range(48*9):
        #     print("ROLL:"+str(i))
        #     arr = np.zeros(48)
        #     # arr[7] = -1.
        #     posneg = (i%9)/2. - 2.
        #     arr[i//9] = posneg
        #     offset = tensor(arr, device='cuda:0')
        #     env.default_roll_out_split(False, offset, "fea_sta"+str(i//9)+"_"+str(i%9))
        # with open('features_sta.json', 'w') as f:
        #     json.dump(env.rew_critic_pair, f)
        # print("ROLL FINISHED!")
    elif test == 6:
        # run the trained model
        model = A2C.load('agentNAA2000001', env=env)
        env.forward = model.policy.forward
        print('Agent Loaded')
        print('START TESTING')
        square = [(0,5),(0,10),(5,10),(10,10),(10,5),(10,0),(5,0),(0,0)]
        lif = []
        for s in square:
            state = env.reset(reborn=[(0,5),(5,5),(5,0),(0,0)])
            env.max_simu = 9999
            done = False
            colors = green_to_red(4)
            while not done:
                action, _ = model.predict(state)
                state, reward, done, info = env.step(action[0])
                # path.append(state)
                joint = env.cur_world_dict['joints'].reshape(22,3).cpu().detach()
                lif.append(joint)
        
        colors = green_to_red(len(lif))
        for i, joint in enumerate(lif):
            plt.scatter(joint[:,0],joint[:,1],s=0.3,c=colors[i])
        print('PREDICT END')
        # fig = plt.figure()
        # plt.axis('equal')
        # env.ax2.scatter([s[0][0] for s in path],[s[0][1] for s in path])
        # plt.savefig('SHOWME_PRED2.png')
        plt.savefig('SHOWME_PRED3.png')
        plt.show()
    elif test == 5:
        # run the trained model
        model = A2C("MlpPolicy",env=env, verbose=1) #, learning_rate=0.01)
        env.forward = model.policy.forward
        useSaved = True
        # env.default_roll_out_split()
        if useSaved:
            model = A2C.load('agentNAA2000001', env=env)
            env.forward = model.policy.forward
            print('Agent Loaded')
        print('START TESTING')
        state = env.reset(reborn=(0,-5))
        done = False
        # path = []
        fig, (sta, m1,m2, end) = plt.subplots(1, 4, gridspec_kw={'width_ratios': [1, 1, 1, 1]}, figsize=(30,18))
        sta.axis('equal')
        m1.axis('equal')
        m2.axis('equal')
        end.axis('equal')
        colors = green_to_red(4)
        p = 0
        joint = env.cur_input_dict['joints'].reshape(22,3).cpu().detach()
        for i in range(joint.shape[0]):
            for c in graph[i]:
                sta.plot([joint[i,p],joint[c,p]],[joint[i,2],joint[c,2]], c='black')
        sta.scatter(joint[:,p],joint[:,2],c=colors[0])
        lif = []
        ii = 0
        while not done:
            print(i)
            action, _ = model.predict(state)
            state, reward, done, info = env.step(action[0])
            # path.append(state)
            joint = env.cur_world_dict['joints'].reshape(22,3).cpu().detach()
            lif.append(joint)
            if ii == 60:
                print('asdasd')
                p = 0
                joint = env.cur_input_dict['joints'].reshape(22,3).cpu().detach()
                for i in range(joint.shape[0]):
                    for c in graph[i]:
                        m1.plot([joint[i,p],joint[c,p]],[joint[i,2],joint[c,2]], c='black')
                m1.scatter(joint[:,p],joint[:,2],c=colors[1])
            if ii == 120:
                print('qweqwe')
                p = 0
                joint = env.cur_input_dict['joints'].reshape(22,3).cpu().detach()
                for i in range(joint.shape[0]):
                    for c in graph[i]:
                        m2.plot([joint[i,p],joint[c,p]],[joint[i,2],joint[c,2]], c='black')
                m2.scatter(joint[:,p],joint[:,2],c=colors[2])
            ii += 1
        print('PREDICT END')
        # fig = plt.figure()
        # plt.axis('equal')
        p = 0
        joint = env.cur_input_dict['joints'].reshape(22,3).cpu().detach()
        for i in range(joint.shape[0]):
            for c in graph[i]:
                end.plot([joint[i,p],joint[c,p]],[joint[i,2],joint[c,2]], c='black')
        end.scatter(joint[:,p],joint[:,2],c=colors[3])
        # env.ax2.scatter([s[0][0] for s in path],[s[0][1] for s in path])
        # plt.savefig('SHOWME_PRED2.png')
        plt.savefig('SHOWME_PRED3.png')
        plt.show()
    elif test == 4:
        # run the trained model
        model = A2C("MlpPolicy",env=env, verbose=1) #, learning_rate=0.01)
        env.forward = model.policy.forward
        useSaved = True
        # env.default_roll_out_split()
        if useSaved:
            # model = A2C.load('agentNAA2000001', env=env)
            model = A2C.load('agentNAAR300000', env=env)
            env.forward = model.policy.forward
            print('Agent Loaded')
        print('START TESTING')
        env.max_simu = 9000
        env.run_some = 0
        state = env.reset(reborn=([(0,10),(10,10)]))#,(5,0),(0,0)]))
        done = False
        # path = []
        fig, (sta, path, end) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 2, 1]}, figsize=(30,18))
        sta.axis('equal')
        path.axis('equal')
        end.axis('equal')
        p = 0
        joint = env.cur_input_dict['joints'].reshape(22,3).cpu().detach()
        for i in range(joint.shape[0]):
            for c in graph[i]:
                sta.plot([joint[i,p],joint[c,p]],[joint[i,2],joint[c,2]], c='black')
        sta.scatter(joint[:,p],joint[:,2],c='#0000FF')
        lif = []
        while not done:
            action, _ = model.predict(state)
            state, reward, done, info = env.step(action[0])
            # path.append(state)
            joint = env.cur_world_dict['joints'].reshape(22,3).cpu().detach().numpy()
            lif.append(joint)
        colors = green_to_red(len(lif))
        for i, joint in enumerate(lif):
            path.scatter(joint[:,0],joint[:,1],s=0.3,c=colors[i])
        print('PREDICT END')
        print(len(lif))
        with open('run_his31_tur.npy', 'wb') as f:
            np.save(f, np.array(lif))
        # fig = plt.figure()
        # plt.axis('equal')
        p = 0
        joint = env.cur_input_dict['joints'].reshape(22,3).cpu().detach()
        for i in range(joint.shape[0]):
            for c in graph[i]:
                end.plot([joint[i,p],joint[c,p]],[joint[i,2],joint[c,2]], c='black')
        end.scatter(joint[:,p],joint[:,2],c='#FF0000')
        # env.ax2.scatter([s[0][0] for s in path],[s[0][1] for s in path])
        # plt.savefig('SHOWME_PRED2.png')
        plt.savefig('SHOWME_PRED3.png')
        plt.show()
    elif test == 2:
        # Visualize the critic network, for 5D simple data
        model = A2C("MlpPolicy",env=env, verbose=1) #, learning_rate=0.01)
        env.forward = model.policy.forward
        useSaved = True
        # env.default_roll_out_split()
        if useSaved:
            try:
                model = A2C.load('agentNAA300000', env=env)
                env.forward = model.policy.forward
                print('Agent Loaded')
            except:
                pass
        inpu = torch.cat([tensor([[x,y,0,0,1,0]], device='cuda:0') for x in np.arange(-10,10,0.1) for y in np.arange(-10,10,0.1)])
        print(inpu.shape)
        actions, value, _ = env.forward(inpu)
        print(value.shape)
        actions = actions.reshape((200,-1)).cpu().detach()
        plt.imshow(actions.T)
        plt.colorbar()
        plt.savefig("SHOWME_ACTION.png")
        value = value.reshape((200,-1)).cpu().detach()
        plt.imshow(value.T)
        plt.colorbar()
        plt.savefig("SHOWME_VALUE.png")
        # plt.show()
    else:
        # Training
        model = A2C("MlpPolicy",env=env, verbose=1) #, learning_rate=0.01)
        env.forward = model.policy.forward
        useSaved = False
        # env.default_roll_out_split()
        if useSaved:
            try:
                model = A2C.load('agentNAA1500000', env=env)
                env.forward = model.policy.forward
                print('Agent Loaded')
            except:
                pass
        print('START LEARNING')
        total_step = 300000
        # eval_callback = EvalCallback(model, best_model_save_path="./myProject/logs/",
        #                      log_path="./myProject/logs/", eval_freq=500,
        #                      deterministic=True, render=False)
        model.learn(total_step)#, callback=eval_callback)
        model.save('agentNAAR'+str(total_step))
        value_pair = np.array(env.rew_critic_pair)
        # plt.cla()
        # plt.scatter(value_pair[:,0], value_pair[:,1], alpha=np.array(range(len(value_pair)))/len(value_pair))
        # plt.show()
        with open('rew_critic_pair_sta_rela.json', 'w') as f:
            json.dump(env.rew_critic_pair, f)
        