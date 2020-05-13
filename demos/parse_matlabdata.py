from scipy.io import loadmat
import PcmPy as pcm
import pickle
X = loadmat('data_recipe_finger7T.mat')
Y = list()
model = list()
for i in range(7):
    Data= X['Y'][0,i]
    obs_des = {'cond_vec': X['condVec'][0,i].reshape(-1),
               'part_vec': X['partVec'][0,i].reshape(-1)}
    Y.append(pcm.Dataset(Data,obs_descriptors = obs_des))

model.append(pcm.ModelFixed('Muscle',X['Model'][0,0][1]))
model.append(pcm.ModelFixed('Naturalstats',X['Model'][0,1][1]))

pickle.dump([Y,model],open('data_finger7T.p','wb'))