from trainops import *

#hyperparams 
batch_size = 100 
epochs = 20 

#Data paths
path = './Dataset/training_data/'
path_objs = path + 'processed_imgs/'
path_false = path + 'processed_false/'

#filelist, labels
dat = pd.read_csv(path + 'labels.csv')

# train / test / val split 
dat_len = len(dat) #Total data points
tr_split = int(0.8 * dat_len) # Data points in training split 
val_split = int(0.1 * dat_len) # Data points in validation split 
tst_split = int(0.1 * dat_len) # Data points in test split 

#Training loop 
for i in range(tr_split // batch_size):
    