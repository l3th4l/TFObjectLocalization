from trainops import *
import os

#hyperparams 
batch_size = 100 
epochs = 200

#Data paths
path = './Dataset/training_data/'
path_objs = path + 'processed_imgs/'
path_false = path + 'processed_false/'

#filelist, labels
list_false = os.listdir(path_false) #false filelist 
cat_false = np.zeros_like(list_false, np.int) #false category labels

dat_false = pd.DataFrame(list(zip(list_false, cat_false)), index = None, columns = ['filename', 'category'])
dat_true = pd.read_csv(path + 'labels.csv') #true


# train / test / val split 
false_dat_len = len(dat_false) #Total false data points
true_dat_len = len(dat_true) #Total true data points
tr_split = int(0.8 * true_dat_len) # Data points in training split 
val_split = int(0.1 * true_dat_len) # Data points in validation split 
tst_split = int(0.1 * true_dat_len) # Data points in test split 

#Global variables initializer
init = tf.global_variables_initializer()

#Session 
sess = tf.InteractiveSession()

#Saver 
saver = tf.train.Saver()

#Training loop 
sess.run(init)

#saver.restore(sess, './models/v1/weights_%i_%i.ckpt' % (364, 50))
'''
#Load model
mdl_path = './models/weightV1.ckpt'
saver.restore(sess, mdl_path)
'''
for epoch in range(epochs):
    for i in range(tr_split // batch_size):
        #True labels 
        d_true, x_true, b_true, c_true = datFetch(dat_true, path_objs, batch_size, i, aug_chance = 0.3)
        #False labels
        ind = np.random.randint(false_dat_len // batch_size - 1) 
        d_false, x_false, _, _ = datFetch(dat_false, path_false, batch_size, ind, aug_chance = 0.3, load_box = False)
        d_false = d_false * 0
    
        #Optimize true labels
        _, _, _, dl_t, cl_t, bl_t = sess.run([d_opt, c_opt, b_opt, d_loss, c_loss, b_loss], feed_dict = {x : x_true, c : c_true, b : b_true, d : d_true})
        #Optimize false labels
        _, dl_f = sess.run([d_opt, d_loss], feed_dict = {x : x_false, d : d_false})

        if i % 50 == 0:
            print('epoch_%i iter_%i det_t : %f cat_t : %f bound_t : %f det_f : %f ' % (epoch, i, np.mean(dl_t), np.mean(cl_t), np.mean(bl_t), np.mean(dl_f)))
            saver.save(sess, './models/v1/weights_%i_%i.ckpt' % (epoch, i))

saver.save(sess, './models/weightV1.ckpt')