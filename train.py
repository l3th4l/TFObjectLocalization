from trainops import *
import os

#hyperparams 
batch_size = 100 
epochs = 50

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
saver = tf.train.Saver(tf.trainable_variables())

#Training loop 
sess.run(init)

for epoch in range(epochs):
    for i in range(tr_split // batch_size):
        #True labels 
        x_true, b_true, s_true = datFetch(dat_true, path_objs, batch_size, i, aug_chance = 0.3)
        #False labels
        ind = np.random.randint(false_dat_len // batch_size - 1) 
        x_false, _, s_false = datFetch(dat_false, path_false, batch_size, ind, aug_chance = 0.3, load_box = False)
    
        #Optimize true labels
        _, _, sl_t, bl_t = sess.run([s_opt, b_opt, s_loss, b_loss], feed_dict = {x : x_true, s : s_true, b : b_true})
        #Optimize false labels
        _, sl_f = sess.run([s_opt, s_loss], feed_dict = {x : x_false, s : s_false})

        if i % 50 == 0:
            print('epoch_%i iter_%i cat_t : %f bound_t : %f cat_f : %f ' % (epoch, i, np.mean(sl_t), np.mean(bl_t), np.mean(sl_f)))
            saver.save(sess, './models/v1/weights_%i_%i.ckpt' % (epoch, i))

saver.save(sess, './models/weightV1.ckpt')