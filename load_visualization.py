import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from skimage.transform import resize

def scatter(data, label, dir, file_name, mus=None, mark_size=2):
    if label.ndim == 2:
        label = np.argmax(label, axis=1)

    df = pd.DataFrame(data={'x':data[:,0], 'y':data[:,1], 'class':label})
    sns_plot = sns.lmplot('x', 'y', data=df, hue='class', fit_reg=False, scatter_kws={'s':mark_size})
    sns_plot.savefig(os.path.join(dir, file_name))
    if mus is not None:
        df_mus = pd.DataFrame(data={'x':mus[:,0], 'y':mus[:,1], 'class':np.asarray(xrange(mus.shape[0])).astype(np.int32)})
        sns_plot_mus = sns.lmplot('x', 'y', data=df_mus, hue='class', fit_reg=False, scatter_kws={'s':mark_size*20})
        sns_plot_mus.savefig(os.path.join(dir, 'mus_'+file_name))


def grad_cam(x_test, y_test, target_num, d_logits1, target_pred, feed_dict, sess, top_conv, num_classes, save_img):
    prob = sess.run(target_pred, feed_dict=feed_dict)[0] # prob is batch_size x num_classes
    preds = (np.argsort(prob)[::-1])[0:5]
    for p in preds:
        print('class number : {}, prob : {}'.format(p,prob[p]))
    # Target class
    predicted_class = preds[0]
    #predicted_class = y_test[target_num]
    
    # Start visualizing activation map 
    print("Setting gradients to 1 for target class and rest to 0")
    one_hot = tf.sparse_to_dense(predicted_class, [num_classes], 1.0)
    signal = tf.multiply(d_logits1, one_hot)
    loss = tf.reduce_mean(signal)
    
    grads = tf.gradients(loss, top_conv)[0]
    # Normalizing the gradients
    norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))
    output, grads_val = sess.run([top_conv, norm_grads], feed_dict=feed_dict)
    output = output[0]           
    grads_val = grads_val[0]	 
                    
    weights = np.mean(grads_val, axis = (0, 1)) 			
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)	

    # Taking a weighted average
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    # Passing through ReLU
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)
    cam = resize(cam, (32,32))
                
    # Converting grayscale to 3-D
    gray_cam3 = np.expand_dims(cam, axis=2)
    gray_cam3 = np.tile(gray_cam3,[1,1,3])
    
    # RGB heatmap
    cmap = plt.get_cmap('jet')
    rgba_img = cmap(cam)
    cam3 = np.delete(rgba_img, 3, 2)

    img = x_test
    img /= img.max()
                    
    # Superimposing the visualization with the image.
    new_img = img+3*gray_cam3
    new_img /= new_img.max()
                
    # Display and save
    fig = plt.figure(figsize=(8,8))
    plt.imshow(img)
    plt.title('Original Image')
    name = save_img + 'original_img.png'
    fig.savefig(name)

    fig = plt.figure(figsize=(8,8))
    plt.imshow(cam3)
    plt.title('Activation Map')
    name = save_img + 'activation_map.png'
    fig.savefig(name)
    
     
    plt.figure(figsize=(8,8))
    plt.imshow(img)
    plt.imshow(cam3, alpha=0.4)
    plt.title('Overlay Image')
    name = save_img + 'overlay_img.png'
    plt.savefig(name)
    
