import matplotlib.pyplot as plt
import numpy as np


def show_image(image):
    shape = image.shape
    
    plt.figure(figsize=(2, 2))
    
    if shape[-1] == 1:
        plt.imshow(np.reshape(image, (shape[0], shape[1])), cmap='gray', interpolation='nearest')
    else:
        plt.imshow(image, interpolation='nearest')
        
    plt.axis('off')
    plt.show()

def show_filter(w_orig):
    w = w_orig.T
    the_shape = w_orig.shape
    f_size = the_shape[0]
    in_dim = the_shape[2]
    out_dim = the_shape[3]
    canvas = np.zeros(((f_size + 1) * out_dim, (f_size + 1) * in_dim))
    for i in range(out_dim):
        for j in range(in_dim):
            canvas[i * (f_size + 1):i * (f_size + 1) + f_size, j * (f_size + 1):j * (f_size + 1) + f_size] = w[i, j]
    plt.figure(figsize=(16, 16))
    canvas_fixed = np.zeros((canvas.shape[0] + 1, canvas.shape[1] + 1))
    canvas_fixed[1:, 1:] = canvas
    plt.imshow(canvas_fixed.T, cmap='Greys', interpolation='nearest')
    plt.axis('off')
    plt.show()