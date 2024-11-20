import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def split_image(image_path, rows, cols):
    image = Image.open(image_path)
    image = image.convert('1') 
    image_array = np.array(image,int)
    img_height, img_width = image_array.shape
    part_height = img_height // rows
    part_width = img_width // cols
    parts = []
    for i in range(rows):
        for j in range(cols):
            part = image_array[i * part_height:(i + 1) * part_height, j * part_width:(j + 1) * part_width]
            part = 2 * part - 1
            parts.append(part)
    return parts

def display(image):
    plt.imshow(image, cmap='gray')
    
def display_grid(images, rows, cols, cm='gray'):
    fig, axes = plt.subplots(rows, cols, figsize=(5,5))
    axes = axes.flatten()
    for image, ax in zip(images, axes):
        image_array = np.array(image)
        ax.imshow(image_array, cmap=cm)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_patterns(patterns, title):
    fig, axes = plt.subplots(1, len(patterns), figsize=(2,1))
    for ax, pattern in zip(axes, patterns):
        ax.imshow(pattern, cmap='winter')
        ax.axis('off')
    plt.suptitle(title)
    plt.show()


def add_noise(pattern, noise_level=0.1):
    noisy_pattern = pattern.copy()
    num_noisy_bits = int(noise_level * pattern.size)
    flip_indices = np.random.choice(pattern.size, num_noisy_bits, replace=False)
    noisy_pattern.flat[flip_indices] *= -1
    return noisy_pattern

simple = [
        np.array([+1,-1,+1,-1,+1,-1,+1,-1,+1]).reshape(3,3),
        np.array([-1,+1,-1,+1,-1,+1,-1,+1,-1]).reshape(3,3),
        np.array([+1,+1,-1,-1,-1,+1,+1,+1,-1]).reshape(3,3),
        #np.array([+1,+1,+1,+1,-1,+1,+1,+1,+1]).reshape(3,3),    
        #np.array([+1,+1,+1,-1,-1,-1,+1,+1,+1]).reshape(3,3),     
        #np.array([-1,+1,-1,-1,+1,-1,-1,+1,-1]).reshape(3,3),    
]

numbers = [
        np.array([[-1,-1,-1,+1,-1],[-1,-1,+1,+1,-1],[-1,+1,-1,+1,-1],[-1,-1,-1,+1,-1],[-1,-1,-1,+1,-1]]), 
        np.array([[-1,+1,+1,-1,-1],[+1,-1,-1,+1,-1],[-1,-1,+1,-1,-1],[-1,+1,-1,-1,-1],[+1,+1,+1,+1,-1]]),
        np.array([[-1,+1,+1,-1,-1],[+1,-1,-1,+1,-1],[-1,-1,+1,-1,-1],[+1,-1,-1,+1,-1],[-1,+1,+1,-1,-1]]),
        np.array([[-1,-1,-1,+1,-1],[-1,-1,+1,+1,-1],[-1,+1,-1,+1,-1],[+1,+1,+1,+1,+1],[-1,-1,-1,+1,-1]]),
        np.array([[+1,+1,+1,+1,-1],[+1,-1,-1,-1,-1],[+1,+1,+1,-1,-1],[-1,-1,-1,+1,-1],[+1,+1,+1,-1,-1]])    
]


if __name__ == '__main__':
    print ("Einige Beispiel-Muster zur Verwendung in Hopfield-Netzwerken")
    