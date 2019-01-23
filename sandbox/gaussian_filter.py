import random as rand
import libs.datasets as ds
import libs.utilities as utl
# import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


dim = 27
mnist_sample = 5514
mnist_category = 9


category = rand.randint(0, 9) if mnist_category is None else mnist_category
data_set = ds.load_images_mnist(category=category, dim=dim)
index = rand.randint(0, len(data_set)) if mnist_sample is None else mnist_sample
print('using mnist sample', index)
sample_origin = data_set[index, :, :]
sample_gauss = gaussian_filter(sample_origin, sigma=1)

# plt.imshow(sample_origin)
# plt.show()
# plt.imshow(sample_gauss)
# plt.show()

for algorithm in ['dot', 'mse', 'ssim', 'gauss']:
    similarity = utl.calc_similarity(sample_origin, sample_origin, algorithm=algorithm)
    print('similarity for algo {} is {}'.format(algorithm, similarity))
    similarity = utl.calc_similarity(sample_gauss, sample_gauss, algorithm=algorithm)
    print('similarity for algo {} is {}'.format(algorithm, similarity))
    similarity = utl.calc_similarity(sample_gauss, sample_origin, algorithm=algorithm)
    print('similarity for algo {} is {}'.format(algorithm, similarity))
