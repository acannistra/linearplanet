import numpy as np
from sklearn.cluster import MiniBatchKMeans
from skimage.exposure import equalize_hist
from multiprocessing import Pool
from itertools import chain
import tensorflow as tf
import tqdm


class FeatureMap(object):
    '''
    FeatureMap is a class which
        1) Extracts random patches from training images
        2) Performs pre-processing (normalization, whitenin) on these patches
        3) Uses K-means clustering to learn a R^d --> R^k feature mapping
    '''

    def __init__(self, dataset, K, num_patches, patch_size,
                 normalize=True, whiten=True):
        '''
            dataset: Tensorflow dataset
            num_patches: number of random patches to extract from each image
            patch_size: number of pixels on one side of square patch
            normalize: whether to normalize each patch, boolean
            whiten: whether to center+sphere each patch, boolean
        '''
        super(FeatureMap, self).__init__()
        self.K = K
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.normalize = normalize
        self.whiten = whiten
        self.dataset = dataset


    def _getRandomPatches(self, image):
        patches = []
        d, x, y = image.shape
        max_x = x - self.patch_size
        max_y = y - self.patch_size

        for i in range(self.num_patches):
            _x = np.random.randint(max_x)
            _y = np.random.randint(max_y)
            patch = image[:, _x:_x + self.patch_size, _y:_y + self.patch_size]
            patches.append(patch)

        return(patches)

    def _normalize(self, image):
        '''
        Use skimage's equalize-hist function to remove contrast. 
        '''
        return(equalize_hist(image))

    def _whiten(self, image):
        raise NotImplementedError("Whitening is unimplemented.")

    def fit(self, batchsize=100, processes=10):
        '''
            Train k-means patch feature mapping.
        '''
        #processPool = Pool(processes=processes)

        def _processBatch(batch, estimator):
            # remove extra dim
            batch = list(map(np.squeeze, batch))
            # get all patches
            patches = list(chain.from_iterable(map(self._getRandomPatches, batch)))

            if self.normalize:
                patches = list(map(self._normalize, patches))
            if self.whiten:
                patches = list(map(self._whiten, patches))
            flattened = np.vstack(map(lambda x: x.flatten(), patches))
            estimator.partial_fit(flattened)

        self.estimator = MiniBatchKMeans(n_clusters=self.K,
                                         batch_size=batchsize,
                                         compute_labels=False)

        batchdata = self.dataset.batch(batchsize)
        dataIterator = tf.data.Iterator.from_structure(batchdata.output_types,
                                                       batchdata.output_shapes)

        initializer = dataIterator.make_initializer(batchdata)
        with tf.Session() as sess:
            sess.run(initializer)
            pbar = tqdm.tqdm()
            while True:
                try:
                    images, labels = sess.run(dataIterator.get_next())
                    images = np.vsplit(images, batchsize)
                    _processBatch(images, self.estimator)
                    pbar.update(batchsize)
                except tf.errors.OutOfRangeError:
                    break
            pbar.close()
