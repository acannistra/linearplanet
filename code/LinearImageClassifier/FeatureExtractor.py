import numpy as np
from warnings import warn
import tensorflow as tf
import tqdm


class FeatureExtractor(object):
    """
    FeatureExtractor is a class which
        1) Divides input images into equally spaced sub-patches
        2) Extract features from patches using a FeatureMap object
        3) Pools features together over reagions of input to reduce
           number of feature values

    We will use patch_size from the FeatureMap passed into the
    constructor.
    """

    def __init__(self, featuremap, stride=2, pools=4):
        super(FeatureExtractor, self).__init__()
        self.featuremap = featuremap
        self.stride = stride
        self.pools = pools
        self.patch_size = self.featuremap.patch_size

    def _generatePatches(self, image):
        d, x, y = image.shape
        patches = []
        for i in range(0, x, self.patch_size + self.stride - 1):
            # row i
            for j in range(0, x, self.patch_size + self.stride - 1):
                # col j
                patch = image[:, i:i + self.patch_size, j:j + self.patch_size]
                if(patch.shape[1] != self.patch_size or
                   patch.shape[2] != self.patch_size):
                    # too close to the end of the image, sub-patch created.
                    warn("Throwing out patch of shape %s; check patch size"
                         "/stride mismatch" % str(patch.shape))
                    continue
                patches.append(patch)

        return(np.vstack(list(map(lambda x: x.flatten(), patches))))

    def extract(self, dataset=None):
        '''
        dataset: a Tensorflow dataset. if None, we use the dataset
        within the given featuremap.

        Using the trained feature mapping in self.featuremap,
        we extract patches from each image and transform each patch
        into a K-dimensional feature vector.
        We then pool these K-dimensional feature vectors to create
        a self.pools * K-diminsional feature vector for each
        input image.
        '''
        if dataset is None:
            dataset = self.featuremap.dataset

        dataIterator = tf.data.Iterator.from_structure(dataset.output_types,
                                                       dataset.output_shapes)

        initializer = dataIterator.make_initializer(dataset)
        with tf.Session() as sess:
            sess.run(initializer)
            pbar = tqdm.tqdm()
            while True:
                try:
                    image, label = sess.run(dataIterator.get_next())
                    patches = self._generatePatches(image)
                    self.featuremap.predict(patches)
                    pbar.update(1)
                except tf.errors.OutOfRangeError:
                    break
            pbar.close()
