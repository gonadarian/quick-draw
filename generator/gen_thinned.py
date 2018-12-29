import time as t
import numpy as np
import random as rand
from libs.concepts import Concept
from generator.gen_centered_lines import generate_thinning_pair as line_sampler
from generator.gen_centered_ellipses import generate_thinning_pair as ellipse_sampler
from generator.gen_centered_bezier import generate_thinning_pair as bezier_sampler


samplers = {
    Concept.LINE: line_sampler,
    Concept.ELLIPSE: ellipse_sampler,
    Concept.BEZIER: bezier_sampler,
}


def gen_thinning():
    images_clear = []
    images_aa = []

    concepts = [Concept.LINE, Concept.ELLIPSE, Concept.BEZIER]
    for concept in concepts:
        for i in range(1000):
            sample_clear, sample_aa = samplers[concept]()
            images_clear.append(sample_clear)
            images_aa.append(sample_aa)

    timestamp = int(t.time())

    images_aa = np.array(images_aa)
    print('shape:', images_aa.shape)
    m, h, w = images_aa.shape
    filename = 'data/mix/mix-thinned-samples-{}-{}x{}x{}x1.npy'
    filename = filename.format(timestamp, m, h, w)
    np.save(filename, images_aa)

    images_clear = np.array(images_clear)
    print('shape:', images_clear.shape)
    m, h, w = images_clear.shape
    filename = 'data/mix/mix-thinned-targets-{}-{}x{}x{}x1.npy'
    filename = filename.format(timestamp, m, h, w)
    np.save(filename, images_clear)


if __name__ == '__main__':
    rand.seed(1)

    gen_thinning()

    print('end')
