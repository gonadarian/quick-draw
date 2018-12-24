import time as t
import numpy as np
import random as rand
import libs.generators as gens
from libs.concepts import Concept
from generator.gen_centered_lines import generate_image as line_sampler
from generator.gen_centered_ellipses import generate_image as ellipse_sampler
from generator.gen_centered_bezier import generate_image as bezier_sampler


samplers = {
    Concept.LINE: line_sampler,
    Concept.ELLIPSE: ellipse_sampler,
    Concept.BEZIER: bezier_sampler,
}


def gen_thinning(dim=27):

    sample = samplers[Concept.ELLIPSE](antialias=True)
    gens.show_image(sample)
    shifted_sample_list = gens.generated_shifted_samples(sample, dim=dim, density=0.1)
    sample = rand.choice(shifted_sample_list)[0]
    gens.show_image(sample)
    return

    timestamp = int(t.time())

    filename = 'data/{}/{}-mixed-samples-{}-{}x{}x{}x1.npy'
    filename = filename.format(concept.code, concept.code, timestamp, m, dim, dim)
    np.save(filename, samples_mix)

    filename = 'data/{}/{}-mixed-encodings-{}-{}x{}x{}x{}.npy'
    filename = filename.format(concept.code, concept.code, timestamp, m, dim, dim, channels_full)
    np.save(filename, encodings_mix)


if __name__ == '__main__':
    # rand.seed(1)

    gen_thinning()

    print('end')
