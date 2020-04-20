import os
import json

import numpy as np
import scipy.fftpack

cur_dir = os.path.dirname(os.path.abspath(__file__))


def flatten_interleave(a):
    a = a.flatten()
    if a.dtype == np.complex128 or a.dtype == np.complex64:
        b = np.zeros(2*a.shape[0])
        b[::2] = a.real
        b[1::2] = a.imag
        return b
    else:
        return a


def compute_transform(fn, x):
    return {
        'size': list(x.shape),
        'x': flatten_interleave(x).tolist(),
        'y': flatten_interleave(fn(x)).tolist()
    }


def main():
    n = 32

    x_real = np.random.rand(n)
    x_complex = x_real + 1j*np.random.rand(n)
    x_2_real = np.random.rand(n, n)
    x_2_complex = x_2_real + 1j*np.random.rand(n, n)

    # x_real = np.arange(n)
    # x_complex = x_real + 1j*np.arange(n)
    # x_2_real = np.arange(n*n).reshape((n, n))
    # x_2_complex = x_2_real + 1j*np.arange(n*n).reshape((n, n))

    test_vectors = {
        'c2c': {
            'fft1d': compute_transform(np.fft.fft, x_complex),
            'fft2d': compute_transform(np.fft.fft2, x_2_complex)
        },
        'r2r': {
            'dct1d': compute_transform(
                lambda x: scipy.fftpack.dct(x, norm=None), x_real),
            'dct2d': compute_transform(scipy.fftpack.dctn, x_2_real),
            'dst1d': compute_transform(scipy.fftpack.dst, x_real),
            'dst2d': compute_transform(scipy.fftpack.dstn, x_2_real),
            'fft1d': compute_transform(scipy.fftpack.rfft, x_real),
        },
        'r2c': {
            'fft1d': compute_transform(np.fft.rfft, x_real),
            'fft2d': compute_transform(np.fft.rfft2, x_2_real),
        }
    }

    file_path = os.path.join(cur_dir, 'test_vectors.json')

    with open(file_path, 'w') as f:
        json.dump(test_vectors, f)


if __name__ == "__main__":
    main()
