#!/bin/bash

set -e

(
    cd fftw-3.3.4
    emconfigure ./configure --disable-fortran --enable-single && emmake make
)

make -f Makefile.emscripten

