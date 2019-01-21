"use strict";

var FFTWModule = require('./FFTW.js');
var fftwModule = FFTWModule({});

const fftwf_malloc = fftwModule.cwrap(
  'fftwf_malloc', 'number', ['number']
)

const fftwf_plan_dft_r2c_1d = fftwModule.cwrap(
    'fftwf_plan_dft_r2c_1d', 'number', ['number', 'number', 'number', 'number']
);

const fftwf_plan_dft_c2r_1d = fftwModule.cwrap(
    'fftwf_plan_dft_c2r_1d', 'number', ['number', 'number', 'number', 'number']
);

const fftwf_plan_r2r_1d = fftwModule.cwrap(
    'fftwf_plan_r2r_1d', 'number', ['number', 'number', 'number', 'number']
);

// fftw_plan fftw_plan_dft_1d(int n, fftw_complex *in, fftw_complex *out,
//                            int sign, unsigned flags);
const fftwf_plan_dft_1d = fftwModule.cwrap(
  'fftwf_plan_dft_1d', 'number', ['number', 'number', 'number', 'number', 'number']
)

// fftw_plan fftw_plan_dft_2d(int n0, int n1,
//                            fftw_complex *in, fftw_complex *out,
//                            int sign, unsigned flags);
const fftwf_plan_dft_2d = fftwModule.cwrap(
  'fftwf_plan_dft_2d', 'number', ['number', 'number', 'number', 'number', 'number', 'number']
)


const fftwf_execute = fftwModule.cwrap(
    'fftwf_execute', 'void', ['number']
);

const fftwf_destroy_plan = fftwModule.cwrap(
    'fftwf_destroy_plan', 'void', ['number']
);

const fftwf_free = fftwModule.cwrap(
  'fftwf_free', 'void', ['number']
)

const FFTW_ESTIMATE = (1 << 6);
const FFTW_R2HC = 0;
const FFTW_HC2R = 1;
const FFTW_FORWARD = -1
const FFTW_BACKWARD = 1


function FFTC2C2D (n0, n1) {

  this.n0 = n0
  this.n1 = n1
  this.size = n0 * n1

  this.c0ptr = fftwf_malloc(2*4*this.size)
  this.c1ptr = fftwf_malloc(2*4*this.size)

  this.c0 = new Float32Array(fftwModule.HEAPU8.buffer, this.c0ptr, 2*this.size) // two for complex
  this.c1 = new Float32Array(fftwModule.HEAPU8.buffer, this.c1ptr, 2*this.size)

  this.fplan = fftwf_plan_dft_2d(this.n0, this.n1, this.c0ptr, this.c1ptr, FFTW_FORWARD, FFTW_ESTIMATE)
  this.iplan = fftwf_plan_dft_2d(this.n0, this.n1, this.c1ptr, this.c0ptr, FFTW_BACKWARD, FFTW_ESTIMATE)

  this.forward = function(cpx) {
      this.c0.set(cpx);
      fftwf_execute(this.fplan);
      return new Float32Array(fftwModule.HEAPU8.buffer, this.c1ptr, 2*this.size)
  }

  this.inverse = function(cpx) {
      this.c1.set(cpx);
      fftwf_execute(this.iplan);
      return new Float32Array(fftwModule.HEAPU8.buffer, this.c0ptr, 2*this.size)
  }

  this.dispose = function() {
      fftwf_destroy_plan(this.fplan)
      fftwf_destroy_plan(this.iplan)
      fftwf_free(this.c0ptr)
      fftwf_free(this.c1ptr)
  }

}



function FFTC2C (size) {

  this.size = size
  // this.c0ptr = fftwf_malloc(2*4*size + 2*4*size)
  // this.c1ptr = this.c0ptr
  this.c0ptr = fftwf_malloc(2*4*size)
  this.c1ptr = fftwf_malloc(2*4*size)

  this.c0 = new Float32Array(fftwModule.HEAPU8.buffer, this.c0ptr, 2*size)
  this.c1 = new Float32Array(fftwModule.HEAPU8.buffer, this.c1ptr, 2*size)

  this.fplan = fftwf_plan_dft_1d(size, this.c0ptr, this.c1ptr, FFTW_FORWARD, FFTW_ESTIMATE)
  this.iplan = fftwf_plan_dft_1d(size, this.c1ptr, this.c0ptr, FFTW_BACKWARD, FFTW_ESTIMATE)

  this.forward = function(cpx) {
      this.c0.set(cpx);
      fftwf_execute(this.fplan);
      return new Float32Array(fftwModule.HEAPU8.buffer, this.c1ptr, 2*this.size)
  }

  this.inverse = function(cpx) {
      this.c1.set(cpx);
      fftwf_execute(this.iplan);
      return new Float32Array(fftwModule.HEAPU8.buffer, this.c0ptr, 2*this.size)
  }

  this.dispose = function() {
      fftwf_destroy_plan(this.fplan)
      fftwf_destroy_plan(this.iplan)
      fftwf_free(this.c0ptr)
      fftwf_free(this.c1ptr)
  }

}


function FFTR2C (size) {

    this.size = size;
    this.rptr = fftwf_malloc(size*4 + (size+2)*4);
    this.cptr = this.rptr + size*4;
    this.r = new Float32Array(fftwModule.HEAPU8.buffer, this.rptr, size);
    this.c = new Float32Array(fftwModule.HEAPU8.buffer, this.cptr, size+2);

    this.fplan = fftwf_plan_dft_r2c_1d(size, this.rptr, this.cptr, FFTW_ESTIMATE);
    this.iplan = fftwf_plan_dft_c2r_1d(size, this.cptr, this.rptr, FFTW_ESTIMATE);

    this.forward = function(real) {
        this.r.set(real);
        fftwf_execute(this.fplan);
        return new Float32Array(fftwModule.HEAPU8.buffer, this.cptr, this.size+2);
    }

    this.inverse = function(cpx) {
        this.c.set(cpx);
        fftwf_execute(this.iplan);
        return new Float32Array(fftwModule.HEAPU8.buffer, this.rptr, this.size);
    }

    this.dispose = function() {
        fftwf_destroy_plan(this.fplan);
        fftwf_destroy_plan(this.iplan);
        fftwf_free(this.rptr);
    }
}


function FFTR2R(size) {
    this.size = size;
    this.rptr = fftwf_malloc(size*4 + size*4);
    this.cptr = this.rptr;
    this.r = new Float32Array(fftwModule.HEAPU8.buffer, this.rptr, size);
    this.c = new Float32Array(fftwModule.HEAPU8.buffer, this.cptr, size);

    this.fplan = fftwf_plan_r2r_1d(size, this.rptr, this.cptr, FFTW_R2HC, FFTW_ESTIMATE);
    this.iplan = fftwf_plan_r2r_1d(size, this.cptr, this.rptr, FFTW_HC2R, FFTW_ESTIMATE);

    this.forward = function(real) {
        this.r.set(real);
        fftwf_execute(this.fplan);
        return new Float32Array(fftwModule.HEAPU8.buffer, this.cptr, this.size);
    };

    this.inverse = function(cpx) {
        this.c.set(cpx);
        fftwf_execute(this.iplan);
        return new Float32Array(fftwModule.HEAPU8.buffer, this.rptr, this.size);
    };

    this.dispose = function() {
        fftwf_destroy_plan(this.fplan);
        fftwf_destroy_plan(this.iplan);
        fftwf_free(this.rptr);
    }
}

module.exports = {
    FFTC2C: FFTC2C,
    FFTR2C: FFTR2C,
    FFTR2R: FFTR2R,
    FFTC2C2D: FFTC2C2D
};
