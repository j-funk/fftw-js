"use strict";

var FFTWModule = require('./FFTW');
var fftwModule = FFTWModule({});

var fftwf_plan_dft_r2c_1d = fftwModule.cwrap(
    'fftwf_plan_dft_r2c_1d', 'number', ['number', 'number', 'number', 'number']
);

var fftwf_plan_dft_c2r_1d = fftwModule.cwrap(
    'fftwf_plan_dft_c2r_1d', 'number', ['number', 'number', 'number', 'number']
);

var fftwf_plan_r2r_1d = fftwModule.cwrap(
    'fftwf_plan_r2r_1d', 'number', ['number', 'number', 'number', 'number']
);

var fftwf_execute = fftwModule.cwrap(
    'fftwf_execute', 'void', ['number']
);

var fftwf_destroy_plan = fftwModule.cwrap(
    'fftwf_destroy_plan', 'void', ['number']
);

var FFTW_ESTIMATE = (1 << 6);
var FFTW_R2HC = 0;
var FFTW_HC2R = 1;


function FFT(size) {

    this.size = size;
    this.rptr = fftwModule._malloc(size*4 + (size+2)*4);
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
        fftwModule._free(this.rptr);
    }
}


function RFFT(size) {
    this.size = size;
    this.rptr = fftwModule._malloc(size*4 + size*4);
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
        fftwModule._free(this.rptr);
    }
}

module.exports = {
    FFT: FFT,
    RFFT: RFFT
};