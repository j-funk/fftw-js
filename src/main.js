const FFTWModule = require('./FFTW.js')

const FFTW_ESTIMATE = (1 << 6)

const FFTW_R2HC = 0
const FFTW_HC2R = 1
const FFTW_DHT = 2
const FFTW_REDFT00 = 3
const FFTW_REDFT10 = 4
const FFTW_REDFT01 = 5
const FFTW_REDFT11 = 6
const FFTW_RODFT00 = 7
const FFTW_RODFT10 = 9
const FFTW_RODFT01 = 8
const FFTW_RODFT11 = 10

const FFTW_FORWARD = -1
const FFTW_BACKWARD = 1

const fftw = FFTWModule()
fftw['ready'] = false
fftw['onReady'] = null

fftw['onRuntimeInitialized'] = () => {
  const fftwf_plan_dft_r2c_1d = fftw.cwrap(
    'fftwf_plan_dft_r2c_1d', 'number', ['number', 'number', 'number', 'number']
  )

  const fftwf_plan_dft_c2r_1d = fftw.cwrap(
    'fftwf_plan_dft_c2r_1d', 'number', ['number', 'number', 'number', 'number']
  )

  // fftw_plan fftw_plan_r2r_1d(int n, double *in, double *out,
  //                            fftw_r2r_kind kind, unsigned flags);
  const fftwf_plan_r2r_1d = fftw.cwrap(
    'fftwf_plan_r2r_1d', 'number', ['number', 'number', 'number', 'number']
  )

  // fftw_plan fftw_plan_r2r_2d(int n0, int n1, double *in, double *out,
  //                            fftw_r2r_kind kind0, fftw_r2r_kind kind1,
  //                            unsigned flags);
  const fftwf_plan_r2r_2d = fftw.cwrap(
    'fftwf_plan_r2r_2d', 'number', ['number', 'number', 'number', 'number', 'number', 'number', 'number']
  )
  // fftw_plan fftw_plan_dft_1d(int n, fftw_complex *in, fftw_complex *out,
  //                            int sign, unsigned flags)
  const fftwf_plan_dft_1d = fftw.cwrap(
    'fftwf_plan_dft_1d', 'number', ['number', 'number', 'number', 'number', 'number']
  )

  // fftw_plan fftw_plan_dft_2d(int n0, int n1,
  //                            fftw_complex *in, fftw_complex *out,
  //                            int sign, unsigned flags)
  const fftwf_plan_dft_2d = fftw.cwrap(
    'fftwf_plan_dft_2d', 'number', ['number', 'number', 'number', 'number', 'number', 'number']
  )

  const fftwf_execute = fftw.cwrap(
      'fftwf_execute', 'void', ['number']
  )

  const fftwf_destroy_plan = fftw.cwrap(
      'fftwf_destroy_plan', 'void', ['number']
  )

  const fftwf_free = fftw.cwrap(
    'fftwf_free', 'void', ['number']
  )

  const fftwf_malloc = fftw.cwrap(
    'fftwf_malloc', 'number', ['number']
  )

  fftw.c2c = {}
  fftw.r2c = {}
  fftw.r2r = {}

  fftw.c2c.fft2d = function (n0, n1) {

    this.n0 = n0
    this.n1 = n1
    this.size = n0 * n1

    this.c0ptr = fftwf_malloc(2*4*this.size)
    this.c1ptr = fftwf_malloc(2*4*this.size)

    this.c0 = new Float32Array(fftw.HEAPU8.buffer, this.c0ptr, 2*this.size) // two for complex
    this.c1 = new Float32Array(fftw.HEAPU8.buffer, this.c1ptr, 2*this.size)

    this.fplan = fftwf_plan_dft_2d(this.n0, this.n1, this.c0ptr, this.c1ptr, FFTW_FORWARD, FFTW_ESTIMATE)
    this.iplan = fftwf_plan_dft_2d(this.n0, this.n1, this.c1ptr, this.c0ptr, FFTW_BACKWARD, FFTW_ESTIMATE)

    this.forward = function(cpx) {
        this.c0.set(cpx)
        fftwf_execute(this.fplan)
        return new Float32Array(fftw.HEAPU8.buffer, this.c1ptr, 2*this.size)
    }

    this.inverse = function(cpx) {
        this.c1.set(cpx)
        fftwf_execute(this.iplan)
        return new Float32Array(fftw.HEAPU8.buffer, this.c0ptr, 2*this.size)
    }

    this.dispose = function() {
        fftwf_destroy_plan(this.fplan)
        fftwf_destroy_plan(this.iplan)
        fftwf_free(this.c0ptr)
        fftwf_free(this.c1ptr)
    }

  }

  fftw.c2c.fft1d = function (size) {

    this.size = size
    // this.c0ptr = fftwf_malloc(2*4*size + 2*4*size)
    // this.c1ptr = this.c0ptr
    this.c0ptr = fftwf_malloc(2*4*this.size)
    this.c1ptr = fftwf_malloc(2*4*this.size)

    this.c0 = new Float32Array(fftw.HEAPU8.buffer, this.c0ptr, 2*size)
    this.c1 = new Float32Array(fftw.HEAPU8.buffer, this.c1ptr, 2*size)

    this.fplan = fftwf_plan_dft_1d(size, this.c0ptr, this.c1ptr, FFTW_FORWARD, FFTW_ESTIMATE)
    this.iplan = fftwf_plan_dft_1d(size, this.c1ptr, this.c0ptr, FFTW_BACKWARD, FFTW_ESTIMATE)

    this.forward = function(cpx) {
        this.c0.set(cpx)
        fftwf_execute(this.fplan)
        return new Float32Array(fftw.HEAPU8.buffer, this.c1ptr, 2*this.size)
    }

    this.inverse = function(cpx) {
        this.c1.set(cpx)
        fftwf_execute(this.iplan)
        return new Float32Array(fftw.HEAPU8.buffer, this.c0ptr, 2*this.size)
    }

    this.dispose = function() {
        fftwf_destroy_plan(this.fplan)
        fftwf_destroy_plan(this.iplan)
        fftwf_free(this.c0ptr)
        fftwf_free(this.c1ptr)
    }

  }


  fftw.r2c.fft1d = function (size) {

      this.size = size
      this.rptr = fftwf_malloc(size*4 + (size+2)*4)
      this.cptr = this.rptr + size*4
      this.r = new Float32Array(fftw.HEAPU8.buffer, this.rptr, size)
      this.c = new Float32Array(fftw.HEAPU8.buffer, this.cptr, size+2)

      this.fplan = fftwf_plan_dft_r2c_1d(size, this.rptr, this.cptr, FFTW_ESTIMATE)
      this.iplan = fftwf_plan_dft_c2r_1d(size, this.cptr, this.rptr, FFTW_ESTIMATE)

      this.forward = function(real) {
          this.r.set(real)
          fftwf_execute(this.fplan)
          return new Float32Array(fftw.HEAPU8.buffer, this.cptr, this.size+2)
      }

      this.inverse = function(cpx) {
          this.c.set(cpx)
          fftwf_execute(this.iplan)
          return new Float32Array(fftw.HEAPU8.buffer, this.rptr, this.size)
      }

      this.dispose = function() {
          fftwf_destroy_plan(this.fplan)
          fftwf_destroy_plan(this.iplan)
          fftwf_free(this.rptr)
      }
  }

  const r2r1dFactory = (forwardType, inverseType) => {
    return function (size) {
        this.size = size
        this.rptr = fftwf_malloc(size*4 + size*4)
        this.cptr = this.rptr
        this.r = new Float32Array(fftw.HEAPU8.buffer, this.rptr, size)
        this.c = new Float32Array(fftw.HEAPU8.buffer, this.cptr, size)

        this.fplan = fftwf_plan_r2r_1d(size, this.rptr, this.cptr, forwardType, FFTW_ESTIMATE)
        this.iplan = fftwf_plan_r2r_1d(size, this.cptr, this.rptr, inverseType, FFTW_ESTIMATE)

        this.forward = function(real) {
            this.r.set(real)
            fftwf_execute(this.fplan)
            return new Float32Array(fftw.HEAPU8.buffer, this.cptr, this.size)
        }

        this.inverse = function(cpx) {
            this.c.set(cpx)
            fftwf_execute(this.iplan)
            return new Float32Array(fftw.HEAPU8.buffer, this.rptr, this.size)
        }

        this.dispose = function() {
            fftwf_destroy_plan(this.fplan)
            fftwf_destroy_plan(this.iplan)
            fftwf_free(this.rptr)
        }
      }
  }

  const r2r2dFactory = (forwardType, inverseType) => {
    return function (n0, n1) {
        this.n0 = n0
        this.n1 = n1
        this.size = this.n0 * this.n1
        this.rptr = fftwf_malloc(this.size*4)
        this.cptr = fftwf_malloc(this.size*4)
        this.r = new Float32Array(fftw.HEAPU8.buffer, this.rptr, this.size)
        this.c = new Float32Array(fftw.HEAPU8.buffer, this.cptr, this.size)

        this.fplan = fftwf_plan_r2r_2d(this.n0, this.n1, this.rptr, this.cptr, forwardType, forwardType, FFTW_ESTIMATE)
        this.iplan = fftwf_plan_r2r_2d(this.n0, this.n1, this.cptr, this.rptr, inverseType, inverseType, FFTW_ESTIMATE)

        this.forward = function(real) {
            this.r.set(real)
            fftwf_execute(this.fplan)
            return new Float32Array(fftw.HEAPU8.buffer, this.cptr, this.size)
        }

        this.inverse = function(cpx) {
            this.c.set(cpx)
            fftwf_execute(this.iplan)
            return new Float32Array(fftw.HEAPU8.buffer, this.rptr, this.size)
        }

        this.dispose = function() {
            fftwf_destroy_plan(this.fplan)
            fftwf_destroy_plan(this.iplan)
            fftwf_free(this.rptr)
        }
      }
  }

  fftw.r2r.fft1d = r2r1dFactory(FFTW_R2HC, FFTW_HC2R)
  fftw.r2r.dct1d = r2r1dFactory(FFTW_REDFT10, FFTW_REDFT01)
  fftw.r2r.dst1d = r2r1dFactory(FFTW_RODFT10, FFTW_RODFT01)

  fftw.r2r.fft2d = r2r2dFactory(FFTW_R2HC, FFTW_HC2R)
  fftw.r2r.dct2d = r2r2dFactory(FFTW_REDFT10, FFTW_REDFT01)
  fftw.r2r.dst2d = r2r2dFactory(FFTW_RODFT10, FFTW_RODFT01)

  fftw['ready'] = true
  if (fftw['onReady'] !== null) {
    fftw['onReady'](fftw)
  }
}

// fftw.ready =  function () {
//   return new Promise ((resolve, reject) => {
//     fftw['onRuntimeInitialized'] = resolve
//   }).then(() => {
//     return fftw
//   })
// }

module.exports = fftw
