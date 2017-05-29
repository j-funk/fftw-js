# fftw-js
Javascript port of FFTW via Emscripten.

Install: `npm install fftw-js`

Thanks to [Chris Cannam](https://code.soundsoftware.ac.uk/projects/js-dsp-test) 
for the original transpile of FFTW.  See his 
[benchmarks](http://all-day-breakfast.com/js-dsp-test/fft/) of the various FFT 
algorithms available.  I've also put together an updated version which adds a 
few more algorithms not included in Chris' original set of tests 
[here](https://github.com/j-funk/js-dsp-test/).  

Example usage for `fftw-js` can be found in the test directory. 
The examples are only of the real valued transform / inverse, however complex 
to complex is also available.
