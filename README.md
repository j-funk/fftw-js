# fftw-js
Javascript port of FFTW via Emscripten.

Install: `npm install fftw-js`

I've added new benchmarks against other 
popular Javascript algorithms [here](https://github.com/j-funk/js-dsp-test/).    

Example usage for `fftw-js` can be found in the test directory. 
The examples are only of the real valued transform / inverse, however complex 
to complex is also available.

Thanks to [Chris Cannam](https://code.soundsoftware.ac.uk/projects/js-dsp-test)
for the original benchmark tool and for transpiling FFTW. This version has 
been optimized to perform much faster since forking Chris' version.
