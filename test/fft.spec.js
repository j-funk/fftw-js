var chai = require('chai');

var FFTW = require('../src/main');
var A2_1024 = require('./audioBuffer.js');


var scaleTransform = function(trans, size) {
    var i = 0,
        bSi = 1.0 / size,
        x = trans;
    while(i < x.length) {
        x[i] *= bSi; i++;
    }
    return x;
};

function getMiscRealBuffer(size) {
    var result = new Float32Array(size);
    for (var i = 0; i < result.length; i++)
        result[i] = (i % 2) / 4.0;
    return result;
}

describe('fftw-js', function() {

    it('should successfully transform and invert real valued input buffer', function() {
        var size = A2_1024.length;
        var fftr = new FFTW.RFFT(size);
        var transform = fftr.forward(A2_1024);
        var transScaled = scaleTransform(transform, size);
        var a2_again = fftr.inverse(transScaled);
        for(var i = 0; i < size; i++) {
            chai.expect(A2_1024[i]).to.be.closeTo(a2_again[i], 0.0000005);
        }

        // Clean up after you're done - NOTE:: dispose will modify the transform array slightly,
        // so only dispose after any use of results are complete
        fftr.dispose();
    });

    it('should successfully transform and invert non-power-of-2 buffers', function() {
        var non2PowSize = 1536;  // 1.5 times test buffer size
        var buffer = getMiscRealBuffer(non2PowSize);
        var fftr = new FFTW.RFFT(non2PowSize);
        var transform = fftr.forward(buffer);
        var transScaled = scaleTransform(transform, non2PowSize);
        var backAgain = fftr.inverse(transScaled);

        for(var i = 0; i < non2PowSize; i++) {
            chai.expect(buffer[i]).to.be.closeTo(backAgain[i], 0.0000005);
        }
        fftr.dispose();
    });
});