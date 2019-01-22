// for use with webpack v4
const path = require("path")

const webpack = require("webpack")

module.exports = {
    mode: 'production',
    entry: [
        "./src/index.js",
    ],
    output: {
        filename: "bundle.js",
        path: path.resolve(__dirname, "dist")
    },
    module: {
        rules: [
            {
                test: /\.js$/,
                include: path.resolve(__dirname, "src"),
                loader: "babel-loader"
            }
        ]
    },
    plugins: [new webpack.IgnorePlugin(/(fs)/)], // for emscripten build
    watchOptions: {
        ignored: /node_modules/
    }
}
