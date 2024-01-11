require('@tensorflow/tfjs-node');
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");

const LinearRegression = require("./linear-regression");

let { features, labels, testFeatures, testLabels} = loadCSV("./cars.csv", {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower'],
    labelColumns: ["mpg"]
});

// console.log(features, labels)

const regression = new LinearRegression(features, labels, {
    learningRate: 0.0001,
    iterations: 100
})

regression.train();

// console.log("Updated M is: ", regression.weights.get(1, 0));
// console.log("updated B is: ", regression.weights.get(0, 0));
const r2 = regression.test(testFeatures, testLabels);

console.log("R2 is: ", r2);