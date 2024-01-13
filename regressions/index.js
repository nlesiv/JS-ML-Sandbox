require('@tensorflow/tfjs-node');
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");

const LinearRegression = require("./linear-regression");
const plot = require("node-remote-plot");

let { features, labels, testFeatures, testLabels} = loadCSV("./cars.csv", {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'weight', 'displacement'],
    labelColumns: ["mpg"]
});

// console.log(features, labels)

const regression = new LinearRegression(features, labels, {
    learningRate: .1,
    iterations: 100
})

regression.features.print();

regression.train();

// console.log("Updated M is: ", regression.weights.get(1, 0));
// console.log("updated B is: ", regression.weights.get(0, 0));
const r2 = regression.test(testFeatures, testLabels);

plot({
    x: regression.mseHistory,
    xLabel: "Iteration #",
    yLabel: "Mean Squared Error"
})
console.log("R2 is: ", r2);