const tf = require("@tensorflow/tfjs");
const _ = require("lodash");
class LinearRegression {
  constructor(features, labels, options) {
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);
    this.mseHistory = [];

    this.options = Object.assign(
      {
        learningRate: 0.1,
        iterations: 1000,
      },
      options
    );

    this.weights = tf.zeros([this.features.shape[1], 1]);
  }

  gradientDescent(features, labels) {
    const currentGuesses = features.matMul(this.weights);
    const differences = currentGuesses.sub(labels);

    const slopes = features
      .transpose()
      .matMul(differences)
      .div(features.shape[0]);

    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
  }
  train() {
    const batchQuantity = Math.floor(
      this.features.shape[0] / this.options.batchSize
    );
    for (let i = 0; i < this.options.iterations; ++i) {
      for (let j = 0; j < batchQuantity; ++j) {
        const { batchSize } = this.options;
        const startIndex = j * batchSize;
        const featureSlice = this.features.slice(
          [startIndex, 0],
          [batchSize, -1]
        );
        const labelSlice = this.labels.slice(
          [startIndex, 0],
          [batchSize, -1]
        );
        this.gradientDescent(featureSlice, labelSlice);
      }
      // console.log(this.options.learningRate);

      this.recordMSE();
      this.updateLearningRate();
    }
  }

  predict(observations) {
    let prediction = this.processFeatures(observations).matMul(this.weights);

    return prediction;
  }

  test(testFeatures, testLabels) {
    testFeatures = this.processFeatures(testFeatures);
    testLabels = tf.tensor(testLabels);
    const predictions = testFeatures.matMul(this.weights);

    // predictions.print()

    const res = testLabels.sub(predictions).pow(2).sum().get();
    const tot = testLabels.sub(testLabels.mean()).pow(2).sum().get();

    return 1 - res / tot;
  }

  processFeatures(features) {
    features = tf.tensor(features);

    if (this.mean && this.variance) {
      features = features.sub(this.mean).div(this.variance.pow(0.5));
    } else {
      features = this.standardize(features);
    }
    features = tf.ones([features.shape[0], 1]).concat(features, 1);
    return features;
  }

  standardize(features) {
    const { mean, variance } = tf.moments(features, 0);

    this.mean = mean;
    this.variance = variance;

    return features.sub(mean).div(variance.pow(0.5));
  }
  /*
   * This will record the history of Mean Squred Error result (NOT the Slope of it, but the actual value)
   * Compare it to the previous result to better adjust the learning rate, to prevent learning too slow
   * or overshooting the min.
   */
  recordMSE() {
    const mse = this.features
      .matMul(this.weights)
      .sub(this.labels)
      .pow(2)
      .sum()
      .div(this.features.shape[0])
      .get();

    this.mseHistory.push(mse);
  }

  updateLearningRate() {
    if (this.mseHistory.length < 2) {
      return;
    }
    const length = this.mseHistory.length;

    const lastValue = this.mseHistory[length - 1];
    const secondLast = this.mseHistory[length - 2];

    if (lastValue > secondLast) {
      // Reduce learning rate
      this.options.learningRate /= 2;
    } else {
      // Increase learning rate.
      this.options.learningRate *= 1.05;
    }
  }
}

module.exports = LinearRegression;
