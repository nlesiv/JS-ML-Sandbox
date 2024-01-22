const tf = require("@tensorflow/tfjs");
const _ = require("lodash");
class LogisticRegression {
  constructor(features, labels, options) {
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);
    this.costHistory = [];

    this.options = Object.assign(
      {
        learningRate: 0.1,
        iterations: 1000,
        decisionBoundary: 0.5,
      },
      options
    );

    this.weights = tf.zeros([this.features.shape[1], this.labels.shape[1]]);
  }

  gradientDescent(features, labels) {
    const currentGuesses = features.matMul(this.weights).softmax();
    const differences = currentGuesses.sub(labels);

    const slopes = features
      .transpose()
      .matMul(differences)
      .div(features.shape[0]);

    return this.weights.sub(slopes.mul(this.options.learningRate));
  }
  train() {
    const batchQuantity = Math.floor(
      this.features.shape[0] / this.options.batchSize
    );
    for (let i = 0; i < this.options.iterations; ++i) {
      for (let j = 0; j < batchQuantity; ++j) {
        this.weights = tf.tidy(() => {
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
          return this.gradientDescent(featureSlice, labelSlice);
        });
      }
      // console.log(this.options.learningRate);

      this.recordCost();
      this.updateLearningRate();
    }
  }

  predict(observations) {
    let prediction = this.processFeatures(observations)
      .matMul(this.weights)
      .softmax()
      .argMax(1);

    return prediction;
  }

  test(testFeatures, testLabels) {
    const predictions = this.predict(testFeatures);
    testLabels = tf.tensor(testLabels).argMax(1);
    const incorrect = predictions.notEqual(testLabels).sum().get();

    return (predictions.shape[0] - incorrect) / predictions.shape[0];
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
    // fix the division by 0 in the variance
    const filler = variance.cast("bool").logicalNot().cast("float32");
    this.mean = mean;
    this.variance = variance.add(filler);

    return features.sub(mean).div(this.variance.pow(0.5));
  }

  /*
   * REcord the error from the run and store it in the history for comparison.
   */
  recordCost() {
    const cost = tf.tidy(() => {
      // Cross Entropy (Vectorized)
      const guesses = this.features.matMul(this.weights).softmax();

      const termOne = this.labels.transpose().matMul(guesses.add(1e-7).log());

      // Modify the terms of for the cross entropy equation
      // Some weird .add(1e-7) is required because taking a log of 0 or -x will 
      // result in a NAN which will throw off the update learning rate logic
      // adding a very tiny number 1 x 10 ^-7 will add a very small number to the result
      const termTwo = this.labels
        .mul(-1)
        .add(1)
        .transpose()
        .matMul(guesses.mul(-1).add(1).add(1e-7).log());

      return termOne
        .add(termTwo)
        .div(this.features.shape[0])
        .mul(-1)
        .get(0, 0);
    });

    this.costHistory.push(cost);
  }

  updateLearningRate() {
    if (this.costHistory.length < 2) {
      return;
    }
    const length = this.costHistory.length;

    const lastValue = this.costHistory[length - 1];
    const secondLast = this.costHistory[length - 2];

    if (lastValue > secondLast) {
      // Reduce learning rate
      this.options.learningRate /= 2;
    } else {
      // Increase learning rate.
      this.options.learningRate *= 1.05;
    }
  }
}

module.exports = LogisticRegression;
