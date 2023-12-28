const outputs = [];
function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  outputs.push([dropPosition, bounciness, size, bucketLabel]);
  // console.log(outputs);
}

function runAnalysis() {
  const testSetSize = 100;

  const k = 10;
  // Write code here to analyze stuff
  // const [testSet, trainingSet] = splitDataset(minMax(outputs,3), testSetSize);

  _.range(0, 3).forEach((feature) => {
    const data = _.map(outputs, row => [row[feature], _.last(row)])
    const [testSet, trainingSet] = splitDataset(minMax(data,1), testSetSize);

    const accuracy = _.chain(testSet)
      .filter((testPoint) => {
        return knn(trainingSet, _.initial(testPoint), k) === _.last(testPoint);
      })
      .size()
      .divide(testSetSize)
      .value();

    // console.log("Arrucarcy for K of ", k, "Accurancy: ", accuracy);
    console.log("For feature of", feature, " accuracy is", accuracy);
  });
}

function knn(data, point, k) {
  return _.chain(data)
    .map((row) => {
      return [distance(_.initial(row), point), _.last(row)];
    }) // Get the distance of each drop point to the predictionPoint
    .sortBy((row) => row[0]) // Sort the resulting Array in accesding order based on the first column(the drop point)
    .slice(0, k) // Slice off the top K items that we are considering
    .countBy((row) => row[1]) // Create a map that counts how many times the number in position 1 occurs (The result)
    .toPairs() // Convert the map of (key => value) pairs to an array of arrays.
    .sortBy((row) => row[1]) // Sort the result array of arrays based on the output variable in ascending order (Landing position)
    .last() // pick the last item in the list, which is the highest occuring item in the list (most frequent)
    .parseInt() // convert the item to integer
    .value();
}
function distanceSimple(pointA, pointB) {
  return Math.abs(pointA - pointB);
}

function distance(pointA, pointB) {
  // PointA, PointB is an array of features.
  // Pathegorean Theorem applied to all attributes [a^2 + b^2 + c^2....] ^ 0.5
  return (
    _.chain(pointA)
      .zip(pointB)
      .map(([a, b]) => {
        return (a - b) ** 2;
      })
      .sum()
      .value() ** 0.5
  );
}

function splitDataset(data, testCount) {
  const shuffled = _.shuffle(data);

  const testSet = _.slice(shuffled, 0, testCount);
  const trainingSet = _.slice(shuffled, testCount);

  return [testSet, trainingSet];
}

function minMax(data, featureCount){
  const clonedData = _.cloneDeep(data);

  for(let i = 0; i < featureCount; ++i) {
    const column = clonedData.map(row => row[i]);

    const min = _.min(column);
    const max = _.max(column);

    for (let j = 0; j < clonedData.length; ++j) {
      clonedData[j][i] = (clonedData[j][i] - min)/ (max - min);
    }
  }

  return clonedData;
}
