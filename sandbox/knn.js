_ = require("lodash");

const outputs = [
  [10, 0.5, 16, 1],
  [200, 0.5, 16, 4],
  [360, 0.5, 16, 4],
  [600, 0.5, 16, 4],
];

const predictionsPoint = 300;
const k = 3;
function distance(point) {
  return Math.abs(point - predictionsPoint);
}

_.chain(outputs)
  .map((row) => [distance(row[0]), row[3]]) // Get the distance of each drop point to the predictionPoint
  .sortBy((row) => row[0]) // Sort the resulting Array in accesding order based on the first column(the drop point)
  .slice(0,k) // Slice off the top K items that we are considering
  .countBy(row => row[1]) // Create a map that counts how many times the number in position 1 occurs (The result)
  .toPairs() // Convert the map of (key => value) pairs to an array of arrays. 
  .sortBy(row => row[1]) // Sort the result array of arrays based on the output variable in ascending order (Landing position)
  .last() // pick the last item in the list, which is the highest occuring item in the list (most frequent)
  .parseInt() // convert the item to integer
  .value();
