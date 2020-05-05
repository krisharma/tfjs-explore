/**
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
 */
async function getData() {
  const carsDataReq = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json'); //retrieve data from source
  const carsData = await carsDataReq.json(); 
  const cleaned = carsData.map(car => ({
    mpg: car.Miles_per_Gallon,
    horsepower: car.Horsepower,
  }))
  .filter(car => (car.mpg != null && car.horsepower != null));
  
  return cleaned;
}

/**
 * Main function that runs the entire loading, training, and testing pipeline
 */
async function run() {
  // Load and plot the original input data that we are going to train on.
  const data = await getData();
  const values = data.map(d => ({
      x: d.horsepower,
      y: d.mpg,
  }));

  // plot initial data
  tfvis.render.scatterplot(
      {name: 'Horsepower v MPG'},
      {values}, 
      {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300
      }
  );

  // Create the model
  const model = createModel();  
  tfvis.show.modelSummary({name: 'Model Summary'}, model);

  // Convert the data to a form we can use for training.
  const tensorData = convertToTensor(data);
  const {inputs, labels} = tensorData;
      
  // Train the model  
  await trainModel(model, inputs, labels);

  // Test model
  testModel(model, data, tensorData);
}

/**
 * Creates the model architecture
 * @returns {tf.sequential} - The model architecture in sequential form
 */
function createModel() {
  // Create a sequential model
  const model = tf.sequential(); 

  // Add a single input layer
  model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));

  // Hidden layer
  model.add(tf.layers.dense({units: 50, activation: 'sigmoid'}));

  // Add an output layer
  model.add(tf.layers.dense({units: 1, useBias: true}));

  return model;
}


/**
 * Convert the input data to tensors. 
 * We will also perform _shuffling_ and _normalization_ of the data
 * 
 * @param {array} - Input data as an array of Car objects
 * @returns {Object} - returns an object containing the input and label tensors
 *                     as well as the range information
 */
function convertToTensor(data) {
    // Wrapping these calculations in a tidy will dispose any 
    // intermediate tensors.
    
  return tf.tidy(() => {
    // Step 1. Shuffle the data    
    tf.util.shuffle(data);

    // Step 2. Convert data to Tensor
    const inputs = data.map(d => d.horsepower)
    const labels = data.map(d => d.mpg);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();  
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      // Return the min/max bounds so we can use them later.
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    }
  });  
 }

/**
 * Trains the given model with the given data
 * 
 * @param {tf.sequential} model - stacked layers ready for training
 * @param {tf.tensor2d} inputs - input values of training data as tensor
 * @param {tf.tensor2d} labels - labels for training data
 * @returns {tf.sequential} - trained model
 */
async function trainModel(model, inputs, labels) {
  // Prepare the model for training.  
  model.compile({
      optimizer: tf.train.adam(),
      loss: tf.losses.meanSquaredError,
      metrics: ['mse'],
  });

  const batchSize = 32;
  const epochs = 50;

  return await model.fit(inputs, labels, {
      batchSize,
      epochs,
      shuffle: true,


      callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'mse'], 
      { height: 200, callbacks: ['onEpochEnd'] }
      )
      });
}

/**
 * Tests the given model with the given data
 * 
 * @param {tf.sequential} model - trained model
 * @param {tensor2d} inputData - original input data (for plotting)
 * @param {Object} normalizationData - contains feature ranges for input values and label values
 */
function testModel(model, inputData, normalizationData) {
  const {inputMax, inputMin, labelMin, labelMax} = normalizationData;  
  
  // Generate predictions for a uniform range of numbers between 0 and 1;
  // We un-normalize the data by doing the inverse of the min-max scaling 
  // that we did earlier.
  const [xs, preds] = tf.tidy(() => {
    
    const xs = tf.linspace(0, 1, 100);      
    const preds = model.predict(xs.reshape([100, 1]));      
    
    const unNormXs = xs
      .mul(inputMax.sub(inputMin))
      .add(inputMin);
    
    const unNormPreds = preds
      .mul(labelMax.sub(labelMin))
      .add(labelMin);
    
    // Un-normalize the data
    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });
  
  
  const predictedPoints = Array.from(xs).map((val, i) => {
    return {x: val, y: preds[i]}
  });
  
  const originalPoints = inputData.map(d => ({
    x: d.horsepower, y: d.mpg,
  }));
  
  
  tfvis.render.scatterplot(
    {name: 'Model Predictions vs Original Data'}, 
    {values: [originalPoints, predictedPoints], series: ['original', 'predicted']}, 
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300
    }
  );
}

/**
 * Simple button handler for showing the visualization window after it has been hidden
 */
function showVis() {
  tfvis.visor().open();
}

document.addEventListener('DOMContentLoaded', run);