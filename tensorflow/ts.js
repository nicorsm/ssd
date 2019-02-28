/*
    // 6 ingressi reali 
    I       // 2 strati nascosti completamente connessi        
    I   H   H        
    I   H   H   O   
    I   H   H   O   // 3 uscite reali
    I   H   H   O
    I   
*/
const model = tf.sequential();

//config for layer
const config_hidden_1 = {
  inputShape: [6],
  activation: 'sigmoid',
  units: 4
}
const config_hidden_2 = {
  units: 4,
  activation: 'sigmoid'
}

const config_output = {
  units: 3,
  activation: 'sigmoid'
}

//defining the hidden and output layer
const hidden1 = tf.layers.dense(config_hidden_1);
const hidden2 = tf.layers.dense(config_hidden_2);
const output = tf.layers.dense(config_output);

//adding layers to model
model.add(hidden1);
model.add(hidden2);
model.add(output);

//define an optimizer
const optimize = tf.train.sgd(0.1);

//config for model
const config = {
  optimizer: optimize,
  loss: 'meanSquaredError'
}

//compiling the model
model.compile(config);
console.log('Model Successfully Compiled');

//Dummy training data
const x_train = tf.tensor([
  [0.1, 0.5, 0.1, 0.1, 0.5, 0.1],
  [0.9, 0.3, 0.4, 0.1, 0.5, 0.1],
  [0.4, 0.5, 0.5, 0.1, 0.5, 0.1],
  [0.7, 0.1, 0.9, 0.1, 0.5, 0.1]
])

//Dummy training labels
const y_train = tf.tensor([
  [0.2, 0.8, 0.1],
  [0.9, 0.10, 0.1],
  [0.4, 0.6, 0.1],
  [0.5, 0.5, 0.1]
])

//Dummy testing data
const x_test = tf.tensor([
  [0.9, 0.1, 0.5, 0.1, 0.5, 0.1]
])

train_data().then(function () {
  console.log('Training is Complete');
  console.log('Predictions :');
  model.predict(x_test).print();
})
async function train_data() {
  for (let i = 0; i < 10; i++) {
    const res = await model.fit(x_train, y_train, epoch = 1000, batch_size = 10);
    console.log(res.history.loss[0]);
  }
}

// Original code https://medium.freecodecamp.org/get-to-know-tensorflow-js-in-7-minutes-afcd0dfd3d2f