/*
    // 6 ingressi reali 
    I       // 2 strati nascosti completamente connessi        
    I   H   H        
    I   H   H   O   
    I   H   H   O   // 3 uscite reali
    I   H   H   O
    I   
*/

var x_train, y_train, x_test, y_test;

const modelName = "ssdmodel"
const modelPath = "localstorage://" + modelName;
const inputs = 9;
const outputs = 3;

loadCSV();

function setupNewModel() {

  console.log("Creating new model");
  const model = tf.sequential();

  //config for layer
  const config_hidden_1 = {
    inputShape: [inputs],
    activation: "relu",
    units: 4
  }

  const config_hidden_2 = {
    units: 4,
    activation: "relu"
  }

  const config_output = {
    units: outputs,
    activation: "softmax"
  }

  //defining the hidden and output layer
  const hidden1 = tf.layers.dense(config_hidden_1);
  const hidden2 = tf.layers.dense(config_hidden_2);
  const output = tf.layers.dense(config_output);

  //adding layers to model
  model.add(hidden1);
  model.add(hidden2);
  model.add(output);

  compileAndPredict(model);
}

function trainOnExistingModel() {
  console.log("Trying to load saved model");
  reload().then(m => {
    compileAndPredict(m);
  });
}

function compileAndPredict(model) {
  //define an optimizer
  const LEARNING_RATE = 0.01;
  const optimize = tf.train.adam(LEARNING_RATE);

  //config for model
  const config = {
    optimizer: optimize,
    loss: "meanSquaredError"
  }

  //compiling the model
  model.compile(config);
  console.log("Model Successfully Compiled");
  train_data(model).then(model => {
    console.log("Training is Complete");
    console.log("Predictions :");
    let prediction = model.predict(x_test);
    let predictions = Array.from(prediction.dataSync());
    let ytest_array = Array.from(y_test.dataSync());
    var i;
    for(i = 0; i < predictions.length; i++) {
      let pred = predictions[i];
      let expected = ytest_array[i];
      var no = "";
      if(pred != expected) {
        no = "<!> "
      }
      console.log(no + "Expected: " + expected + ", predicted: " + pred);
    }
    save(model)
  })
}

async function train_data(model) {
  for (let i = 0; i < 10; i++) {
    const res = await model.fit(x_train, y_train, epoch = 1000, batch_size = 10);
    console.log(res.history.loss[0]);
  }
  return model;
}

async function save(model) {
  const save = await model.save(modelPath);
  //console.log("Result of saving is  " + JSON.stringify(save));
}

async function reload() {
  const savedModel = await tf.loadLayersModel(modelPath);
  //console.log("Content of model is: " + JSON.stringify(savedModel));
  return savedModel;
}

function clearStorage() {
  localStorage.clear();
  location.reload();
}

function loadCSV() {
  loadLocalCSV("train").then(trainResult => {
    x_train = tf.tensor(trainResult[0]);
    y_train = tf.tensor(trainResult[1]);
    loadLocalCSV("test").then(testResult => {
      x_test = tf.tensor(testResult[0]);
      y_test = tf.tensor(testResult[1]);
      if (localStorage.getItem("tensorflowjs_models/" + modelName + "/info") != null) {
        trainOnExistingModel();
      } else {
        setupNewModel();
      }
    });
  });
}

function loadLocalCSV(identifier) {
  return new Promise((resolve, reject) => {
    var file, fr;
    file = "https://raw.githubusercontent.com/nicorsm/ssd/new-data/dataset/" + identifier + ".csv";
    var xhr = new XMLHttpRequest();
    xhr.open("GET", file, true);
    xhr.responseType = "blob";
    xhr.onload = function (e) {
      if (this.status == 200) {
        const fileObject = new File([this.response], "temp");
        fr = new FileReader();
        fr.onload = function (e) {
          let lines = fr.result.split("\n");
          var inputLines = [];
          var outputLines = [];

          lines.forEach(x => {
            let values = x.split(",");
            inputLines.push(values.slice(0, inputs).map(str => {
              return parseFloat(str);
            }));
            outputLines.push(values.slice(inputs, inputs+outputs).map(str => {
              return parseFloat(str);
            }));
          });
          resolve([inputLines, outputLines]);
        };
        fr.onerror = reject;
        fr.readAsText(fileObject);
      }
    };
    xhr.onerror = reject;
    xhr.send();
  });

}



// Inspired by original code at https://medium.freecodecamp.org/get-to-know-tensorflow-js-in-7-minutes-afcd0dfd3d2f