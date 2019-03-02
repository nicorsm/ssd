/*
    // 6 ingressi reali 
    I       // 2 strati nascosti completamente connessi        
    I   H   H        
    I   H   H   O   
    I   H   H   O   // 3 uscite reali
    I   H   H   O
    I   
*/


//Dummy training data
var x_train = tf.tensor([
  [0.1, 0.5, 0.1, 0.1, 0.5, 0.1],
  [0.9, 0.3, 0.4, 0.1, 0.5, 0.1],
  [0.4, 0.5, 0.5, 0.1, 0.5, 0.1],
  [0.7, 0.1, 0.9, 0.1, 0.5, 0.1]
])

//Dummy training labels
var y_train = tf.tensor([
  [0.2, 0.8, 0.1],
  [0.9, 0.1, 0.1],
  [0.4, 0.6, 0.1],
  [0.5, 0.5, 0.1]
])

//Dummy testing data
var x_test = tf.tensor([
  //[0.9, 0.1, 0.5, 0.1, 0.5, 0.1]
])

const modelName = "ssdmodel"
const modelPath = "localstorage://" + modelName;

if (localStorage.getItem("tensorflowjs_models/" + modelName + "/info") != null) {
  trainOnExistingModel();
} else {
  //setupNewModel();
  loadCSV();
}

function setupNewModel() {

  console.log("Creating new model");
  const model = tf.sequential();

  //config for layer
  const config_hidden_1 = {
    inputShape: [6],
    activation: "sigmoid",
    units: 4
  }
  
  const config_hidden_2 = {
    units: 4,
    activation: "sigmoid"
  }

  const config_output = {
    units: 3,
    activation: "sigmoid"
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
  const optimize = tf.train.sgd(0.1);

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
    model.predict(x_test).print();
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
  loadLocalCSV("xtrain").then(xtrain => {
    x_train = tf.tensor(xtrain);
    loadLocalCSV("ytrain").then(ytrain => {
      y_train = tf.tensor(ytrain);
      loadLocalCSV("xtest").then(xtest => {
        x_test = tf.tensor(xtest);
        setupNewModel();
      });
    });
  });
}

function loadLocalCSV(identifier) {
  return new Promise((resolve, reject) => {
    var input, file, fr;
/*
    if (typeof window.FileReader !== 'function') {
       alert("The file API isn't supported on this browser yet.");
       return;
    }
    input = "../dataset/" + identifier + ".csv"; // document.getElementById(identifier);
    if (!input) {
       alert("Couldn't find the fileinput element.");
    } else if (!input.files) {
       alert("This browser doesn't seem to support the `files` property of file inputs.");
    } else if (!input.files[0]) {
       alert("Please select a file before clicking 'Load'");
    } else {
       file = input.files[0];*/

       file = "../dataset/" + identifier + ".csv";
       var xhr = new XMLHttpRequest();
       xhr.open("GET", file, true);
       xhr.responseType = "blob";
       xhr.onload = function(e) {
        if(this.status == 200) {
          const fileObject = new File([this.response], "temp");
          fr = new FileReader();
          fr.onload = function(e) {
            let lines = fr.result.split("\n");
            var array = lines.map(x => {
              return x.split(";").map(str => {
                return parseFloat(str);
              });
            });
            console.log("Array: " + array);
            resolve(array);
          };
          fr.onerror = reject;
          fr.readAsDataURL(fileObject);
        }
       };
       xhr.onerror = reject;
       xhr.send();





       //fr.readAsText(file);
    //}
  });

}



// Inspired by original code at https://medium.freecodecamp.org/get-to-know-tensorflow-js-in-7-minutes-afcd0dfd3d2f