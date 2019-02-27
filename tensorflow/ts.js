// 6 ingressi reali
// 3 uscite reali
// 2 strati nascosti completamente connessi


/*
    I   
    I   H   H
    I   H   H   O
    I   H   H   O
    I   H   H   O
    I   

*/

const model = tf.sequential();

model.add(
  tf.layers.dense({inputShape:[6], units:3, activation: "softmax"}),
  tf.layers.dense({units:4}),
  tf.layers.dense({units:4}),
  tf.layers.dense({units:3})
);

model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

const xs = tf.tensor2d([[1], [2], [3], [4], [5], [6]], [6, 1]);
const ys = tf.tensor2d([[1], [3], [5], [7], [9], [11]], [6, 1]);

model.fit(xs, ys, {epochs: 1000});

model.predict(tf.tensor2d([5], [1,1])).print();