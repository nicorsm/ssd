using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Accord.Controls;
using Accord.IO;
using Accord.Math;
using Accord.Statistics.Distributions.Univariate;
using Accord.MachineLearning.Bayes;
using Accord.Neuro;
using Accord.Neuro.Learning;
using System.IO;

namespace AccordNet
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private double[][] LoadCSV(String name)
        {
            String completePath = @"\\VBOXSVR\Downloads\ssd\dataset\" + name + ".csv";
            var reader = new StreamReader(completePath);

            List<List<double>> fileContent = new List<List<double>>();
            while (!reader.EndOfStream)
            {
                var line = reader.ReadLine();
                var values = line.Split(';').Select(Double.Parse).ToList();
                fileContent.Add(values);
            }

            return fileContent.Select(x => x.ToArray()).ToArray();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            double[][] xTrain = this.LoadCSV("xtrain");
            double[][] yTrain = this.LoadCSV("ytrain");
            double[][] xTest = this.LoadCSV("xtest");

            // Convert the DataTable to input and output vectors
            //double[][] inputs = table.ToJagged<double>("X", "Y");
            //int[] outputs = table.Columns["G"].ToArray<int>();

            // Plot the data
            //ScatterplotBox.Show("Yin-Yang", inputs, outputs).Hold();
          

            // Since we would like to learn binary outputs in the form
            // [-1,+1], we can use a bipolar sigmoid activation function
            IActivationFunction function = new BipolarSigmoidFunction();

            // In our problem, we have 2 inputs (x, y pairs), and we will 
            // be creating a network with 5 hidden neurons and 1 output:
            //
            var network = new ActivationNetwork(function,
                inputsCount: 2, neuronsCount: new[] { 5, 1 });

            // Create a Levenberg-Marquardt algorithm
            var teacher = new LevenbergMarquardtLearning(network)
            {
                UseRegularization = true
            };

            // Because the network is expecting multiple outputs,
            // we have to convert our single variable into arrays
            //
            //var y = outputs.ToDouble().ToArray();

            // Iterate until stop criteria is met
            double error = double.PositiveInfinity;
            double previous;

            do
            {
                previous = error;

                // Compute one learning iteration
                error = teacher.RunEpoch(xTrain, yTrain);

            } while (Math.Abs(previous - error) < 1e-10 * previous);


            // Classify the samples using the model
            int[] answers = xTrain.Apply(network.Compute).GetColumn(0).Apply(System.Math.Sign);

            Console.WriteLine(answers);

            // Plot the results
            /*ScatterplotBox.Show("Expected results", inputs, outputs);
            ScatterplotBox.Show("Network results", inputs, answers).Hold();*/
        }
    }
}
