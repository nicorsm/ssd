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
using System.Globalization;

namespace AccordNet
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private const string modelPath = "mlp.accord";

        private double[][] LoadCSV(String name)
        {
            String completePath = @"\\VBOXSVR\Downloads\ssd\dataset\" + name + ".csv";
            var reader = new StreamReader(completePath);

            List<double[]> fileContent = new List<double[]>();
            while (!reader.EndOfStream)
            {
                var line = reader.ReadLine();
                var values = line.Split(';').Select(s => double.Parse(s, CultureInfo.InvariantCulture)).ToArray();
                fileContent.Add(values);
            }

            return fileContent.ToArray();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            double[][] xTrain = this.LoadCSV("xtrain");
            double[][] yTrain = this.LoadCSV("ytrain");
            double[][] xTest = this.LoadCSV("xtest");

            IActivationFunction function = new SigmoidFunction();

            // In our problem, we have 2 inputs (x, y pairs), and we will 
            // be creating a network with 5 hidden neurons and 1 output:
            //
            ActivationNetwork network;
            
            if(File.Exists(modelPath))
            {
                network = Serializer.Load<ActivationNetwork>(modelPath);
            } else
            {
                network = new ActivationNetwork(
                    function, 
                    inputsCount: 6, 
                    neuronsCount: new[] { 4, 4, 3 }
                    );
            }

            LevenbergMarquardtLearning teacher = new LevenbergMarquardtLearning(network, true);

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
            double[][] answers = xTrain.Apply(network.Compute);
        
            Console.WriteLine(answers);
            
            Serializer.Save(network, modelPath);
            
        }
    }
}


// Original code here: https://github.com/accord-net/framework/wiki/Classification