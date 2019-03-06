using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using System.Data;
using Accord.IO;
using Accord.Math;
using Accord.Neuro;
using Accord.Neuro.Learning;
using System.IO;
using System.Globalization;


namespace accordnet
{
    class Program
    {
        static void Main(string[] args)
        {
            while(true)
            {
                Console.WriteLine("Press ENTER to compute values from the current model.");
                Console.WriteLine("Press C to clear the current model.");
                string ln = Console.ReadLine();
                if(ln.ToLower() == "c")
                {
                    File.Delete(modelPath);
                } else
                {
                    compute();
                }
            }
        }


        private const string modelPath = "mlp.accord";

        private static double[][] LoadCSV(String name)
        {
            String completePath = @"../../../../dataset/" + name + ".csv";
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

        private static void compute()
        {
            double[][] xTrain = LoadCSV("xtrain");
            double[][] yTrain = LoadCSV("ytrain");
            double[][] xTest = LoadCSV("xtest");

            IActivationFunction function = new SigmoidFunction();

            // In our problem, we have 2 inputs (x, y pairs), and we will 
            // be creating a network with 5 hidden neurons and 1 output:
            //
            ActivationNetwork network;

            if (File.Exists(modelPath))
            {
                network = Serializer.Load<ActivationNetwork>(modelPath);
            }
            else
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
            
            for(int i = 0; i < answers.Length; i++)
            {
                Console.WriteLine(string.Join(", ", answers[i]));
            }

            Serializer.Save(network, modelPath);
        }
    }
}
