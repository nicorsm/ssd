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

        private class Result
        {
            public List<double[]> inputs = new List<double[]>();
            public List<int> outputs = new List<int>();
        }


        private const string modelPath = "mlp.accord";

        private static Result LoadCSV(String name)
        {
            String completePath = @"../../../../dataset/" + name + ".csv";
            var reader = new StreamReader(completePath);

            Result res = new Result();

            while (!reader.EndOfStream)
            {
                //Input type: %f,%f,%f,%f,%f,%f,%d (%f = inputs, %d = expected output)
                var line = reader.ReadLine();
                var values = line.Split(',');
                res.inputs.Add(values.Take(6).Select(s => double.Parse(s, CultureInfo.InvariantCulture)).ToArray());
                res.outputs.Add(values.Skip(6).Take(1).Select(s => int.Parse(s, CultureInfo.InvariantCulture)).First());
            }

            return res;
        }

        private static void compute()
        {
            Result train = LoadCSV("train");
            Result test = LoadCSV("test");

            IActivationFunction function = new SigmoidFunction();

            // In our problem, we have 2 inputs (x, y pairs), and we will 
            // be creating a network with 5 hidden neurons and 1 output:
            //
            ActivationNetwork network;
            int nOutputs = 3;

            if (File.Exists(modelPath))
            {
                network = Serializer.Load<ActivationNetwork>(modelPath);
            }
            else
            {
                network = new ActivationNetwork(
                    function,
                    inputsCount: 6,
                    neuronsCount: new[] { 4, 4, nOutputs }
                    );
            }

            // Iterate until stop criteria is met
            double previous;

            double[][] jaggedOutputs = Jagged.OneHot(train.outputs.ToArray());

            // Heuristically randomize the network
            new NguyenWidrow(network).Randomize();

            // Create the learning algorithm
            var teacher = new LevenbergMarquardtLearning(network);
            
            // Teach the network for 10 iterations:
            double error = Double.PositiveInfinity;

            do
            {
                previous = error;
                error = teacher.RunEpoch(train.inputs.ToArray(), jaggedOutputs);
            } while (Math.Abs(previous - error) < 1e-10 * previous);


            // At this point, the network should be able to 
            // perfectly classify the training input points.

            double[][] inputs = test.inputs.ToArray();
            int[] outputs = test.outputs.ToArray();

            for (int i = 0; i < inputs.Length; i++)
            {
                int answer;
                double[] input = inputs[i];
                double[] output = network.Compute(input);
                double response = output.Max(out answer);

                int expected = outputs[i];

                // at this point, the variables 'answer' and
                // 'expected' should contain the same value.
                Console.WriteLine(string.Join(",", input));
                Console.WriteLine("Expected: " + expected + ", answer: " + answer);
            }

            Serializer.Save(network, modelPath);
        }
    }
}
