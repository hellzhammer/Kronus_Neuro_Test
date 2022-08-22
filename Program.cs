using Kronus_Neural.Activations;
using Kronus_Neural.MultiLayerPerceptron;
using KronusML.Numerics.ML;
using KronusML.Numerics.ML.Regression;
using KronusML.Numerics.Models;
using KronusML.BagOfWords;
using System.IO;

public class Program
{
    public static void Main(string[] args)
    {
        /*
        
        various network tests to ensure that neural net can find the best solution for varying problems. 
         
        */

        //Run_XOR_Test();

        //Run_IRIS_Test();

        Run_TEXT_Test();
    }

    public static void Run_TEXT_Test()
    {
        //SentenceTypeData.tsv

        string[] ex_data = File.ReadAllLines(Environment.CurrentDirectory + "/SentenceTypeData.tsv");
        List<double[]> expected_outputs = new List<double[]>();
        List<string> training_data = new List<string>();
        for (int i = 1; i < ex_data.Length; i++) // skip first line
        {
            string[] split = ex_data[i].Split('\t');

            // add training data
            training_data.Add(split[1]);

            // add to outputs
            if (split[2] == "statement")
            {
                expected_outputs.Add(new double[] { 1, 0, 0 });
            }
            else if (split[2] == "exclamation")
            {
                expected_outputs.Add(new double[] { 0, 1, 0 });
            }
            else if (split[2] == "command")
            {
                expected_outputs.Add(new double[] { 0, 0, 1 });
            }
        }

        BOW bow = new BOW(training_data.ToArray(), ' ');
        bow.initBow(new string[] { " " });

        MLP mlp = new MLP(bow.BowMatrix[0].Length, 3, Initializer.xavier, new IActivation<double>[] { new Swish() }, new Sigmoid());
        mlp.Learn(0.001, 25000, bow.BowMatrix, expected_outputs.ToArray());

        mlp.FeedForward(bow.NewSample("go over there please")); // command
        var out1 = mlp.GetNetworkOutputs();

        mlp.FeedForward(bow.NewSample("this is really cool")); // exclamation
        var out2 = mlp.GetNetworkOutputs();

        mlp.FeedForward(bow.NewSample("that is a cat")); //statement
        var out3 = mlp.GetNetworkOutputs();
    }

    public static void Run_XOR_Test()
    {
        double[,] trainingInputs = new double[,] { { 0, 0, 1 }, { 1, 1, 1 }, { 1, 0, 1 }, { 0, 1, 1 } };
        double[,] trainingInputs_nobias = new double[,] { { 0, 0 }, { 1, 1 }, { 1, 0 }, { 0, 1 } };
        double[,] trainingOutputs = new double[,] { { 0 }, { 1 }, { 1 }, { 0 } };
        double[,] double_trainingOutputs = new double[,] { { 1, 0 }, { 0, 1 }, { 0, 1 }, { 1, 0 } };

        Console.WriteLine("Running Neural Networks");
        Console.WriteLine("Running XOR Network Test");
        Console.WriteLine("\n");

        // Enter neural network
        MLP mlp = new MLP(2, 1, Initializer.xavier, new IActivation<double>[] { new Swish() }, new Sigmoid());
        mlp.Learn(0.001, 25000, KronusML.Numerics.Math.ToJagged(trainingInputs_nobias), KronusML.Numerics.Math.ToJagged(trainingOutputs));

        mlp.FeedForward(new double[] { 0, 0 });
        var out1 = mlp.GetNetworkOutputs();

        mlp.FeedForward(new double[] { 1, 1 });
        var out2 = mlp.GetNetworkOutputs();

        mlp.FeedForward(new double[] { 1, 0 });
        var out3 = mlp.GetNetworkOutputs();

        mlp.FeedForward(new double[] { 0, 1 });
        var out4 = mlp.GetNetworkOutputs();
    }

    public static void Run_IRIS_Test()
    {
        string[] ex_data = File.ReadAllLines(Environment.CurrentDirectory + "/Iris.csv");
        List<string[]> split_ex_data = new List<string[]>();
        for (int i = 0; i < ex_data.Length; i++)
        {
            split_ex_data.Add(ex_data[i].Split(','));
        }

        // extract the expected outputs of the network
        double[][] expected_labels = new double[ex_data.Length][];
        for (int i = 0; i < split_ex_data.Count; i++)
        {
            if ("Iris-setosa" == split_ex_data[i][5])
            {
                expected_labels[i] = new double[] { 1, 0, 0 };
            }
            else if ("Iris-versicolor" == split_ex_data[i][5])
            {
                expected_labels[i] = new double[] { 0, 1, 0 };
            }
            else if ("Iris-virginica" == split_ex_data[i][5])
            {
                expected_labels[i] = new double[] { 0, 0, 1 };
            }
        }

        // extract the actual usable inputs of the network
        int[] cols_ignore = new int[] { 0, 5 };
        double[][] iris_training_inputs = new double[expected_labels.Length][];
        for (int x = 0; x < split_ex_data.Count; x++)
        {
            List<double> iris_vals_to_add = new List<double>();
            for (int y = 0; y < split_ex_data[x].Length; y++)
            {
                bool can_add = true;
                for (int i = 0; i < cols_ignore.Length; i++)
                {
                    if (cols_ignore[i] == y)
                    {
                        can_add = false;
                    }
                }

                if (can_add)
                {
                    iris_vals_to_add.Add(double.Parse(split_ex_data[x][y]));
                }
            }
            //iris_vals_to_add.Add(1.0);
            iris_training_inputs[x] = iris_vals_to_add.ToArray();
        }

        // prepare neural network for testing
        MLP mlp1 = new MLP(4 /* if adding bias == 5*/, 3, Initializer.zeros, new IActivation<double>[] { new Sigmoid() }, new Sigmoid());
        mlp1.Learn(0.001, 150000, iris_training_inputs, expected_labels);

        // test the actual outputs.....                         1.0 == extra bias
        mlp1.FeedForward(new double[] { 7.6, 3.0, 6.6, 2.1 }); //, 1.0 }); // virginica = 001
        var out11 = mlp1.GetNetworkOutputs();

        mlp1.FeedForward(new double[] { 7.0, 3.2, 4.7, 1.4 }); //, 1.0 }); // versi-color = 010
        var out31 = mlp1.GetNetworkOutputs();

        mlp1.FeedForward(new double[] { 4.8, 3.4, 1.9, 0.2 }); //, 1.0 }); // setosa = 100
        var out41 = mlp1.GetNetworkOutputs();
    }
}
