using Kronus_Neural.Activations;
using Kronus_Neural.MultiLayerPerceptron;
using KronusML.Numerics.ML;
using KronusML.Numerics.ML.Regression;
using KronusML.Numerics.Models;
using KronusML.BagOfWords;
using System.IO;
using Kronus_Neural.Loss_Functions;
using Newtonsoft.Json;
using Kronus_Neural.NEAT;
using Kronus_Neural.NEAT.Kronus_Neat;

public class Program
{
    public static void Main(string[] args)
    {

    }

    public static void IRIS_NEAT()
    {
        Console.WriteLine("Creating IRIS Project");
        Kronus_Console.NEAT_IRIS neat = new Kronus_Console.NEAT_IRIS(5, 3, true, 0.60, 200, 5000, 15);
        neat.max_hidden_nodes = 50;

        Console.WriteLine("Training Networks, This could take a while.");
        neat.NEAT_Train();


        foreach (var item in neat.gene_tracker.Known_Connection_IDs)
        {
            Console.WriteLine("Gene: " + item.Key + ", Generation: " + item.Value.ToString());
        }

        if (neat.winning_network != null)
        {
            Console.WriteLine("Running IRIS test on fittest Network");
            Console.WriteLine(neat.winning_network.network_id);
            Console.WriteLine(neat.winning_network.species_id);
            Console.WriteLine("fitness: " + neat.winning_network.current_fitness);
        }
        Console.WriteLine();
    }

    public static void XOR_NEAT()
    {
        Console.WriteLine("Creating XOR Project");
        Kronus_Console.NEAT neat = new Kronus_Console.NEAT(3, 1, false, 0.60, 200, 2500);
        neat.max_hidden_nodes = 50;

        Console.WriteLine("Training Networks, This could take a while.");
        neat.NEAT_Train();


        foreach (var item in neat.gene_tracker.Known_Connection_IDs)
        {
            Console.WriteLine("Gene: " + item.Key + ", Generation: " + item.Value.ToString());
        }

        if (neat.winning_network != null)
        {
            Console.WriteLine("Running XOR test on fittest Network");
            Console.WriteLine(neat.winning_network.network_id);
            Console.WriteLine(neat.winning_network.species_id);
            Console.WriteLine("fitness: " + neat.winning_network.current_fitness);
            Console.WriteLine("error: " + neat.winning_network.current_error);
            for (int i = 0; i < 10; i++)
            {
                Console.WriteLine("** Test: " + i.ToString() + " **");
                var out1 = neat.winning_network.Feed_Forward(new double[] { 0, 0, 1 });
                var out2 = neat.winning_network.Feed_Forward(new double[] { 0, 1, 1 });

                var out3 = neat.winning_network.Feed_Forward(new double[] { 1, 1, 1 });
                var out4 = neat.winning_network.Feed_Forward(new double[] { 1, 0, 1 });

                Console.WriteLine("Input 001: " + out1[0]);
                Console.WriteLine("Input 011: " + out2[0]);
                Console.WriteLine("Input 111: " + out3[0]);
                Console.WriteLine("Input 101: " + out4[0]);
            }

        }
        Console.WriteLine();
    }


    static BOW KronusWordDict { get; set; }
    static Dictionary<string, (MLP mlp, BOW bow)> ContextClassifiers = new Dictionary<string, (MLP mlp, BOW bow)>();


    public static void Main1(string[] args)
    {
        NeatNetwork net = new NeatNetwork();
        net = Network_Generator.Generate_New_Network_Neurons(4, 3);
        net = Network_Generator.Init_Connections_random_connections(net, 0.5);
        var st1 = net.Get_Structure();
        for (int i = 0; i < 5; i++)
        {
            //net = Mutation.Add_Node(net, 7 + i);
        }
        var st2 = net.Get_Structure();
        //net = Mutation.Remove_Connection(net);
        //net = Mutation.Add_Connection(net);
        //net = Mutation.Remove_Node(net);
        var st3 = net.Get_Structure();


        NeatNetwork n2 = new NeatNetwork();
        n2 = Network_Generator.Generate_New_Network_Neurons(4, 3);
        n2 = Network_Generator.Init_Connections_random_connections(n2, 0.60);

        var outp = Gene_Sequencer.get_sequence(net, n2);
        Console.WriteLine();
    }

    #region test logic
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

        MLP mlp = new MLP(bow.BowMatrix[0].Length, 3, new int[] { 4, 3 }, new IActivation<double>[] { new Gaussian() }, new Sigmoid(), Initializer.zeros);
        // mlp.Learn(0.01, 7500, bow.BowMatrix, expected_outputs.ToArray());

        mlp.FeedForward(bow.NewSample("go over there please")); // command
        var out1 = mlp.GetNetworkOutputs();

        mlp.FeedForward(bow.NewSample("i want to go shopping")); // exclamation
        var out2 = mlp.GetNetworkOutputs();

        mlp.FeedForward(bow.NewSample("he did this last night")); //statement
        var out3 = mlp.GetNetworkOutputs();
    }

    public static void Run_XOR_Test()
    {
        //double[,] trainingInputs = new double[,] { { 0, 0, 1 }, { 1, 1, 1 }, { 1, 0, 1 }, { 0, 1, 1 } };
        double[,] trainingInputs_nobias = new double[,] { { 0, 0 }, { 1, 1 }, { 1, 0 }, { 0, 1 } };
        double[,] trainingOutputs = new double[,] { { 0 }, { 1 }, { 1 }, { 0 } };
        //double[,] double_trainingOutputs = new double[,] { { 1, 0 }, { 0, 1 }, { 0, 1 }, { 1, 0 } };

        // test new construcable neural network
        MLP mlp2 = new MLP(2, 1, new int[] { 3, 3, 4, 2 }, new IActivation<double>[] { new TanH(), new TanH(), new TanH(), new TanH() }, new Sigmoid(), Initializer.xavier);
        mlp2.Learn(0.05, 5000, KronusML.Numerics.Math.ToJagged(trainingInputs_nobias), KronusML.Numerics.Math.ToJagged(trainingOutputs), LossFunctions.quadratic_Loss);


        mlp2.FeedForward(new double[] { 0, 0 });
        var out1 = mlp2.GetNetworkOutputs();
        string o1 = "0,1 = ";
        for (int i = 0; i < out1.Length; i++)
        {
            o1 += out1[i].ToString();
        }
        Console.WriteLine(o1);


        mlp2.FeedForward(new double[] { 1, 1 });
        var out2 = mlp2.GetNetworkOutputs();
        string o2 = "0,1 = ";
        for (int i = 0; i < out2.Length; i++)
        {
            o2 += out2[i].ToString();
        }
        Console.WriteLine(o2);


        mlp2.FeedForward(new double[] { 1, 0 });
        var out3 = mlp2.GetNetworkOutputs();
        string o3 = "1,0 = ";
        for (int i = 0; i < out3.Length; i++)
        {
            o3 += out3[i].ToString();
        }
        Console.WriteLine(o3);


        mlp2.FeedForward(new double[] { 0, 1 });
        var out4 = mlp2.GetNetworkOutputs();
        string o4 = "0,1 = ";
        for (int i = 0; i < out4.Length; i++)
        {
            o4 += out4[i].ToString();
        }
        Console.WriteLine(o4);
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
        MLP mlp1 = new MLP(4 /* if adding bias == 5*/, 3, new int[] { 8 }, new IActivation<double>[] { new Gaussian() }, new Sigmoid(), Initializer.zeros);
        // mlp1.Learn(0.01, 100, iris_training_inputs, expected_labels);

        // test the actual outputs.....                         1.0 == extra bias
        mlp1.FeedForward(new double[] { 7.8, 2.8, 6.8, 2.4 }); //, 1.0 }); // virginica = 001
        var out11 = mlp1.GetNetworkOutputs();

        mlp1.FeedForward(new double[] { 7.0, 3.2, 4.7, 1.4 }); //, 1.0 }); // versi-color = 010
        var out31 = mlp1.GetNetworkOutputs();

        mlp1.FeedForward(new double[] { 4.8, 3.4, 1.9, 0.2 }); //, 1.0 }); // setosa = 100
        var out41 = mlp1.GetNetworkOutputs();

        mlp1.FeedForward(new double[] { 5.0, 3.5, 2.1, 0.4 }); //, 1.0 }); // setosa = 100
        var out411 = mlp1.GetNetworkOutputs();
    }
    #endregion
}
