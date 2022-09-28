using Kronus_Neural.NEAT;
using Kronus_Neural.NEAT.Kronus_Neat;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Kronus_Console
{
    public class NEAT_IRIS : Kronus_NEAT
    {
        public double[][] input_data { get; set; }
        public double[][] output_data { get; set; }

        public NeatNetwork winning_network { get; set; }

        public NEAT_IRIS(
            int input_count,
            int output_count,
            bool initFullyConnected,
            double chanceToConnect,
            int pop_max,
            int training_epochs,
            int species_target)
        {
            this.nets = new Dictionary<string, NeatNetwork>();
            this.species = new Dictionary<string, Species>();
            this.gene_tracker = new Genetic_Dictionary();

            this.chance_to_make_inital_connection = chanceToConnect;
            this.input_neuron_count = input_count;
            this.output_neuron_count = output_count;
            this.init_fully_connected = initFullyConnected;
            this.PopulationMax = pop_max;
            this.total_epochs = training_epochs;

            this.totalSpeciesCountTarget = species_target;

            this.init_project();
        }

        public void NEAT_Train()
        {
            for (epoch = 1; epoch < total_epochs; epoch++)
            {
                this.FitnessTest();

                this.Train();

                // catalogue all connections
                foreach (var net in nets)
                {
                    foreach (var connection in net.Value.All_Connections)
                    {
                        if (!this.gene_tracker.Connection_Exists(connection.Key))
                        {
                            this.gene_tracker.Add_Connection(connection.Key, epoch);
                        }
                    }
                    foreach (var node in net.Value.Hidden_Neurons)
                    {
                        if (!this.gene_tracker.Neuron_Exists(node.Key))
                        {
                            this.gene_tracker.Add_Node(node.Key, epoch);
                        }
                    }
                }
            }
        }

        private void FitnessTest()
        {
            // convert to jagged arrays
            var data = this.input_data;
            var outputData = this.output_data;

            string fittest = string.Empty;

            // iter through all nets
            foreach (var net in nets)
            {
                double fitness = 0;
                //double loss = 0;
                // assess fitness of each net
                for (int i = 0; i < data.Length; i++)
                {
                    var output = net.Value.Feed_Forward(data[i]);
                    bool output_match = this.Arrays_Match(output, outputData[i]);
                    if (output_match)
                    {
                        fitness++;
                    }
                }

                //net.Value.current_error = 1 - loss;
                net.Value.current_fitness = fitness / input_data.Length;
                if (fittest == string.Empty)
                {
                    fittest = net.Key;
                }
                else
                {
                    if (fitness > nets[fittest].current_fitness)
                    {
                        fittest = net.Key;
                    }
                }
            }

            if (winning_network != null)
            {
                double new_fitness = 0;
                //double new_loss = 0;
                for (int t = 0; t < data.Length; t++)
                {
                    var output = winning_network.Feed_Forward(data[t]);
                    bool output_match = this.Arrays_Match(output, outputData[t]);
                    if (output_match)
                    {
                        new_fitness++;
                    }
                }

                new_fitness /= input_data.Length;

                if (new_fitness < nets[fittest].current_fitness)
                {
                    winning_network = nets[fittest];
                }
            }
            else if (winning_network == null)
            {
                winning_network = nets[fittest];
            }
        }

        bool Arrays_Match(double[] actual_output, double[] expected_output)
        {
            bool rtnval = true;
            if (actual_output.Length != expected_output.Length)
            {
                throw new Exception("Arrays do not match.");
            }

            for (int i = 0; i < actual_output.Length; i++)
            {
                if (actual_output[i] != expected_output[i])
                {
                    if (expected_output[i] == 1)
                    {
                        if (actual_output[i] < 0.65 || actual_output[i] > 1.0)
                        {
                            rtnval = false;
                            break;
                        }
                    }
                    else if (expected_output[i] == 0)
                    {
                        if (actual_output[i] > 0.35 || actual_output[i] < 0.0)
                        {
                            rtnval = false;
                            break;
                        }
                    }
                }
            }
            return rtnval;
        }

        private void init_project()
        {
            this.init_data();
            for (int i = 0; i < this.PopulationMax; i++)
            {
                var net = Network_Generator.Generate_New_Network_Neurons(this.input_neuron_count, this.output_neuron_count);
                if (this.init_fully_connected)
                {
                    net = Network_Generator.Init_Connections_fully_connected(net);
                }
                else
                {
                    net = Network_Generator.Init_Connections_random_connections(net, this.chance_to_make_inital_connection);
                }
                nets.Add(net.network_id, net);
            }
        }

        private void init_data()
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
                iris_vals_to_add.Add(1.0);
                iris_training_inputs[x] = iris_vals_to_add.ToArray();
            }
            this.input_data = iris_training_inputs;
            this.output_data = expected_labels;
        }
    }
}
