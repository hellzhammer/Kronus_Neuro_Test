using Kronus_Neural.NEAT;
using Kronus_Neural.NEAT.Kronus_Neat;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Kronus_Console
{
    public class NEAT : Kronus_NEAT
    {
        double[,] trainingInputs_with_bias = new double[,] { { 0, 0, 1 }, { 1, 1, 1 }, { 1, 0, 1 }, { 0, 1, 1 } };
        double[,] trainingOutputs = new double[,] { { 0 }, { 1 }, { 1 }, { 0 } };

        public NeatNetwork winning_network { get; set; }

        public NEAT(
            int input_count,
            int output_count,
            bool initFullyConnected,
            double chanceToConnect,
            int pop_max,
            int training_epochs)
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

            this.totalSpeciesCountTarget = 10;

            this.init_project();
        }

        private void init_project()
        {
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

        private void FitnessTest()
        {
            // convert to jagged arrays
            var data = KronusML.Numerics.Math.ToJagged(this.trainingInputs_with_bias);
            var outputData = KronusML.Numerics.Math.ToJagged(this.trainingOutputs);

            string fittest = string.Empty;

            // iter through all nets
            foreach (var net in nets)
            {
                double fitness = 0;
                double loss = 0;
                // assess fitness of each net
                for (int i = 0; i < data.Length; i++)
                {                    
                    var output = net.Value.Feed_Forward(data[i]);
                    for (int y = 0; y < output.Length; y++)
                    {
                        if (output[y] >= 0.80 && output[y] <= 1.0)
                        {
                            if (outputData[i][y] == 1)
                            {
                                loss += 1 - output[y];
                                fitness += output[y];
                            }
                        }
                        else if(output[y] <= 0.20 && output[y] >= 0.0)
                        {
                            if (outputData[i][y] == 0)
                            {
                                loss += output[y];
                                fitness += 1 - output[y];
                            }
                        }
                    }
                }

                net.Value.current_error = 1 - loss;
                net.Value.current_fitness = fitness;
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
                double new_loss = 0;
                for (int t = 0; t < data.Length; t++)
                {
                    var output = winning_network.Feed_Forward(data[t]);
                    for (int y = 0; y < output.Length; y++)
                    {
                        if (output[y] >= 0.80 && output[y] <= 1.0)
                        {
                            if (outputData[t][y] == 1)
                            {
                                new_loss += 1 - output[y];
                                new_fitness += output[y];
                            }
                        }
                        else if (output[y] <= 0.20 && output[y] >= 0.0)
                        {
                            if (outputData[t][y] == 0)
                            {
                                new_loss += output[y];
                                new_fitness += 1 - output[y];
                            }
                        }
                    }
                }


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
    }
}
