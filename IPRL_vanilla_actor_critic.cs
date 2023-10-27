using System;
using System.Linq;
using System.Text;
using System.IO;
using System.Windows.Input;
using File = System.IO.File;
using System.Collections.Generic;
using System.Reflection;
using System.Windows.Forms;
//using NumSharp;
//using Tensorflow.Keras.Engine;
//using Tensorflow.Keras.Layers;
//using Tensorflow.Keras.Layers;
using Tensorflow.Keras;
using Tensorflow;
using static Tensorflow.Binding;
//using static Tensorflow.KerasApi;


namespace NeuralNetworkExample
{

    
    /*
    public class Fnn
    {
        Tensorflow.Keras.Engine.IModel model;
        Tensorflow.NumPy.NDArray x_train, y_train, x_test, y_test;
        

        public void PrepareData()
        {
            Tensorflow.Keras.Datasets.Mnist min_data =new Tensorflow.Keras.Datasets.Mnist();

            (x_train, y_train, x_test, y_test) = min_data.load_data();
         
            x_train = x_train.reshape((60000, 784)) / 255f;
            x_test = x_test.reshape((10000, 784)) / 255f;
        }

        public void BuildModel()
        {
            Tensorflow.Keras.IKerasApi keras_dr = new Tensorflow.Keras.KerasInterface();
            
            var inputs = keras_dr.Input(shape: 784);
            var outputs = keras_dr.layers.Dense(64, activation: keras_dr.activations.Relu).Apply(inputs);

            outputs = keras_dr.layers.Dense(10).Apply(outputs);

            model = keras_dr.Model(inputs, outputs, name: "mnist_model");
            model.summary();

            model.compile(loss: keras_dr.losses.SparseCategoricalCrossentropy(from_logits: true),
                optimizer: keras_dr.optimizers.Adam(),
                metrics: new[] { "accuracy" });
        }

        public void Train()
        {
            model.fit(x_train, y_train, batch_size: 10, epochs: 2);
            model.evaluate(x_test, y_test);
        }


        
    }

    */
    public class Actor
    {
        public Tensorflow.Keras.Engine.IModel actor_network;
        public Tensorflow.Keras.IKerasApi keras_api = new Tensorflow.Keras.KerasInterface();
        public Tensorflow.Tensors inputs;
        public Actor(int state_dimension)// constrcutors: special methods used to initialize instances of a class when objects of that class are created 
        {
            inputs = keras_api.Input(shape: state_dimension);
            var hidden_1 = keras_api.layers.Dense(128, activation: keras_api.activations.Relu).Apply(inputs);
            var hidden_2 = keras_api.layers.Dense(256, activation: keras_api.activations.Relu).Apply(hidden_1);
            var outputs = keras_api.layers.Dense(3, activation: keras_api.activations.Softmax).Apply(hidden_2);  //output the aciton probability
            actor_network = keras_api.Model(inputs, outputs, name: "Actor Network");
        }
        



    }

    public class Critic
    {
        public Tensorflow.Keras.Engine.IModel critic_network;
        public Tensorflow.Keras.IKerasApi keras_api = new Tensorflow.Keras.KerasInterface();
        public Tensorflow.Tensors inputs;
        public Critic(int state_dimension)// constrcutors: special methods used to initialize instances of a class when objects of that class are created 
        {
            inputs = keras_api.Input(shape: state_dimension);
            var hidden_1 = keras_api.layers.Dense(128, activation: keras_api.activations.Relu).Apply(inputs);
            var hidden_2 = keras_api.layers.Dense(256, activation: keras_api.activations.Relu).Apply(hidden_1);
            var outputs = keras_api.layers.Dense(3).Apply(hidden_2);  //output the aciton probability
            critic_network = keras_api.Model(inputs, outputs, name: "Critic Network");
            
        }

    }

    public class toy_environment
    {
        // create a toy-env to debug the Reinforcement Learning Algrithm
        double[] action_list = new double[] { 1.01, 1.00, 0.99 };
        double[] state = new double[4];
        int done = 0;
        public double[] reset()
        {
            double[] initial_state = new double[] { 92, 104, 14, 18 };
            state = initial_state;
            return initial_state;
        }
        public (double[], double, int)  step(int action_index)
        {
            double[] new_state = new double[state.Length];
            Array.Copy(state, new_state, state.Length);


            if (action_index == 0)
            {
                new_state[0] = new_state[0] * 1.02;
                new_state[1] = new_state[1] * 1.02;
                new_state[2] = new_state[2] * 1.005;
                new_state[3] = new_state[3] * 1.005;
            }
            if (action_index == 1)
            {
                new_state[0] = new_state[0] * 0.99;
                new_state[1] = new_state[1] * 0.99;
                new_state[2] = new_state[2] * 0.96;
                new_state[3] = new_state[3] * 0.99;
            }
            if (action_index == 2)
            {
                new_state[0] = new_state[0] * 0.99;
                new_state[1] = new_state[1] * 0.99;
                new_state[2] = new_state[2] * 0.99;
                new_state[3] = new_state[3] * 0.96;
            }
            double old_cost = cost_calc(state);
            double new_cost = cost_calc(new_state);
            double step_reward = old_cost - new_cost;
            
            
            //Console.WriteLine("Old cost {0} new cost {1}", old_cost.ToString(), new_cost.ToString());

            state = new_state;
            if (new_cost == 0)
            {
                done = 1;
            }


            return (new_state, step_reward,done);


        }

        public double cost_calc(double[] state)
        {
            // constraints
            double[] constraints =new double[] { 100, 110, 20, 20 };
            double cost_0 = Math.Max(constraints[0] - state[0], 0);
            double cost_1 = Math.Max(state[1] - constraints[1], 0);
            double cost_2 = Math.Max(state[2] - constraints[2], 0);
            double cost_3 = Math.Max(state[3] - constraints[3], 0);
            double total_cost = cost_0 + cost_1 + cost_2 + cost_3;
            return total_cost;

        }

    
    }

    public class SF
    { 
        // statistical functions

        public int random_choice(Tensorflow.Tensor probability)
        {
            
            Random random = new Random();
            double randomValue = random.NextDouble(); // Generate a random value between 0 and 1
            double cumulativeProbability = (double)probability.numpy()[0][0];
            //Console.WriteLine("Init cum prob "+ cumulativeProbability.ToString());
            int index = 100;
            for (int i = 0; i < probability.shape[1]; i++)
            {
                
                
                if (randomValue < cumulativeProbability)
                {
                    index = i;
                    break;

                }
                else
                {
                    cumulativeProbability += (double)probability.numpy()[0][i+1];
                    
                }
            }
            //Console.WriteLine("random value {0} index {1}", randomValue.ToString(), index.ToString());
            return index;

        }
    
    }


        class Program
    {
        static void Main(string[] args)
        {

            
            Tensorflow.tensorflow tf_operator = new Tensorflow.tensorflow();
            Tensorflow.Graph tf_graph = new Tensorflow.Graph();


            Tensorflow.Keras.IKerasApi keras_api = new Tensorflow.Keras.KerasInterface();
            Tensorflow.tensorflow.MathApi math_api = new Tensorflow.tensorflow.MathApi();
            
            //Tensorflow.Gradients.GradientTape tape_ac = tf_operator.GradientTape(); //Gradient Tape Set Record operations for automatic differentiation
            //Tensorflow.Gradients.GradientTape tape_cc = tf_operator.GradientTape();
            Tensorflow.Keras.Losses.Huber huber_api = new Tensorflow.Keras.Losses.Huber(reduction: Tensorflow.Keras.Losses.ReductionV2.SUM);
            Tensorflow.Keras.Optimizers.Adam optimizer_actor = new Tensorflow.Keras.Optimizers.Adam(learning_rate: (float)1e-5);
            Tensorflow.Keras.Optimizers.Adam optimizer_critic = new Tensorflow.Keras.Optimizers.Adam(learning_rate: (float)1e-7);


            SF statis_funcs = new SF();
            
            
            double running_reward = 0;
            int episode_count = 0;
            double gamma = 0.99;


            Critic critic = new Critic(4);
            Actor actor = new Actor(4);
            Tensorflow.Keras.Engine.IModel critic_net = critic.critic_network;
            Tensorflow.Keras.Engine.IModel actor_net = actor.actor_network;

            toy_environment env_01 = new toy_environment();
            double[] state1 = env_01.reset();
            Tensorflow.Tensor tf_state1 = tf_operator.convert_to_tensor(state1, Tensorflow.TF_DataType.TF_FLOAT);

            StreamWriter training_writer = new StreamWriter(@"P:\Private\Jackie Wu Lab\Dongrong Yang\Inverse_RL\Projects\IPRL_debug\training_output.txt");
               
            for (int epoch = 0; epoch <= 100000; epoch++)
            {


                // epoch training
                // initiate a new env
                toy_environment env_0 = new toy_environment();
                double[] state = env_0.reset();
                double episode_reward = 0;
                List<Tensor> action_probs_history = new List<Tensor>();
                List<double> rewards_history = new List<double>();
                List<Tensor> critic_value_history = new List<Tensor>();

                using (Tensorflow.Gradients.GradientTape t_a = tf.GradientTape(), t_b = tf.GradientTape())
                {


                    for (int timestep = 0; timestep <= 10; timestep++)
                    {

                        //Console.WriteLine("-------------Episode {0} Step {1}-------------", epoch.ToString(), timestep.ToString());

                        Tensorflow.Tensor tf_state = tf_operator.convert_to_tensor(state, Tensorflow.TF_DataType.TF_FLOAT);
                        tf_state = tf_operator.expand_dims(tf_state, 0);

                        t_a.watch(tf_state);
                        t_b.watch(tf_state);
                        Tensorflow.Tensor action_pros = actor_net.Apply(tf_state, training: true);


                        Tensorflow.Tensor critic_value = critic_net.Apply(tf_state, training: true);





                        int action = statis_funcs.random_choice(action_pros);

                        /*
                        Console.WriteLine("State: {0}", tf_state.ToString());
                        Console.WriteLine("Action Pros " + action_pros.ToString());
                        Console.WriteLine("Critic Value " + critic_value.ToString());
                        */
                        var action_prob = action_pros[0, action];



                        var action_log_prob = tf.log(action_prob);
                        // Note: need to recheck the paper or pseudo
                        critic_value_history.Add(critic_value[0, action]);
                        //Console.WriteLine("Action Q  " + critic_value[0, action].ToString());
                        action_probs_history.Add(action_log_prob);


                        //Console.WriteLine("Action Index " + action.ToString() + "  prob  "+action_prob.ToString()+ "   log prob " + action_log_prob.ToString() + " Q value " + critic_value.numpy()[0][action].ToString()); ;

                        (double[] new_state, double reward, double done) = env_0.step(action);
                        //Console.WriteLine("new state {0} reward {1}", new_state.ToString(), reward.ToString());
                        rewards_history.Add(reward);
                        episode_reward += reward;

                        // update the state
                        state = new_state;

                    }
                    running_reward = 0.05 * episode_reward + (1 - 0.95) * running_reward;
                    Console.WriteLine("Episode {0} running reward {1}", epoch.ToString(), running_reward.ToString());
                    training_writer.WriteLine(episode_reward.ToString()+","+ running_reward.ToString());



                    // Calculate expected value from rewards
                    // - At each timestep what was the total reward received after that timestep
                    // - Rewards in the past are discounted by multiplying them with gamma
                    // - These are the labels for our critic
                    List<Tensorflow.Tensor> return_list = new List<Tensorflow.Tensor>();
                    double discounted_sum = 0;
                    for (int i = rewards_history.Count - 1; i > -1; i--)
                    {
                        //Console.WriteLine(i.ToString() + "   " + rewards_history[i].ToString());
                        discounted_sum = rewards_history[i] + gamma * discounted_sum;
                        Tensorflow.Tensor tensor_dis_sum = tf_operator.convert_to_tensor(discounted_sum, Tensorflow.TF_DataType.TF_FLOAT);
                        return_list.Add(tensor_dis_sum);

                    }



                    // Calculate loss to update the network parameter
                    Tensorflow.Tensor actor_loss_sum = tf_operator.convert_to_tensor(0, Tensorflow.TF_DataType.TF_FLOAT);
                    Tensorflow.Tensor critic_loss_sum = tf_operator.convert_to_tensor(0, Tensorflow.TF_DataType.TF_FLOAT);
                    for (int p = 0; p < action_probs_history.Count; p++)
                    {
                        Tensorflow.Tensor return_v = return_list[p];
                        Tensorflow.Tensor predict_v = critic_value_history[p];
                        return_v = tf_operator.expand_dims(return_v, 0);
                        predict_v = tf_operator.expand_dims(predict_v, 0);
                        //Console.WriteLine("RETURN VALUE  " + return_v.ToString());
                        //Console.WriteLine("PREDICTION " + predict_v.ToString());
                        Tensorflow.Tensor temporal_difference = return_v - predict_v;
                        //Console.WriteLine("Temporal Difference {0}", temporal_difference.ToString());
                        Tensorflow.Tensor actor_loss = -tf_operator.convert_to_tensor(action_probs_history[p], Tensorflow.TF_DataType.TF_FLOAT) * temporal_difference;
                        Tensorflow.Tensor critic_loss = temporal_difference * temporal_difference;
                        //Console.WriteLine("Actor loss {0}  Critic Loss {1} ", actor_loss.ToString(), critic_loss.ToString());
                        actor_loss_sum = actor_loss_sum + actor_loss;
                        critic_loss_sum = critic_loss_sum + critic_loss;
                    }
                    //Console.WriteLine("SUM ACTOR LOSS " + actor_loss_sum.ToString());

                    //Console.WriteLine("SUM Critic LOSS " + critic_loss_sum.ToString());
                    Tensorflow.Tensor[] actor_gradient = t_a.gradient(actor_loss_sum, actor_net.TrainableVariables);
                    optimizer_actor.apply_gradients(zip(actor_gradient,actor_net.TrainableVariables));


                    Tensorflow.Tensor[] critic_gradient = t_b.gradient(critic_loss_sum, critic_net.TrainableVariables);
                    optimizer_critic.apply_gradients(zip(critic_gradient, critic_net.TrainableVariables));

                    

                    


                }
                Console.WriteLine("Epidose {0} reward {1}", epoch.ToString(),episode_reward.ToString());
                //Console.WriteLine("finish one episode.");







            }
            training_writer.Close();
            Console.WriteLine("Optimization finished. Press Enter to exit...");
            Console.ReadLine();





















        }
    }
}
