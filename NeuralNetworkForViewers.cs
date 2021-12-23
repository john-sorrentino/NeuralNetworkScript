using UnityEngine;

public class NeuralNetworkForViewers : MonoBehaviour
{
    int inputNodes = 4;
    int hiddenLayerNodes = 8;
    int numberOfHiddenLayers = 2;
    int outputNodes = 4;
    
    //This is where my raycasts came from. Since you don't have the script for this it won't work unless you make your own. 
    //Raycast r;

    //This was the built in car controller the asset came with (how I controlled the car)
    //WheelVehicle wheelVehicle;
    public Layer_Dense [] layers;

    //This is how I got the velocity of the car
    //Rigidbody m_rigidbody;
    public float currentVel;

    // Start is called before the first frame update
    public void Start()
    {
        //This sets the seed for our random numbers to be the current time (allows for really random numbers). 
        Random.InitState((int)System.DateTime.Now.Ticks);
        // this creates the array of layers, the reason we add one to this is because the ouput layer is also a layer but it isn't a hidden layer.
        layers = new Layer_Dense[numberOfHiddenLayers + 1];

        
        //r = this.GetComponentInParent<Raycast>();
        //wheelVehicle = this.GetComponentInParent<WheelVehicle>();
        

        // since the ouput layer is the final layer we don't use the activation function on it.

        //This loop initializes the neural network with an empty network 
        //the network sizes are deternimed by the following variables
        // inputNodes
        // hiddenLayerNodes
        // ouputNodes 
        for(int i = 0; i < layers.Length; i++)
        {
            if(i == 0)
            {
                //this is the first layer of the network that connects the inputs to the nodes in the network (the input layer)
                layers[i] = new Layer_Dense(inputNodes, hiddenLayerNodes);
            }
            else if(i == layers.Length-1)
            {
                //This is the last layer fo the network that connects the nodes of the network to the final ouputs (the ouput layer)
                layers[i] = new Layer_Dense(hiddenLayerNodes, outputNodes);
            }
            else
            {
                //These layers are the hidden layers and you can have as many of these layers as you want (*there can only be one input and ouput layer)
                //The way this is setup all of the hidden layers have the same number of nodes 
                //however it is possible to set things up so each hidden layer can have a custom number of nodes
                layers[i] = new Layer_Dense(hiddenLayerNodes, hiddenLayerNodes);
            }
        }
        
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        if(true)//Random.value*100 <50)
        {
            //float [] rays = r.getRayLengths();

            //m_rigidbody = this.GetComponent<Rigidbody>();
            //float [] velocity = new float [1];
            //velocity[0] = m_rigidbody.velocity.magnitude/25;
            //currentVel = velocity[0];
            //velocity[1] = m_rigidbody.velocity.y / 100;
            //velocity[2] = m_rigidbody.velocity.z / 100;
            //float [] brainInputs = rays.Concat(velocity).ToArray();

            //The Brain function takes in an array of numbers which are the inputs to the network. 
            //Make sure to replace this zero filled array with a real array with numbers that you want to give as inputs to the network.
            //The array needs to be the same length as the number of inputNodes you have set in the variable at the top of this file. 
            float [] brainInputs = {0,0,0,0};


            float [] brainOutputs = Brain(brainInputs);

            //This is an example of how I used the output array in my driving ai video
            //wheelVehicle.externalThrottleInput = brainOutputs[0]/10;
            //wheelVehicle.externalTurnInput = brainOutputs[1];
            //wheelVehicle.externalDriftInput = brainOutputs[2];
            //wheelVehicle.externalBoostInput = brainOutputs[3];
        }
    }
    float[] Brain(float [] inputs)
    {
        //This loop applies the forward pass and activations functions for every layer in the network
        //The input and output layer do not have the activation function being applied (you can apply the activation function to the final ouput if you need too)
        //this is what it looks like for a network with 2 hidden layers:    
            //input->(forward->activation)->(forward->activation)->forward->output
        for(int i = 0; i < layers.Length; i++)
        {
            if(i == 0)
            {
                layers[i].Forward(inputs);
            }
            else
            {
                layers[i].Forward(Activation(layers[i - 1].outputArray)); 
            }                
        }

        return(layers[layers.Length - 1].outputArray);
    }

    public Layer_Dense[] copyLayers()
    {
        //This function copies the neural network of the current object onto another object
        //We use this during the reproduction phase of the genetic algorithm
        Layer_Dense[] tmpLayers = new Layer_Dense[numberOfHiddenLayers + 1];

        //This loop deep copies the network to a new one layer at a time. 
        for(int i = 0; i < tmpLayers.Length; i++)
        {
            if(i == 0)
            {
                tmpLayers[i] = new Layer_Dense(inputNodes, hiddenLayerNodes);
            }
            else if(i == layers.Length-1)
            {
                tmpLayers[i] = new Layer_Dense(hiddenLayerNodes, outputNodes);
            }
            else
            {
                tmpLayers[i] = new Layer_Dense(hiddenLayerNodes, hiddenLayerNodes);
            }


            tmpLayers[i].weightsArray = new float [this.layers[i].weightsArray.GetLength(0), this.layers[i].weightsArray.GetLength(1) ];
            tmpLayers[i].biasesArray = new float [this.layers[i].biasesArray.GetLength(0), this.layers[i].biasesArray.GetLength(1) ];
            System.Array.Copy (this.layers[i].weightsArray, tmpLayers[i].weightsArray, this.layers[i].weightsArray.GetLength(0)*this.layers[i].weightsArray.GetLength(1));
            System.Array.Copy (this.layers[i].biasesArray, tmpLayers[i].biasesArray, this.layers[i].biasesArray.GetLength(0)*this.layers[i].biasesArray.GetLength(1));
        }
        return(tmpLayers);
    }

    //The Layer_Dense class allows us to organize the network in groups of layers.
    //Each layer has its own array for weights and biases and also an output array.
    //The ouput array is what the current layer outputs as its values. 
    //These values could end up going through another layer or they may actually be the final ouput for the network. 
    public class Layer_Dense
    {
        //attributes
        public float[,] weightsArray;
        public float[,] biasesArray;
        public float [] outputArray;

        //constructor
        public Layer_Dense(int n_inputs, int n_neurons)
        {
            if (weightsArray== null)
            {
                weightsArray = new float [n_inputs, n_neurons];
                for(int i = 0; i <= weightsArray.GetUpperBound(0); i++)
                {
                    for(int j = 0; j <= weightsArray.GetUpperBound(1); j++)
                    {
                        weightsArray[i, j] = (Random.value-.5f)*2;
                    }
                }
            }
            else
            {
                //Debug.Log("weights array is not null +++++++++++++++++");
            }
            //creates a zero filled vector (This actually doesn't need to be a 2d array anymore but it originally needed to be before I removed the matrix manipulations so it stayed)
            biasesArray = new float[1, n_neurons];
        }

        public void Randomness(float mutationChance, float mutationAmount)
        {
            //This function mutates the values of the neural network and this is what allows the children to be slightly different than the parent 
            //This randomness allows the evolution to occur
            //You need some way to call this function periodically in order for any learning to happen. 
            //In the car sim this was called on each car in between the rounds but after I sorted the cars based on how they performed.
            //The top 10 cars did not get any randomness but they were copied to the rest of the cars and the rest of the cars are the ones that got the randomness applied. 
            int rows = weightsArray.GetUpperBound(0);
            int columns = weightsArray.GetUpperBound(1);
            bool useNorm = true;

            //this section applies the randomness to the weights
            for(int i = 0; i <= rows; i++)
            {
                for(int j = 0; j <= columns; j++)
                {
                    if(Random.value < mutationChance)
                    {
                        if(useNorm)
                        {
                            //Adding a random number to another random number creates a normal distribution (bell curve) 
                            //This is why when you roll 2 dice you are much more likely to roll a 7 than a 2 or a 12
                            weightsArray[i,j] += ( (Random.Range(-1.0f, 1.0f)*mutationAmount)+(Random.Range(-1.0f, 1.0f)*mutationAmount) )/2;
                        }
                        else
                        {
                            weightsArray[i,j] += Random.Range(-1.0f, 1.0f)*mutationAmount;
                        }
                    }
                }
            }

            rows = biasesArray.GetUpperBound(0);
            columns = biasesArray.GetUpperBound(1);

            //This section applies the randomness to the biases
            for(int i = 0; i <= rows; i++)
            {
                for(int j = 0; j <= columns; j++)
                {
                    if(Random.value < mutationChance)
                    {
                        biasesArray[i,j] += Random.Range(-1.0f, 1.0f)*mutationAmount;
                    }
                }
            }  
        }

        //Forward function
        public float [] Forward(float [] forwardInputs)
        {
            //The weights array is the weights assiated with the current layer. Same for biases.
            int rows = weightsArray.GetUpperBound(0);
            int columns = weightsArray.GetUpperBound(1);

            //initialize empty arrays with the correct length
            float [] forwardOutputArray = new float [columns+1];
            if(outputArray == null)
            {
                outputArray = new float [columns+1];
            }  

            //Multiply the weights times the inputs for node/neuron in the layer. 
            for(int i = 0; i <= rows; i++)
            {
                for(int j = 0; j <= columns; j++)
                {
                    forwardOutputArray[j] += forwardInputs[i]*weightsArray[i, j];
                }
            }

            //Adds the bias to each node after the sum of weights and inputs. 
            for(int i = 0; i < forwardOutputArray.Length; i++)
            {
                forwardOutputArray[i] += biasesArray[0,i];
            }

            //update the current layers output values with the values calculated in the forward pass.
            outputArray = forwardOutputArray;

            return(forwardOutputArray);
        }
    }

    //Activation function this function decides whether each node should be active or not.
    //There are many different types of activation functions 
    //The one we are using is called Relu (rectified linear)
    //ReLU is a very simple activation function that basically means if the value is negative then the node is off.
    //Here is a website with other possible activation functions. 
    public float [] Activation(float [] inputs)
    {
        float [] outputArray = new float[inputs.Length];

        for(int i = 0; i < inputs.Length; i++)
        {
            //this if else section clamps the values to be above 0.
            //If the number is less then zero the node is off.
            if(inputs[i] < 0)
            {
                outputArray[i] = 0;
            }
            else
            {
                outputArray[i] = inputs[i];
            }
        }
        return(outputArray);
    }
}