using UnityEngine;

public class NN : MonoBehaviour
{
    public Layer[] Layers;
    public int[] NetworkShape = { 2, 4, 4, 2 };

    public class Layer
    {
        public float[,] WeightsArray;
        public float[] BiasesArray;
        public float[] NodeArray;

        private int n_nodes;
        private int n_inputs;

        public Layer(int n_inputs, int n_nodes)
        {
            this.n_nodes = n_nodes;
            this.n_inputs = n_inputs;

            WeightsArray = new float[n_nodes, n_inputs];
            BiasesArray = new float[n_nodes];
            NodeArray = new float[n_nodes];
        }

        public void Forward(float[] inputsArray)
        {
            NodeArray = new float[n_nodes];

            for (int i = 0; i < n_nodes; i++)
            {
                //sum of weights times inputs
                for (int j = 0; j < n_inputs; j++)
                {
                    NodeArray[i] += WeightsArray[i, j] * inputsArray[j];
                }

                //add the bias
                NodeArray[i] += BiasesArray[i];
            }
        }

        public void Activation()
        {
            for (int i = 0; i < n_nodes; i++)
            {
                if (NodeArray[i] < 0)
                {
                    NodeArray[i] = 0;
                }
            }
        }
    }

    public void Awake()
    {
        Layers = new Layer[NetworkShape.Length - 1];
        for (int i = 0; i < Layers.Length; i++)
        {
            Layers[i] = new Layer(NetworkShape[i], NetworkShape[i + 1]);
        }

    }

    public float[] Brain(float[] inputs)
    {
        for (int i = 0; i < Layers.Length; i++)
        {
            if (i == 0)
            {
                Layers[i].Forward(inputs);
                Layers[i].Activation();
            }
            else if (i == Layers.Length - 1)
            {
                Layers[i].Forward(Layers[i - 1].NodeArray);
            }
            else
            {
                Layers[i].Forward(Layers[i - 1].NodeArray);
                Layers[i].Activation();
            }
        }

        return (Layers[Layers.Length - 1].NodeArray);
    }
}
