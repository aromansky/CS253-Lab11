using System;
using System.Collections.Generic;
using System.Collections;
using System.Linq;

namespace AIMLTGBot
{
    // Изменили список классов под задание (10 букв)
    public enum FigureType : byte
    {
        A = 0,
        B,
        W,
        G,
        D,
        E,
        V,
        Z,
        I,
        K,
        Undef = 255
    }

    /// <summary>
    /// Класс для хранения образа – входной массив сигналов на сенсорах, выходные сигналы сети, и прочее
    /// </summary>
    public class Sample
    {
        public double[] input = null;
        public double[] error = null;
        public FigureType actualClass;
        public FigureType recognizedClass;

        public Sample(double[] inputValues, int classesCount, FigureType sampleClass = FigureType.Undef)
        {
            input = (double[])inputValues.Clone();
            Output = new double[classesCount];
            if (sampleClass != FigureType.Undef) Output[(int)sampleClass] = 1;

            recognizedClass = FigureType.Undef;
            actualClass = sampleClass;
        }

        public double[] Output { get; private set; }

        public FigureType ProcessPrediction(double[] neuralOutput)
        {
            Output = neuralOutput;
            if (error == null)
                error = new double[Output.Length];

            recognizedClass = 0;
            // Находим индекс нейрона с максимальной активацией
            int maxIndex = 0;
            double maxVal = double.NegativeInfinity;

            for (int i = 0; i < Output.Length; ++i)
            {
                error[i] = (Output[i] - (i == (int)actualClass ? 1 : 0));
                if (Output[i] > maxVal)
                {
                    maxVal = Output[i];
                    maxIndex = i;
                }
            }
            recognizedClass = (FigureType)maxIndex;
            return recognizedClass;
        }

        public double EstimatedError()
        {
            double Result = 0;
            for (int i = 0; i < Output.Length; ++i)
                Result += Math.Pow(error[i], 2);
            return Result;
        }

        public bool Correct()
        {
            return actualClass == recognizedClass;
        }
    }

    /// <summary>
    /// Выборка образов.
    /// </summary>
    public class SamplesSet : IEnumerable
    {
        public List<Sample> samples = new List<Sample>();

        public void AddSample(Sample image)
        {
            samples.Add(image);
        }

        public int Count => samples.Count;

        public IEnumerator GetEnumerator()
        {
            return samples.GetEnumerator();
        }

        public Sample this[int i]
        {
            get => samples[i];
            set => samples[i] = value;
        }

        public double TestNeuralNetwork(BaseNetwork network)
        {
            double correct = 0;
            foreach (var sample in samples)
            {
                if (sample.actualClass == network.Predict(sample)) ++correct;
            }
            return correct / samples.Count;
        }
    }
}