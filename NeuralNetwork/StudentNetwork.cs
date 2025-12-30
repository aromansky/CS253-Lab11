using Accord.Math;
using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AIMLTGBot
{
    public class StudentNetwork : BaseNetwork
    {
        // Структура сети: массив, где значение элемента - количество нейронов в слое
        // structure[0] - входной слой, structure[Last] - выходной
        private int[] _structure;

        // Веса: [слой][нейрон_откуда][нейрон_куда]
        // Слой 0 соединяет входной слой и первый скрытый
        private double[][][] _weights;

        // Смещения (biases): [слой][нейрон]
        // Слой 0 соответствует первому скрытому слою
        private double[][] _biases;

        // Значения на выходах нейронов после активации: [слой][нейрон]
        // Слой 0 - это входные данные, Слой Last - выходные данные сети
        private double[][] _outputs;

        // Значения ошибок (дельт) для обратного распространения: [слой][нейрон]
        private double[][] _deltas;

        // Скорость обучения
        private const double LearningRate = 0.1;

        // Поле для хранения флага, нужно ли использовать многопоточность
        private bool _isParallel = false;

        private Random _rand;

        public StudentNetwork(int[] structure)
        {
            _structure = structure;
            _rand = new Random();

            // Инициализация массивов
            int layersCount = structure.Length;

            _outputs = new double[layersCount][];
            _deltas = new double[layersCount][];

            // Весов на один массив меньше, чем слоев нейронов (связи между слоями)
            _weights = new double[layersCount - 1][][];
            _biases = new double[layersCount - 1][];

            // 1. Инициализация нейронов (выходы и дельты)
            for (int i = 0; i < layersCount; i++)
            {
                _outputs[i] = new double[structure[i]];
                _deltas[i] = new double[structure[i]];
            }

            // 2. Инициализация весов и смещений случайными значениями
            for (int layer = 0; layer < layersCount - 1; layer++)
            {
                int neuronsInCurrentLayer = structure[layer];
                int neuronsInNextLayer = structure[layer + 1];

                _weights[layer] = new double[neuronsInCurrentLayer][];
                _biases[layer] = new double[neuronsInNextLayer];

                for (int i = 0; i < neuronsInCurrentLayer; i++)
                {
                    _weights[layer][i] = new double[neuronsInNextLayer];
                    for (int j = 0; j < neuronsInNextLayer; j++)
                    {
                        // Инициализация весов небольшими случайными числами (-0.5 ... 0.5)
                        _weights[layer][i][j] = _rand.NextDouble() - 0.5;
                    }
                }

                for (int j = 0; j < neuronsInNextLayer; j++)
                {
                    _biases[layer][j] = _rand.NextDouble() - 0.5;
                }
            }
        }

        public void Save(string path)
        {
            using (StreamWriter sw = new StreamWriter(path))
            {
                // 1. Сохраняем архитектуру
                sw.WriteLine(string.Join(";", _structure));

                // 2. Собираем все веса в одну длинную строку
                StringBuilder sb = new StringBuilder();

                // Сначала веса связей
                for (int layer = 0; layer < _weights.Length; layer++)
                {
                    for (int i = 0; i < _weights[layer].Length; i++)
                    {
                        for (int j = 0; j < _weights[layer][i].Length; j++)
                        {
                            sb.Append(_weights[layer][i][j].ToString(System.Globalization.CultureInfo.InvariantCulture)).Append(" ");
                        }
                    }
                }

                // Затем смещения (biases)
                for (int layer = 0; layer < _biases.Length; layer++)
                {
                    for (int j = 0; j < _biases[layer].Length; j++)
                    {
                        sb.Append(_biases[layer][j].ToString(System.Globalization.CultureInfo.InvariantCulture)).Append(" ");
                    }
                }

                sw.WriteLine(sb.ToString().Trim());
            }
        }

        public void Load(string path)
        {
            if (!File.Exists(path)) return;

            using (StreamReader sr = new StreamReader(path))
            {
                // Пропускаем архитектуру (предполагаем, что сеть уже создана с правильной структурой)
                sr.ReadLine();

                string data = sr.ReadLine();
                if (string.IsNullOrEmpty(data)) return;

                // Парсим все числа
                //double[][] allValues = data.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries).Select(v => double.Parse(v, System.Globalization.CultureInfo.InvariantCulture)).ToArray();
                var allValues = data.Split().Select(v => double.Parse(v, System.Globalization.CultureInfo.InvariantCulture)).ToArray();

                int index = 0;

                // Загружаем веса связей
                for (int layer = 0; layer < _weights.Length; layer++)
                {
                    for (int i = 0; i < _weights[layer].Length; i++)
                    {
                        for (int j = 0; j < _weights[layer][i].Length; j++)
                        {
                            _weights[layer][i][j] = allValues[index++];
                        }
                    }
                }

                // Загружаем смещения
                for (int layer = 0; layer < _biases.Length; layer++)
                {
                    for (int j = 0; j < _biases[layer].Length; j++)
                    {
                        _biases[layer][j] = allValues[index++];
                    }
                }
            }
        }

        /// <summary>
        /// Сигмоидальная функция активации: 1 / (1 + e^-x)
        /// </summary>
        private double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        /// <summary>
        /// Производная сигмоиды для обратного распространения.
        /// Аргумент y - это уже вычисленное значение функции активации (output).
        /// f'(x) = f(x) * (1 - f(x)) = y * (1 - y)
        /// </summary>
        private double SigmoidDerivative(double y)
        {
            return y * (1.0 - y);
        }

        /// <summary>
        /// Прямой проход (Forward Propagation)
        /// </summary>
        protected override double[] Compute(double[] input)
        {
            // Копируем входные данные в выход 0-го слоя
            for (int i = 0; i < input.Length; i++)
            {
                _outputs[0][i] = input[i];
            }

            // Проходим по всем слоям (кроме входного)
            for (int layer = 0; layer < _structure.Length - 1; layer++)
            {
                int nextLayerCount = _structure[layer + 1];
                int currentLayerCount = _structure[layer];

                // Локальная функция для вычисления состояния одного нейрона.
                // Вынесена отдельно, чтобы вызывать её и в обычном цикле, и в Parallel.For
                void ComputeNeuron(int nextNeuron)
                {
                    double sum = 0;

                    // Сумма взвешенных входов
                    for (int currNeuron = 0; currNeuron < currentLayerCount; currNeuron++)
                    {
                        sum += _outputs[layer][currNeuron] * _weights[layer][currNeuron][nextNeuron];
                    }

                    // Добавляем смещение
                    sum += _biases[layer][nextNeuron];

                    // Применяем функцию активации
                    _outputs[layer + 1][nextNeuron] = Sigmoid(sum);
                }

                // Если включено распараллеливание - используем Parallel.For
                if (_isParallel)
                {
                    Parallel.For(0, nextLayerCount, ComputeNeuron);
                }
                else
                {
                    // Иначе считаем последовательно
                    for (int nextNeuron = 0; nextNeuron < nextLayerCount; nextNeuron++)
                    {
                        ComputeNeuron(nextNeuron);
                    }
                }
            }

            // Возвращаем результат последнего слоя
            return _outputs[_structure.Length - 1];
        }

        /// <summary>
        /// Метод обратного распространения ошибки для одного образа
        /// </summary>
        /// <param name="expectedOutput">Ожидаемый выход (Target)</param>
        /// <returns>Квадратичная ошибка для данного образа</returns>
        private double RunBackPropagation(double[] expectedOutput)
        {
            int outputLayerIndex = _structure.Length - 1;
            double totalError = 0;

            // 1. Вычисление ошибки выходного слоя
            // Здесь операций мало (обычно 4 нейрона), параллелить смысла нет
            for (int i = 0; i < _structure[outputLayerIndex]; i++)
            {
                double actualOutput = _outputs[outputLayerIndex][i];
                double error = expectedOutput[i] - actualOutput; // (Target - Output)

                totalError += error * error; // Суммируем квадратичную ошибку

                // Delta = Error * f'(x)
                _deltas[outputLayerIndex][i] = error * SigmoidDerivative(actualOutput);
            }

            // 2. Вычисление ошибки для скрытых слоев (идем от конца к началу)
            for (int layer = outputLayerIndex - 1; layer > 0; layer--)
            {
                int currentLayerSize = _structure[layer];
                int nextLayerSize = _structure[layer + 1];

                // Локальная функция для вычисления дельты одного нейрона скрытого слоя
                void ComputeHiddenDelta(int i)
                {
                    double sum = 0;
                    // Суммируем вклад ошибки от нейронов следующего слоя
                    for (int j = 0; j < nextLayerSize; j++)
                    {
                        sum += _deltas[layer + 1][j] * _weights[layer][i][j];
                    }

                    // Delta = (Sum of deltas * weights) * f'(x)
                    _deltas[layer][i] = sum * SigmoidDerivative(_outputs[layer][i]);
                }

                // Параллелим вычисление ошибок скрытого слоя (актуально для больших слоев)
                if (_isParallel)
                {
                    Parallel.For(0, currentLayerSize, ComputeHiddenDelta);
                }
                else
                {
                    for (int i = 0; i < currentLayerSize; i++)
                    {
                        ComputeHiddenDelta(i);
                    }
                }
            }

            // 3. Обновление весов и смещений (Градиентный спуск)
            for (int layer = 0; layer < _structure.Length - 1; layer++)
            {
                int currentLayerSize = _structure[layer];
                int nextLayerSize = _structure[layer + 1];

                // Локальная функция обновления весов для конкретного нейрона следующего слоя 'j'
                void UpdateWeightsForNeuron(int j)
                {
                    double delta = _deltas[layer + 1][j];

                    // Обновляем bias
                    _biases[layer][j] += LearningRate * delta;

                    // Обновляем веса
                    for (int i = 0; i < currentLayerSize; i++)
                    {
                        // Weight_new = Weight_old + LearningRate * Delta * Input
                        _weights[layer][i][j] += LearningRate * delta * _outputs[layer][i];
                    }
                }

                // Параллелим самый тяжелый цикл
                if (_isParallel)
                {
                    // Это безопасно, так как каждый поток пишет в свой столбец 'j' и не пересекается с другими
                    Parallel.For(0, nextLayerSize, UpdateWeightsForNeuron);
                }
                else
                {
                    for (int j = 0; j < nextLayerSize; j++)
                    {
                        UpdateWeightsForNeuron(j);
                    }
                }
            }

            return totalError / 2.0; // Обычно берут половину квадратичной суммы для удобства производной, но здесь это просто метрика
        }

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            _isParallel = parallel;
            int iteration = 0;
            double error = double.MaxValue;
            const int maxIterations = 10000; // Защита от бесконечного цикла

            while (error > acceptableError && iteration < maxIterations)
            {
                // Прямой проход
                Compute(sample.input);

                // Обратный проход и обновление весов
                error = RunBackPropagation(sample.Output);

                iteration++;
            }
            return iteration;
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            _isParallel = parallel;
            double currentError = double.MaxValue;
            var startTime = DateTime.Now;

            for (int epoch = 0; epoch < epochsCount; epoch++)
            {
                samplesSet.samples.Shuffle();
                double epochErrorSum = 0;

                // В стохастическом градиентном спуске порядок важен, параллелить цикл по образцам 
                // без блокировок весов (Race Condition) нельзя.
                // Для учебного примера делаем последовательно для стабильности сходимости.
                foreach (var sample in samplesSet.samples)
                {
                    Compute(sample.input);
                    epochErrorSum += RunBackPropagation(sample.Output);
                }

                currentError = epochErrorSum; // Можно делить на Count, но для графика важна динамика

                // Вызываем событие обновления прогресса
                OnTrainProgress(
                    (double)epoch / epochsCount,
                    currentError,
                    DateTime.Now - startTime
                );

                // Если достигли нужной точности - выходим
                if (currentError < acceptableError) break;
            }

            return currentError;
        }
    }
}