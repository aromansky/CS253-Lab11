using AIMLbot;
using AIMLbot.AIMLTagHandlers;
using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Telegram.Bot;
using Telegram.Bot.Exceptions;
using Telegram.Bot.Extensions.Polling;
using Telegram.Bot.Types;
using Telegram.Bot.Types.Enums;

namespace AIMLTGBot
{
    public class TelegramService : IDisposable
    {
        private readonly TelegramBotClient client;
        private readonly AIMLService aiml;
        // CancellationToken - инструмент для отмены задач, запущенных в отдельном потоке
        private readonly CancellationTokenSource cts = new CancellationTokenSource();
        public string Username { get; }
        private StudentNetwork _neuralNetwork;

        public void LoadModel(string filePath)
        {
            string structureLine = System.IO.File.ReadLines(filePath).First();

            int[] structure = structureLine.Split(';')
                                           .Select(int.Parse)
                                           .ToArray();

            _neuralNetwork = new StudentNetwork(structure);

            // 3. Загружаем веса
            _neuralNetwork.Load(filePath);

            Console.WriteLine("Модель успешно загружена!");
        }

        public TelegramService(string token, AIMLService aimlService)
        {
            LoadModel("MyNetwork.txt");
            aiml = aimlService;
            client = new TelegramBotClient(token);
            client.StartReceiving(HandleUpdateMessageAsync, HandleErrorAsync, new ReceiverOptions
            {   // Подписываемся только на сообщения
                AllowedUpdates = new[] { UpdateType.Message }
            },
            cancellationToken: cts.Token);
            // Пробуем получить логин бота - тестируем соединение и токен
            Username = client.GetMeAsync().Result.Username;
        }

        async Task HandleUpdateMessageAsync(ITelegramBotClient botClient, Update update, CancellationToken cancellationToken)
        {
            var message = update.Message;
            var chatId = message.Chat.Id;
            var username = message.Chat.FirstName;
            if (message.Type == MessageType.Text)
            {
                var messageText = update.Message.Text;

                Console.WriteLine($"Received a '{messageText}' message in chat {chatId} with {username}.");

                // Echo received message text
                await botClient.SendTextMessageAsync(
                    chatId: chatId,
                    text: aiml.Talk(chatId, username, messageText),
                    cancellationToken: cancellationToken);
                return;
            }
            // Загрузка изображений пригодится для соединения с нейросетью
            if (message.Type == MessageType.Photo)
            {
                await HandlePhotoMessage(message, chatId, cancellationToken);
                return;
            }
            // Можно обрабатывать разные виды сообщений, просто для примера пробросим реакцию на них в AIML
            if (message.Type == MessageType.Video)
            {
                await client.SendTextMessageAsync(message.Chat.Id, aiml.Talk(chatId, username, "Видео"), cancellationToken: cancellationToken);
                return;
            }
            if (message.Type == MessageType.Audio)
            {
                await client.SendTextMessageAsync(message.Chat.Id, aiml.Talk(chatId, username, "Аудио"), cancellationToken: cancellationToken);
                return;
            }
        }

        Task HandleErrorAsync(ITelegramBotClient botClient, Exception exception, CancellationToken cancellationToken)
        {
            var apiRequestException = exception as ApiRequestException;
            if (apiRequestException != null)
                Console.WriteLine($"Telegram API Error:\n[{apiRequestException.ErrorCode}]\n{apiRequestException.Message}");
            else
                Console.WriteLine(exception.ToString());
            return Task.CompletedTask;
        }

        public void Dispose()
        {
            // Заканчиваем работу - корректно отменяем задачи в других потоках
            // Отменяем токен - завершатся все асинхронные таски
            cts.Cancel();
        }

        private async Task HandlePhotoMessage(Message message, long chatId, CancellationToken cancellationToken)
        {
            Console.WriteLine($"Пользователь {chatId} отправил изображение");
            var photoId = message.Photo.Last().FileId;
            Telegram.Bot.Types.File fl = await client.GetFileAsync(photoId);
            var imageStream = new MemoryStream();
            await client.DownloadFileAsync(fl.FilePath, imageStream, cancellationToken: cancellationToken);
            var img = System.Drawing.Image.FromStream(imageStream);

            System.Drawing.Bitmap bm = new System.Drawing.Bitmap(img);

            // 3. Предобработка (превращаем в массив 0 и 1 размером 40x40)
            double[] input = ImageProcessor.ProcessImage(bm);

            // 4. Создаем Sample и прогоняем через сеть
            Sample sample = new Sample(input, Enum.GetNames(typeof(FigureType)).Length - 1, FigureType.Undef);
            _neuralNetwork.Predict(sample);

            Console.WriteLine($"Распознано {sample.recognizedClass}");
            client.SendTextMessageAsync(chatId, $"Расознана буква {sample.recognizedClass}. Хотите узнать что-нибудь про эту букву?");
        }
    }
}
