using AIMLbot;
using AIMLbot.AIMLTagHandlers;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Telegram.Bot.Types;

namespace AIMLTGBot
{
    public class AIMLService
    {
        readonly Bot bot;
        readonly Dictionary<long, AIMLbot.User> users = new Dictionary<long, AIMLbot.User>();

        public AIMLService()
        {
            bot = new Bot();
            bot.loadSettings();
            bot.isAcceptingUserInput = false;
            bot.loadAIMLFromFiles();
            bot.isAcceptingUserInput = true;
        }

        public string Talk(long userId, string userName, string phrase)
        {
            var result = "";
            AIMLbot.User user;
            if (!users.ContainsKey(userId))
            {
                user = new AIMLbot.User(userId.ToString(), bot);
                users.Add(userId, user);
                Request r = new Request($"Меня зовут {userName}", user, bot);
                result += bot.Chat(r).Output + System.Environment.NewLine;
                return result;
            }
            else
            {
                user = users[userId];
            }
            if (phrase.Contains("ё")) phrase = phrase.Replace('ё', 'е');

            result += bot.Chat(new Request(phrase.ToLower(), user, bot)).Output;
            return result;
        }
    }
}
