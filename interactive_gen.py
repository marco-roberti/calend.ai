import signal
from argparse import ArgumentParser

from bot.model import CalendaBot
from twitter import Tweet


def main(args):
    bot = CalendaBot(args)

    name = input('Display name:\t')
    username = '@' + input('Username:\t@')
    tweet = Tweet('', username, name)

    print('From now on, you can write your tweets to @Calend_AI')
    while True:
        tweet.text = input('> ')
        if not tweet.text:
            bot.interactive_set_args()
            continue
        answers = bot.reply_to(tweet)
        for answer in answers:
            print(f'@Calend_AI: {answer}')
        print()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda signum, frame: exit())
    parser = ArgumentParser('Interactive generation of @Calend_AI\'s response tweets')
    parser.add_argument('model_path')
    parser.add_argument('config_file')
    main(parser.parse_args())
