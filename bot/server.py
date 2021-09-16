import logging
from argparse import ArgumentParser

from model import CalendaBot
from twitter import Stream


def main(args):
    bot = CalendaBot(args)

    stream = Stream([{"value": "@Calend_AI -is:retweet -is:reply"}])
    stream.watch(handler=bot.on_quote)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    parser = ArgumentParser('Automatic reply to tweets mentioning @Calend_AI')
    parser.add_argument('model_path')
    parser.add_argument('config_file')
    main(parser.parse_args())
