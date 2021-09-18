import logging
from argparse import ArgumentParser

from model import CalendaBot
from twitter import Stream


def main(args):
    bot = CalendaBot(args, interactive=True)

    stream = Stream([
        {"value": "@Calend_AI -is:retweet -is:reply"},
        {"value": "(#Calenda OR #CalendaSindaco OR #RomaSulSerio) "
                  "-is:retweet -is:reply -has:links -has:media -has:mentions", "tag": "confirm"}
    ])
    stream.watch(handler=bot.on_quote)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    parser = ArgumentParser('Automatic reply to tweets mentioning @Calend_AI')
    parser.add_argument('model_path')
    parser.add_argument('config_file')
    parser.add_argument('--interactive_timeout', '-t', type=int, default=30,
                        help="Timeout when confirming an answer to #calenda stream's tweet")
    main(parser.parse_args())
