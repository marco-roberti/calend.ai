import logging
from argparse import ArgumentParser

from model import CalendaBot
from twitter import Stream


def main(args):
    bot = CalendaBot(args, interactive=True)

    stream = Stream([
        {"value": "@Calend_AI -is:retweet -is:reply -has:links", "tag": "qt_follow_reply"},
        {"value": "(Calenda OR #CalendaSindaco OR #RomaSulSerio) "
                  "-is:retweet -is:reply -has:links", "tag": "ht_follow"},
        {"value": "(from:virginiaraggi OR from:gualtierieurope OR from:EnricoMichetti) "
                  "-is:retweet -is:reply -has:links", "tag": "cd_reply"}
    ])
    while True:
        try:
            stream.watch(handler=bot.on_quote)
        except (ConnectionError, KeyError):
            continue


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    parser = ArgumentParser(description='Automatic reply to tweets mentioning @Calend_AI')
    parser.add_argument('model_path')
    parser.add_argument('config_file')
    parser.add_argument('--blacklist', '-b', default=None)
    main(parser.parse_args())
