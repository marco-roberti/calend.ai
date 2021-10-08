import logging
import time
import traceback
from argparse import ArgumentParser

from bot.model import maybe_send_notification
from model import CalendaBot
from twitter import Stream


def main(args):
    bot = CalendaBot(args, interactive=True)

    stream = Stream([
        {"value": "@Calend_AI -is:retweet -is:reply -has:links", "tag": "qt_reply"}
    ])
    while True:
        try:
            stream.watch(handler=bot.on_quote)
        except Exception:
            print(traceback.format_exc())
            maybe_send_notification(traceback.format_exc())
            time.sleep(30)
            continue


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    parser = ArgumentParser(description='Automatic reply to tweets mentioning @Calend_AI')
    parser.add_argument('model_path')
    parser.add_argument('config_file')
    parser.add_argument('--blacklist', '-b', default=None)
    main(parser.parse_args())
