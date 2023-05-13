import logging
import random
from argparse import ArgumentParser

from tqdm import tqdm

from bot.model import CalendaBot
from twitter import Tweet, post_reply


def main(args):
    bot = CalendaBot(args, interactive=True)

    for idx in tqdm(args.ids):
        to_tweet = Tweet.from_id(idx)

        cite = bot.cite
        replies = bot.reply_to(to_tweet, cite)
        reply = random.choice(replies)

        logging.info(f'[sync] Replying to tweet {to_tweet}')
        post_reply(reply, to_tweet, cite)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    parser = ArgumentParser(description='Automatic reply to tweets mentioning @Calend_AI')
    parser.add_argument('model_path')
    parser.add_argument('config_file')
    parser.add_argument('--blacklist', '-b', default=None)
    parser.add_argument('ids', nargs='+')
    main(parser.parse_args())
