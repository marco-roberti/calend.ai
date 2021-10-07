import json
import random
import signal
from argparse import ArgumentParser

from bot.model import CalendaBot
from twitter import Tweet


def main(args):
    bot = CalendaBot(args)

    with open(args.test_file) as f:
        lines = f.readlines()
        random.shuffle(lines)
        for line in lines:
            example = json.loads(line)
            inp, ref = example['input'], example['output']
            answers = bot.reply_to(Tweet.from_str(inp))
            print(f'inp> {inp}')
            print(f'ref> {ref}')
            for answer in answers:
                print(f'gen> {answer}')
            print()
            if input():
                bot.interactive_set_args()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda signum, frame: exit())
    parser = ArgumentParser(description='Interactive test of @Calend_AI\'s response tweets')
    parser.add_argument('model_path')
    parser.add_argument('config_file')
    parser.add_argument('--blacklist', '-b', default=None)
    parser.add_argument('--test_file', default='data/test.json')
    main(parser.parse_args())
