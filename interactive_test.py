import json
import signal
from argparse import ArgumentParser

from bot.model import CalendaBot


def main(args):
    bot = CalendaBot(args)

    with open(args.test_file) as f:
        for line in f:
            example = json.loads(line)
            inp, ref = example['input'], example['output']
            answers = bot.reply_to(inp)
            print(f'inp> {inp}')
            print(f'ref> {ref}')
            for answer in answers:
                print(f'ref> {answer}')
            print()
            if input():
                bot.interactive_set_args()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda signum, frame: exit())
    parser = ArgumentParser(description='Interactive test of @Calend_AI\'s response tweets')
    parser.add_argument('model_path')
    parser.add_argument('config_file')
    parser.add_argument('--test_file', default='data/test.json')
    main(parser.parse_args())
