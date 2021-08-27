import json
from argparse import ArgumentParser
import signal

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def main(args):
    with open(args.config_file) as f:
        gen_args = json.load(f)

    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    name = input('Display name:\t')
    username = '@' + input('Username:\t@')

    print('From now on, you can write your tweets to @CarloCalenda')
    while True:
        text = f'{name} {username} : ' + input('> ')
        text = tokenizer(text, return_tensors='pt')
        answers = model.generate(**text, **gen_args)
        answers = tokenizer.batch_decode(answers, skip_special_tokens=True)
        for answer in answers:
            print(f'@CarloCalenda: {answer}')
        print()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda signum, frame: exit())
    parser = ArgumentParser('Interactive generation of @CarloCalenda\'s response tweets')
    parser.add_argument('model_path')
    parser.add_argument('config_file')
    main(parser.parse_args())
