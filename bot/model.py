import json
import logging
import random

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from twitter import post_reply, Tweet


class CalendaBot:
    def __init__(self, args):
        with open(args.config_file) as f:
            self.gen_args = json.load(f)

        model_path = args.model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    def on_quote(self, response_bytes):
        logging.info(f'Replying to tweet {response_bytes}')
        to_tweet = Tweet.from_http(json.loads(response_bytes)['data'])
        replies = self.reply_to(to_tweet)
        reply = random.choice(replies)
        if f'@{to_tweet.username.lower()}' not in reply:
            reply = f'@{to_tweet.username.lower()} {reply}'
        post_reply(reply, to_tweet)

    def reply_to(self, tweet: Tweet):
        text = f'{tweet.name} {tweet.username} : {tweet.text}'
        text = self.tokenizer(text, return_tensors='pt')
        answers = self.model.generate(**text, **self.gen_args)
        return self.tokenizer.batch_decode(answers, skip_special_tokens=True)

    def interactive_set_args(self):
        while True:
            keys = list(self.gen_args.keys())
            for i, k in enumerate(keys):
                print(f'{i}. {k}\t= {self.gen_args[k]}')
            i = input('Key to set: ')
            if not i:
                return
            k = keys[int(i)]
            v = input(f'{i}. {k}\t= ')
            self.gen_args[k] = type(self.gen_args[k])(v)
            print()
