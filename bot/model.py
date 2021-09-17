import json
import logging
import random
import re

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
        post_reply(reply, to_tweet)

    def reply_to(self, tweet: Tweet):
        text = f'{tweet.name} {tweet.username} : {tweet.text}'
        text = self.tokenizer(text, return_tensors='pt')
        replies = self.model.generate(**text, **self.gen_args)
        replies = self.tokenizer.batch_decode(replies, skip_special_tokens=True)
        return [self.post_process(reply, tweet) for reply in replies]

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

    @staticmethod
    def post_process(reply, to_tweet):
        # Remove unrelated mentions
        re_username = r'@([a-zA-Z0-9_]+)'
        for username in re.findall(re_username, reply):
            if username not in [to_tweet.username.lower()] + re.findall(re_username, to_tweet.text):
                reply = reply.replace(f'@{username}', '')
        # Remove old-fashioned parties
        reply = re.sub(r'-?siamoeuropei', '', reply)
        # Ensure mention
        if f'@{to_tweet.username.lower()}' not in reply:
            reply = f'@{to_tweet.username.lower()} {reply}'
        return reply
