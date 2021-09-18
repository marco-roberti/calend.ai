import json
import logging
import random
import re
from queue import Queue
from threading import Thread

import telegram_send
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from twitter import post_reply, Tweet


class CalendaBot:
    def __init__(self, args, interactive=False):
        with open(args.config_file) as f:
            self.gen_args = json.load(f)

        model_path = args.model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

        self.interactive = interactive
        if self.interactive:
            # Some tweets' answers needs to be manually confirmed. This is done in a separate thread
            self.queue = Queue()
            Thread(target=self.worker, daemon=True).start()

    def worker(self):
        while True:
            self.reply_and_check(*self.queue.get())
            self.queue.task_done()

    @staticmethod
    def reply_and_check(to_tweet, replies, message):
        # ask for confirmation
        print(message)
        try:
            i = int(input('Choose your reply > '))
        except ValueError:
            i = -1

        if i in range(len(replies)):
            logging.info(f'[async] Replying to tweet {to_tweet}')
            post_reply(replies[i], to_tweet)
        else:
            logging.info(f'Tweet "{to_tweet}" ignored.')

    def on_quote(self, response_bytes):
        # Parse HTTP response
        response = json.loads(response_bytes)
        to_tweet = Tweet.from_http(response['data'])

        # Generate reply
        replies = self.reply_to(to_tweet)

        # Check manual confirmation
        if self.interactive and \
                len(response['matching_rules']) == 1 and 'confirm' in response['matching_rules'][0]['tag']:
            message = 'Choice required:\n' \
                      'tweet > {to_tweet}\n' + \
                      ('\n'.join(f'{i} > {reply}' for i, reply in enumerate(replies)))
            self._maybe_send_notification(message)
            self.queue.put((to_tweet, replies, message))
            return

        reply = random.choice(replies)

        # Post reply
        logging.info(f'[sync] Replying to tweet {to_tweet}')
        self._maybe_send_notification(f'New reply sent:\ntweet > {to_tweet}\nreply > {reply}')
        post_reply(reply, to_tweet)

    def reply_to(self, tweet: Tweet):
        text = self.tokenizer(str(tweet), return_tensors='pt')
        replies = self.model.generate(**text, **self.gen_args)
        replies = self.tokenizer.batch_decode(replies, skip_special_tokens=True)
        return list(filter(
            lambda reply: reply != f'@{tweet.username}',
            set(self.post_process(reply, tweet) for reply in replies)
        ))

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
                reply = re.sub(rf' ?@{username}', '', reply)
        # Remove old-fashioned parties
        reply = re.sub(r'[-/]?siamoeuropei', '', reply)
        # Ensure mention
        if f'@{to_tweet.username.lower()}' not in reply:
            reply = f'@{to_tweet.username.lower()} {reply.strip()}'
        return reply.strip()

    @staticmethod
    def _maybe_send_notification(message):
        try:
            telegram_send.send(messages=[message])
        except Exception as e:
            logging.warning(f"Couldn't send Telegram notification: {e}")
