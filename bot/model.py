import json
import logging

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
        to_tweet = Tweet(json.loads(response_bytes)['data'])
        replies = self.reply_to(to_tweet)
        for reply in set(replies):
            if f'@{to_tweet.username.lower()}' not in reply:
                reply = f'@{to_tweet.username.lower()} {reply}'
            post_reply(reply, to_tweet)

    def reply_to(self, tweet: Tweet):
        text = f'{tweet.name} {tweet.username} : {tweet.text}'
        text = self.tokenizer(text, return_tensors='pt')
        answers = self.model.generate(**text, **self.gen_args)
        return self.tokenizer.batch_decode(answers, skip_special_tokens=True)
