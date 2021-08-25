import json
import os
import random
from os import path
from time import sleep

from tqdm import tqdm

from twitter import connect_to_endpoint

search_url = 'https://api.twitter.com/2/tweets/search/all'
profile = 'CarloCalenda'
eval_size = 100

start_time = '2018-06-01T00:00:00+01:00'


def process_tweet(tweet):
    hashtags = ' '.join('#' + ht['tag'] for ht in tweet['entities']['hashtags'])
    text = tweet['text']
    return {'input': hashtags, 'output': text}


def download_tweets():
    if not path.isdir('data'):
        os.mkdir('data')

    tweets = []
    query_params = {
        'query': f'from:{profile} -is:retweet has:hashtags',  # -is:quote -is:reply
        'start_time': start_time,
        'tweet.fields': 'entities',
        'max_results': 100
    }
    pbar = tqdm(desc=profile, unit='Tw')

    def process_response():
        new_tweets = [process_tweet(tw) for tw in json_response['data']]
        tweets.extend(new_tweets)
        pbar.update(len(new_tweets))

    json_response = connect_to_endpoint(search_url, query_params)
    process_response()

    while 'next_token' in json_response['meta']:
        # sleep(1)  # avoids HTTP response 429 (Too many requests)
        query_params['next_token'] = json_response['meta']['next_token']
        json_response = connect_to_endpoint(search_url, query_params)
        process_response()

    random.shuffle(tweets)
    with open(path.join('data', f'valid.json'), 'w') as f:
        for _ in range(eval_size):
            f.write(json.dumps(tweets.pop()) + '\n')
    with open(path.join('data', f'train.json'), 'w') as f:
        while tweets:
            f.write(json.dumps(tweets.pop()) + '\n')


if __name__ == "__main__":
    download_tweets()
