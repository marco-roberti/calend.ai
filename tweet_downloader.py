import calendar
import datetime
import json
import os
import random
import re
from argparse import ArgumentParser
from datetime import date, timedelta
from os import path

from tqdm import tqdm

from twitter import connect_to_endpoint, tweet_url, search_url, get_profile_data

profile = 'CarloCalenda'
profile_id = '2416067982'


def process_tweet(response_tweet):
    # Ensure this tweet answers to an existing one
    if 'referenced_tweets' not in response_tweet or len(response_tweet['referenced_tweets']) != 1:
        return None, response_tweet['created_at']
    # Get the tweet the current one answers to
    input_tweet = connect_to_endpoint(
        tweet_url + response_tweet['referenced_tweets'][0]['id'], {'tweet.fields': 'author_id'})
    # Ensure this tweet answers to an existing one (again)
    if 'errors' in input_tweet:
        return None, response_tweet['created_at']
    input_tweet = input_tweet['data']
    # Exclude self-answers and non-answers
    if input_tweet['author_id'] == profile_id or profile not in input_tweet['text']:
        return None, response_tweet['created_at']
    # Get input tweet's username
    username, name = get_profile_data(input_tweet['author_id'])
    # Return actual data
    input_text = f'{name} @{username} : ' + input_tweet['text'].replace('@CarloCalenda', '@Calend_AI')
    response_text = re.sub(r' http[^ ]+$', '', response_tweet['text']).replace('@CarloCalenda', '@Calend_AI')
    return {'input': input_text, 'output': response_text}, response_tweet['created_at']


def load_tweets(filename):
    with open(filename) as f:
        return [json.loads(line) for line in f]


def save_tweets(tweets, filename):
    with open(filename, 'w') as f:
        f.write('\n'.join(json.dumps(tw) for tw in tweets))


def download_chunk(chunk_start, chunk_end):
    chunk_file = path.join('data', f'{chunk_start.isoformat()}_{chunk_end.day}.json')
    if path.isfile(chunk_file):
        return load_tweets(chunk_file)

    tweets = []
    query_params = {
        'query': f'from:{profile} (is:quote OR is:reply) -is:retweet',
        'start_time': chunk_start.isoformat() + 'T00:00:00+01:00',
        'end_time': chunk_end.isoformat() + 'T00:00:00+01:00',
        'tweet.fields': 'referenced_tweets,created_at',
        'max_results': 100
    }
    pbar = tqdm(desc=profile, unit='Tw')

    def process_response():
        for tw in json_response['data']:
            tw, time = process_tweet(tw)
            if tw is not None:
                tweets.append(tw)
                pbar.set_postfix_str(time)
                pbar.update()

    json_response = connect_to_endpoint(search_url, query_params)
    process_response()

    while 'next_token' in json_response['meta']:
        query_params['next_token'] = json_response['meta']['next_token']
        json_response = connect_to_endpoint(search_url, query_params)
        process_response()

    save_tweets(tweets, chunk_file)
    return tweets


def months_iterator(start_date, end_date):
    month_start = date(start_date.year, start_date.month + 1, 1)
    month_end = month_start - timedelta(days=1)

    iterator = [(start_date, month_end)]
    while month_end < end_date:
        days_in_month = timedelta(days=calendar.monthrange(month_start.year, month_start.month)[1])
        month_end += days_in_month
        iterator.append((month_start, month_end))
        month_start += days_in_month
    iterator[-1] = (date(end_date.year, end_date.month, 1), end_date)
    return iterator


def main(args):
    if not path.isdir('data'):
        os.mkdir('data')

    tweets = list()
    start_date = date.fromisoformat(args.start_date)
    end_date = date.today() - datetime.timedelta(days=1)
    for start, end in months_iterator(start_date, end_date):
        tweets.extend(download_chunk(start, end))

    # Shuffle, split and save dataset
    random.shuffle(tweets)
    eval_size = round(0.05 * len(tweets))
    save_tweets(tweets[:eval_size], path.join('data', f'test.json'))
    save_tweets(tweets[eval_size:2 * eval_size], path.join('data', f'valid.json'))
    save_tweets(tweets[2 * eval_size:], path.join('data', f'train.json'))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--start_date', '-d', default='2020-10-18', help='YYYY-MM-DD')
    main(parser.parse_args())
