import json
from typing import List

import stanza

stanza.download('it')
analyze = stanza.Pipeline('it', processors='tokenize, mwt, pos')


def truecase(tweet):
    doc = analyze(tweet)
    tweet: List = list(tweet)
    for sentence in doc.sentences:
        for i, token in enumerate(sentence.tokens):
            if i == 0 or any(w.upos in ['PROPN', 'X'] for w in token.words):
                tweet[token.start_char] = tweet[token.start_char].upper()
    return ''.join(tweet)


if __name__ == '__main__':
    with open('data/test.json') as f:
        for line in f:
            ref = json.loads(line)['output']
            pred = truecase(ref.lower())
            print(f'ref > {ref}')
            print(f'pred> {pred}')
            print('      ', end='')
            for c1, c2 in zip(ref, pred):
                print('^' if c1 != c2 else ' ', end='')
            print()
            input()
