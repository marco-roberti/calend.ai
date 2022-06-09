## The @Calend_AI Twitter bot
This repository contains the source code I used to create [@Calend_AI](https://twitter.com/Calend_AI), a small toy Language Model I developed for fun in September 2021.

### FAQ
*Who is (or what is) [@Calend_AI](https://twitter.com/Calend_AI)?*

It is an automated Italian Twitter profile. It mimics the peculiar writing style and content of the Italian politician [Carlo Calenda](https://en.wikipedia.org/wiki/Carlo_Calenda), former Italian Minister of Economic Development, former candidate for mayor of Rome and currently MEP.

*How does it work?*

It replies to tweets that mention [@Calend_AI](https://twitter.com/Calend_AI), as long as they are not in turn reply tweets, and do not contain links, images, or videos.

*Who created it?*

I am [Marco Roberti](https://marcoroberti.com), a Ph.D. candidate at the Department of Computer Science, University of Turin. I created this bot partly for fun, partly to practice, partly on inspiration of some chats with my old friend [Federico](https://federicojose.com/).

*What technologies does it use?*

The bot is a [Hugging Face 🤗 Transformers](https://huggingface.co/docs/transformers/index) Language Model, obtained by fine-tuning a dataset created specifically from [@CarloCalenda](https://twitter.com/CarloCalenda) 's tweets, starting from the Italian T5 model developed by [Gabriele Sarti](https://github.com/gsarti) (University of Groningen).

### HowTo

#### Requirements
I ran this code on Python 3.9, but it should be compatible with Python 3.8+.

Required libraries are listed in the `requirements.txt` file, use **one of the following commands** to install them, depending on your environment:
```bash
pip install requirements.txt
# XOR
conda install --file requirements.txt
```

You'll also need to install and configure [Twurl](https://github.com/twitter/twurl).

#### Dataset (download and pre-processing)
In order to massively download tweets, you'll need a [Twitter developer account](https://developer.twitter.com/en/docs/twitter-api/getting-started/getting-access-to-the-twitter-api).

The following command will download and process tweets from October 18th, 2020 (date of Calenda's official candidacy for mayor of Rome) until today:
```bash
BEARER_TOKEN=<bearer_token> python3 tweet_downloader.py
```
You can change the starting date via the `--start_date YYYY-MM-DD` argument. 

#### Training
The training script is a modified version of HuggingFace's [`run_summarization.py`](https://github.com/huggingface/transformers/blob/v4.9.1/examples/pytorch/summarization/run_summarization.py) file.
```bash
python3 main.py config/train.json
```
The model @Calend_AI is currently running is downloadable [here](https://datacloud.di.unito.it/index.php/s/peCc4PWD72yP9rY/download). Its Tensorboard training log [is available as well](https://tensorboard.dev/experiment/xWNfja3RQcqA19pJkGVcxg/#scalars&_smoothingWeight=0).

#### Interactive generation
To check offline your model and tune the `config/generate.json` file, you can use one of the `interactive_*.py` scripts.

Replying to custom tweets you can write on-the-fly:
```bash
python3 interactive_gen.py <checkpoint> config/generate.json -b config/blacklist.txt
```

Replying to tweets in the test set:
```bash
python3 interactive_test.py <checkpoint> config/generate.json --test_file data/test.json -b config/blacklist.txt
```

The `-b config/blacklist.txt` argument is optional on both scripts.

#### Interacting with Twitter
Once you have your optimal model and generating configuration, you can go online!
```bash
PYTHONPATH=. BEARER_TOKEN=<bearer_token> TOKENIZERS_PARALLELISM=true python3 bot/server.py <checkpoint> config/generate.json -b config/blacklist.txt
```

### Useful links
* [Italian T5 Base model](https://huggingface.co/gsarti/it5-base)
* [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (Raffel et al., 2020)](https://arxiv.org/pdf/1910.10683.pdf)
