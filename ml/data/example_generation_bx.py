#   Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Prepare TF.Examples for on-device recommendation model.

Following functions are included: 1) downloading raw data 2) processing to user
activity sequence and splitting to train/test data 3) convert to TF.Examples
and write in output location.

More information about the bx dataset can be found here:
https://grouplens.org/datasets/book-crossing/
"""

import collections
import json
import os
import random
import re

from absl import app
from absl import flags
from absl import logging
import pandas as pd
import tensorflow as tf

FLAGS = flags.FLAGS

# Permalinks to download movielens data.
BX_URL = "http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip"
BX_ZIP_FILENAME = "BX-CSV-Dump.zip"
BX_ZIP_HASH = "adc7b47afb47c8e29e5f0b70a9816569b3367c5588064ba7a916f16e3ef5d060"
BX_EXTRACTED_DIR = "BX-CSV-Dump"

RATINGS_FILE_NAME = "BX-Book-Ratings.csv"
BX_FILE_NAME = "BX-Books.csv"
USERS_FILE_NAME = "BX-Users.csv"

RATINGS_DATA_COLUMNS = ["User-ID", "ISBN", "Book-Rating"]
BX_DATA_COLUMNS = ["ISBN", "Book-Title", "Book-Author", "Year-Of-Publication", "Publisher"]
USERS_DATA_COLUMNS = ["User-ID", "Location", "Age"]

OUTPUT_BX_VOCAB_FILENAME = "bx_vocab.json"
OUTPUT_BX_YEAR_VOCAB_FILENAME = "bx_year_vocab.txt"
OUTPUT_BX_AUTHOR_VOCAB_FILENAME = "bx_author_vocab.txt"
OUTPUT_BX_TITLE_UNIGRAM_VOCAB_FILENAME = "bx_title_unigram_vocab.txt"
OUTPUT_BX_TITLE_BIGRAM_VOCAB_FILENAME = "bx_title_bigram_vocab.txt"

#MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
#MOVIELENS_ZIP_FILENAME = "ml-1m.zip"
#MOVIELENS_ZIP_HASH = "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20"
#MOVIELENS_EXTRACTED_DIR = "ml-1m"
#RATINGS_FILE_NAME = "ratings.dat"
#MOVIES_FILE_NAME = "movies.dat"
#RATINGS_DATA_COLUMNS = ["UserID", "MovieID", "Rating", "Timestamp"]
#MOVIES_DATA_COLUMNS = ["MovieID", "Title", "Genres"]
OUTPUT_TRAINING_DATA_FILENAME = "train_bx.tfrecord"
OUTPUT_TESTING_DATA_FILENAME = "test_bx.tfrecord"
#OUTPUT_MOVIE_VOCAB_FILENAME = "movie_vocab.json"
#OUTPUT_MOVIE_YEAR_VOCAB_FILENAME = "movie_year_vocab.txt"
#OUTPUT_MOVIE_GENRE_VOCAB_FILENAME = "movie_genre_vocab.txt"
#OUTPUT_MOVIE_TITLE_UNIGRAM_VOCAB_FILENAME = "movie_title_unigram_vocab.txt"
#OUTPUT_MOVIE_TITLE_BIGRAM_VOCAB_FILENAME = "movie_title_bigram_vocab.txt"
PAD_BX_ID = 0
PAD_RATING = 0.0
PAD_MOVIE_YEAR = 0
UNKNOWN_STR = "UNK"
VOCAB_BX_ID_INDEX = 0
VOCAB_COUNT_INDEX = 3


def define_flags():
  """Define flags."""
  flags.DEFINE_string("data_dir", "\data",
                      "Path to download and store bx data.")
  flags.DEFINE_string("output_dir", None,
                      "Path to the directory of output files.")
  flags.DEFINE_bool("build_vocabs", True,
                    "If yes, generate bx feature vocabs.")
  flags.DEFINE_integer("min_timeline_length", 3,
                       "The minimum timeline length to construct examples.")
  flags.DEFINE_integer("max_context_length", 10,
                       "The maximum length of user context history.")
  flags.DEFINE_integer(
      "min_rating", None, "Minimum rating of book that will be used to in "
      "training data")
  flags.DEFINE_float("train_data_fraction", 0.9, "Fraction of training data.")


class BookInfo(
    collections.namedtuple(
        "BookInfo", ["isbn", "title", "author", "publisher", "year", "rating"])):
  """Data holder of basic information of a book."""
  __slots__ = ()

  def __new__(cls,
              isbn=PAD_BX_ID,
              title="",
              author="",
              publisher="",
              year=9999,
              rating=PAD_RATING):
    return super(BookInfo, cls).__new__(cls, isbn, title, author, publisher, year, rating)


def download_and_extract_data(data_directory,
                              url=BX_URL,
                              fname=BX_ZIP_FILENAME,
                              file_hash=BX_ZIP_HASH,
                              extracted_dir_name=BX_EXTRACTED_DIR):
  """Download and extract zip containing BX data to a given directory.

  Args:
    data_directory: Local path to extract dataset to.
    url: Direct path to BX dataset .zip file. See constants above for
      examples.
    fname: str, zip file name to download.
    file_hash: str, SHA-256 file hash.
    extracted_dir_name: str, extracted dir name under data_directory.

  Returns:
    Downloaded and extracted data file directory.
  """
  if not tf.io.gfile.exists(data_directory):
    tf.io.gfile.makedirs(data_directory)
  
  path_to_zip = tf.keras.utils.get_file(
      fname=fname,
      origin=url,
      file_hash=file_hash,
      hash_algorithm="sha256",
      extract=True,
      cache_dir=data_directory)
  extracted_file_dir = os.path.dirname(path_to_zip)
  #os.path.join(os.path.dirname(path_to_zip), extracted_dir_name)
        
  return extracted_file_dir


def read_data(data_directory, min_rating=None):
  """Read bx ratings.dat and books.dat file into dataframe."""  
  
  ratings_df = pd.read_csv(RATINGS_FILE_NAME, sep=";", header=0, quoting=3, quotechar='"', names=RATINGS_DATA_COLUMNS, engine='python', encoding="unicode_escape")  
  # May contain unicode. Need to escape.
#  ratings_df["Timestamp"] = ratings_df["Timestamp"].apply(int)
  #if min_rating is not None:
    #ratings_df = ratings_df[ratings_df["Book-Rating"] >= min_rating]
    
  #bx_df = pd.read_csv(os.path.join(data_directory, BX_FILE_NAME), sep="::", names=BX_DATA_COLUMNS, engine='python', encoding="unicode_escape")
  bx_df = pd.read_csv(BX_FILE_NAME, sep=";", header=0, quoting=3, quotechar='"', names=BX_DATA_COLUMNS, engine='python', encoding="unicode_escape", error_bad_lines=False)
  # May contain unicode. Need to escape.
  
  return ratings_df, bx_df


def convert_to_timelines(ratings_df):
  """Convert ratings data to user."""
  timelines = collections.defaultdict(list)
  bx_counts = collections.Counter()
  for user_id, isbn, rating in ratings_df.values:
    timelines[user_id].append(
        BookInfo(isbn, "", "", "", 0, rating))
    bx_counts[isbn] += 1
  # Sort per-user timeline by timestamp
  #for (user_id, context) in timelines.items():
   # context.sort(key=lambda x: x.timestamp)
    #timelines[user_id] = context
  return timelines, bx_counts


def generate_bx_dict(bx_df):
  """Generates books dictionary from books dataframe."""

  bx_dict = {
      isbn: BookInfo(isbn=isbn, title=BookTitle, author=BookAuthor, publisher=Publisher, year=YearOfPublication)
      for isbn, BookTitle, BookAuthor, Publisher, YearOfPublication in bx_df.values
  }
  bx_dict[0] = BookInfo()
  return bx_dict


def generate_book_authors(bx_dict, books):
  """Create a feature of the author of each book.

  Save author as a feature for the book.

  Args:
    bx_dict: Dict of books, keyed by isbn with value of (isbn, title, author, publisher, year)
    books: list of books to extract authors.

  Returns:
    book_authors: list of authors of all input books.
  """
  book_authors = []
  for book in books:
    if not bx_dict[book.isbn].author:
      continue
    authors = [
        tf.compat.as_bytes(author)
        for author in bx_dict[book.isbn].author.split("|")
    ]
    authors.extend(author)

  return book_authors


def _pad_or_truncate_book_feature(feature, max_len, pad_value):
    feature.extend([pad_value for _ in range(max_len - len(feature))])
    return feature[:max_len]


def generate_examples_from_single_timeline(timeline,
                                           bx_dict,
                                           max_context_len=100):
  """Generate TF examples from a single user timeline.

  Generate TF examples from a single user timeline. Timeline with length less
  than minimum timeline length will be skipped. And if context user history
  length is shorter than max_context_len, features will be padded with default
  values.

  Args:
    timeline: The timeline to generate TF examples from.
    bx_dict: Dictionary of all BookInfos.
    max_context_len: The maximum length of the context. If the context history
      length is less than max_context_length, features will be padded with
      default values.
    max_context_book_authors_len: The length of book author feature.

  Returns:
    examples: Generated examples from this single timeline.
  """
  examples = []
  for label_idx in range(1, len(timeline)):
    start_idx = max(0, label_idx - max_context_len)
    context = timeline[start_idx:label_idx]
    # Pad context with out-of-vocab bx id 0.
    while len(context) < max_context_len:
      context.append(BookInfo())
      
#ValueError: invalid literal for int() with base 10: '"0449006522"'   
    
#    label_isbn = (timeline[label_idx].isbn)
    
    context_bx_id = [book.isbn for book in context]
    context_bx_rating = [book.rating for book in context]
    context_bx_year = [book.year for book in context]
#    context_movie_genres = generate_movie_genres(movies_dict, context)
#    context_movie_genres = _pad_or_truncate_movie_feature(
#        context_movie_genres, max_context_movie_genre_len,
#        tf.compat.as_bytes(UNKNOWN_STR))
    context_bx_author = [book.author for book in context]
    
    #logging.info("Context bx id: %s", context_bx_id)

    feature = {
        "context_bx_id":
            tf.train.Feature(
                bytes_list=tf.train.BytesList(value=context_bx_id)),
        "context_bx_rating":
            tf.train.Feature(
                bytes_list=tf.train.BytesList(value=context_bx_rating)),
        #"context_movie_genre":
         #   tf.train.Feature(
          #      bytes_list=tf.train.BytesList(value=context_movie_genres)),
        "context_bx_year":
            tf.train.Feature(
                bytes_list=tf.train.BytesList(value=context_bx_year)),
        "context_bx_author":
            tf.train.Feature(
                bytes_list=tf.train.BytesList(value=context_bx_author))
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    examples.append(tf_example)

  return examples


def generate_examples_from_timelines(timelines,
                                     bx_df,
                                     min_timeline_len=3,
                                     max_context_len=100,
                                     train_data_fraction=0.9,
                                     random_seed=None,
                                     shuffle=True):
  """Convert user timelines to tf examples.

  Convert user timelines to tf examples by adding all possible context-label
  pairs in the examples pool.

  Args:
    timelines: The user timelines to process.
    movies_df: The dataframe of all movies.
    min_timeline_len: The minimum length of timeline. If the timeline length is
      less than min_timeline_len, empty examples list will be returned.
    max_context_len: The maximum length of the context. If the context history
      length is less than max_context_length, features will be padded with
      default values.
    train_data_fraction: Fraction of training data.
    random_seed: Seed for randomization.
    shuffle: Whether to shuffle the examples before splitting train and test
      data.

  Returns:
    train_examples: TF example list for training.
    test_examples: TF example list for testing.
  """
  examples = []
  bx_dict = generate_bx_dict(bx_df)
  progress_bar = tf.keras.utils.Progbar(len(timelines))
  for timeline in timelines.values():
    if len(timeline) < min_timeline_len:
      progress_bar.add(1)
      continue
    single_timeline_examples = generate_examples_from_single_timeline(
        timeline=timeline,
        bx_dict=bx_dict,
        max_context_len=max_context_len)
    examples.extend(single_timeline_examples)
    progress_bar.add(1)
  # Split the examples into train, test sets.
  if shuffle:
    random.seed(random_seed)
    random.shuffle(examples)
  last_train_index = round(len(examples) * train_data_fraction)

  train_examples = examples[:last_train_index]
  test_examples = examples[last_train_index:]
  return train_examples, test_examples


def generate_bx_feature_vocabs(bx_df, bx_counts):
  """Generate vocabularies for book features.

  Generate vocabularies for book features (), sorted by
  usage count. Vocab id 0 will be reserved for default padding value.

  Args:
    bx_df: Dataframe for books.
    bx_counts: Counts that each book is rated.

  Returns:
    bx_id_vocab: List of all book ids paired with book usage count, and
      sorted by counts.
    book_author_vocab: List of all book authors, sorted by author usage counts.
    book_year_vocab: List of all book years, sorted by year usage counts.
  """
  book_vocab = []
  book_author_counter = collections.Counter()
  book_year_counter = collections.Counter()
  for isbn, title, author, year in bx_df.values:
    count = bx_counts.get(isbn) or 0
    book_vocab.append([isbn, title, author, year, count])
    book_year_counter[year] += 1
    book_author_counter[author] += 1

  book_vocab.sort(key=lambda x: x[VOCAB_COUNT_INDEX], reverse=True)  # by count
  book_year_vocab = [0] + [x for x, _ in book_year_counter.most_common()]
  book_author_vocab = [UNKNOWN_STR
                      ] + [x for x, _ in book_author_counter.most_common()]

  return (book_vocab, book_year_vocab, book_vocab)


def write_tfrecords(tf_examples, filename):
  """Writes tf examples to tfrecord file, and returns the count."""
  with tf.io.TFRecordWriter(filename) as file_writer:
    length = len(tf_examples)
    progress_bar = tf.keras.utils.Progbar(length)
    for example in tf_examples:
      file_writer.write(example.SerializeToString())
      progress_bar.add(1)
    return length


def write_vocab_json(vocab, filename):
  """Write generated movie vocabulary to specified file."""
  with open(filename, "w", encoding="utf-8") as jsonfile:
    json.dump(vocab, jsonfile, indent=2)


def write_vocab_txt(vocab, filename):
  with open(filename, "w", encoding="utf-8") as f:
    for item in vocab:
      f.write(str(item) + "\n")


def generate_datasets(extracted_data_dir,
                      output_dir,
                      min_timeline_length,
                      max_context_length,
                      min_rating=None,
                      build_vocabs=True,
                      train_data_fraction=0.9,
                      train_filename=OUTPUT_TRAINING_DATA_FILENAME,
                      test_filename=OUTPUT_TESTING_DATA_FILENAME,
                      vocab_filename=OUTPUT_BX_VOCAB_FILENAME,
                      vocab_year_filename=OUTPUT_BX_YEAR_VOCAB_FILENAME,
                      vocab_author_filename=OUTPUT_BX_AUTHOR_VOCAB_FILENAME):
  """Generates train and test datasets as TFRecord, and returns stats."""

  ratings_df, bx_df = read_data(extracted_data_dir, min_rating=min_rating)
  logging.info("Generating book rating user timelines.")
  timelines, book_counts = convert_to_timelines(ratings_df)
  logging.info("Generating train and test examples.")
  train_examples, test_examples = generate_examples_from_timelines(
      timelines=timelines,
      bx_df=bx_df,
      min_timeline_len=min_timeline_length,
      max_context_len=max_context_length,
      train_data_fraction=train_data_fraction)

  if not tf.io.gfile.exists(output_dir):
    tf.io.gfile.makedirs(output_dir)
  logging.info("Writing generated training examples.")
  train_file = os.path.join(output_dir, train_filename)
  train_size = write_tfrecords(tf_examples=train_examples, filename=train_file)
  logging.info("Writing generated testing examples.")
  test_file = os.path.join(output_dir, test_filename)
  test_size = write_tfrecords(tf_examples=test_examples, filename=test_file)
  stats = {
      "train_size": train_size,
      "test_size": test_size,
      "train_file": train_file,
      "test_file": test_file,
  }

  if build_vocabs:
    (book_vocab, book_year_vocab, book_author_vocab) = (
        generate_bx_feature_vocabs(
            bx_df=bx_df, book_counts=book_counts))
    vocab_file = os.path.join(output_dir, vocab_filename)
    write_vocab_json(book_vocab, filename=vocab_file)
    stats.update({
        "vocab_size": len(book_vocab),
        "vocab_file": vocab_file,
        "vocab_max_id": max([arr[VOCAB_BX_ID_INDEX] for arr in book_vocab])
    })

    for vocab, filename, key in zip([book_year_vocab, book_author_vocab],
                                    [vocab_year_filename, vocab_author_filename],
                                    ["year_vocab", "author_vocab"]):
      vocab_file = os.path.join(output_dir, filename)
      write_vocab_txt(vocab, filename=vocab_file)
      stats.update({
          key + "_size": len(vocab),
          key + "_file": vocab_file,
      })

  return stats


def main(_):
  logging.info("Downloading and extracting data.")
  extracted_data_dir = download_and_extract_data(data_directory=FLAGS.data_dir)

  stats = generate_datasets(
      extracted_data_dir=extracted_data_dir,
      output_dir=FLAGS.output_dir,
      min_timeline_length=FLAGS.min_timeline_length,
      max_context_length=FLAGS.max_context_length,
      min_rating=FLAGS.min_rating,
      build_vocabs=FLAGS.build_vocabs,
      train_data_fraction=FLAGS.train_data_fraction,
  )
  logging.info("Generated dataset: %s", stats)


if __name__ == "__main__":
  define_flags()
  app.run(main)
