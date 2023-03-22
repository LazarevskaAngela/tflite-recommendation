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
"""Tests for example_generation_bx."""

import pandas as pd
import tensorflow as tf
from data import example_generation_bx as example_gen

from google.protobuf import text_format


BOOKS_DF = pd.DataFrame([
    {
        'ISBN': '0195153448',
        'Book-Title': 'Classical Mythology',
        'Book-Author': 'Mark P. O. Morford',
        'Year-Of-Publication': int(2002),
        'Publisher': 'Oxford University Press'
    },
    {
        'ISBN': '0002005018',
        'Book-Title': 'Clara Callan',
        'Book-Author': 'Richard Bruce Wright',
        'Year-Of-Publication': int(2001),
        'Publisher': 'HarperFlamingo Canada'
    },
    {
        'ISBN': '0060973129',
        'Book-Title': 'Decision in Normandy',
        'Book-Author': 'Carlo D\'Este',
        'Year-Of-Publication': int(1991),
        'Publisher': 'HarperPerennial'
    },
    {
        'ISBN': '0374157065',
        'Book-Title': 'Flu: The Story of the Great Influenza Pandemic of 1918 and the Search for the Virus That Caused It',
        'Book-Author': 'Gina Bari Kolata',
        'Year-Of-Publication': int(1999),
        'Publisher': 'Farrar Straus Giroux'
    },
    {
        'ISBN': '0393045218',
        'Book-Title': 'The Mummies of Urumchi',
        'Book-Author': 'E. J. W. Barber',
        'Year-Of-Publication': int(1999),
        'Publisher': 'W. W. Norton &amp; Company'
    },
])

RATINGS_DF = pd.DataFrame([
    {
        'User-ID': int(276725),
        'ISBN': '034545104X',
        'Book-Rating': int(0)
    },
    {
        'User-ID': int(276726),
        'ISBN': '0155061224',
        'Book-Rating': int(5)
    },
    {
        'User-ID': int(276727),
        'ISBN': '0446520802',
        'Book-Rating': int(0)
    },
    {
        'User-ID': int(276729),
        'ISBN': '052165615X',
        'Book-Rating': int(3)
    },
    {
        'User-ID': int(276729),
        'ISBN': '0521795028',
        'Book-Rating': int(6)
    },
])

EXAMPLE1 = text_format.Parse(
    """
    features {
        feature {
          key: "context_isbn"
          value {
            bytes_list {
              value: ["034545104X", "0155061224", "0446520802", "052165615X", "0521795028"]
            }
          }
        }
        feature {
          key: "context_book_rating"
          value {
            float_list {
              value: [0, 5, 0, 3, 6]
            }
          }
        }
        feature {
          key: "context_book_author"
          value {
            bytes_list {
              value: ["Mark P. O. Morford", "Richard Bruce Wright", "Carlo D'Este", "Gina Bari Kolata"]
            }
          }
        }
        feature {
          key: "context_book_year"
          value {
            int64_list {
              value: [2002, 2001, 1991, 1999, 1999]
            }
          }
        }
      }
      """, tf.train.Example())

EXAMPLE2 = text_format.Parse(
    """
    features {
        feature {
          key: "context_isbn"
          value {
            bytes_list {
              value: ["034545104X", "0155061224", "0446520802", "052165615X", "0521795028"]
            }
          }
        }
        feature {
          key: "context_book_rating"
          value {
            float_list {
              value: [0, 1, 0, 1, 2]
            }
          }
        }
        feature {
          key: "context_book_author"
          value {
            bytes_list {
              value: ["Mark P. O. Morford", "Richard Bruce Wright", "Carlo D'Este", "Gina Bari Kolata"]
            }
          }
        }
        feature {
          key: "context_book_year"
          value {
            int64_list {
              value: [1995, 1999, 2001, 1999, 1999]
            }
          }
        }
      }
      """, tf.train.Example())

EXAMPLE3 = text_format.Parse(
    """
        features {
        feature {
          key: "context_isbn"
          value {
            bytes_list {
              value: ["034545104X", "0155061224", "0446520802", "052165615X", "0521795028"]
            }
          }
        }
        feature {
          key: "context_book_rating"
          value {
            float_list {
              value: [9, 8, 7, 6, 8]
            }
          }
        }
        feature {
          key: "context_book_author"
          value {
            bytes_list {
              value: ["Mark P. O. Morford", "Richard Bruce Wright", "Carlo D'Este", "Gina Bari Kolata"]
            }
          }
        }
        feature {
          key: "context_book_year"
          value {
            int64_list {
              value: [2001, 2002, 2001, 2002, 1999]
            }
          }
        }
      }
      """, tf.train.Example())


class ExampleGenerationBXTest(tf.test.TestCase):

  def test_example_generation(self):
    timelines, _ = example_gen.convert_to_timelines(RATINGS_DF)
    train_examples, test_examples = example_gen.generate_examples_from_timelines(
        timelines=timelines,
        bx_df=BX_DF,
        min_timeline_len=2,
        max_context_len=5,
        train_data_fraction=0.66,
        shuffle=False)
    self.assertLen(train_examples, 2)
    self.assertLen(test_examples, 1)
    self.assertProtoEquals(train_examples[0], EXAMPLE1)
    self.assertProtoEquals(train_examples[1], EXAMPLE2)
    self.assertProtoEquals(test_examples[0], EXAMPLE3)

  def test_vocabs_generation(self):
    _, bx_counts = example_gen.convert_to_timelines(RATINGS_DF)
    (_, book_year_vocab, book_author_vocab) = (
        example_gen.generate_bx_feature_vocabs(
            bx_df=BX_DF, bx_counts=bx_counts))
    self.assertAllEqual(book_year_vocab, [1991, 2000, 1995])
    self.assertAllEqual(book_author_vocab, [
        'Mark P. O. Morford', 'Richard Bruce Wright'
    ])


if __name__ == '__main__':
  tf.test.main()
