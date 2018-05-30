# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Data generators for SearchQA (https://github.com/nyu-dl/SearchQA).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import collections
import os
import zipfile

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import tokenizer
from tensor2tensor.utils import registry

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS


_DRIVE_URL = 'https://drive.google.com/file/d/0B51lBZ1gs1XTR3BIVTJQWkREQU0'
_FILENAME = 'SearchQA'
_UNK = "<UNK>"
PAD = text_encoder.PAD_ID


def _normalize_string(raw_str):
  """Normalizes the string using tokenizer.encode.

  Args:
    raw_str: the input string

  Returns:
   A string which is ready to be tokenized using split()
  """
  return ' '.join(
      token.strip()
      for token in tokenizer.encode(text_encoder.native_to_unicode(raw_str)))


def _build_vocab(generator, vocab_dir, vocab_name, vocab_size):
  """Build a vocabulary from examples.

  Args:
    generator: text generator for creating vocab.
    vocab_dir: directory where to save the vocabulary.
    vocab_name: vocab file name.

  Returns:
    text encoder.
  """
  vocab_path = os.path.join(vocab_dir, vocab_name)
  if not tf.gfile.Exists(vocab_path):
    data = []
    for line in generator:
      data.extend(line.split())
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    words = [_UNK] + list(words[:vocab_size])
    encoder = text_encoder.TokenTextEncoder(None, vocab_list=words)
    encoder.store_to_file(vocab_path)
  else:
    encoder = text_encoder.TokenTextEncoder(vocab_path)
  return encoder



class FeatureNames(object):
  """Feature names, i.e keys for storing SearchQa data in TFExamples."""
  SNIPPETS = 'snippets'
  QUESTION = 'question'
  ANSWER = 'answer'

  @classmethod
  def features(cls):
    for attr, value in cls.__dict__.items():
      if not attr.startswith('__') and not callable(getattr(cls, attr)):
        yield value



def _parse_and_generate_searchqa_sampls(dataset_file):

  with tf.gfile.GFile(dataset_file, mode='r') as fp:
    for line in fp:
      example = line.strip().split('|||')
      question = example[1].strip()
      answer = example[2].strip()
      if len(question) > 0 and len(answer) > 0:
        snippets = example[0].strip()[4:-4].split('</s>  <s>')
        assert (len(snippets) != 0)
        for s in snippets:
          assert (len(s) > 0)
        yield {
            FeatureNames.QUESTION: question,
            FeatureNames.ANSWER: answer,
            FeatureNames.SNIPPETS: snippets
        }


def generate_text_for_vocab(raw_data_path):
  for example in _parse_and_generate_searchqa_sampls(raw_data_path):
    yield ' '.join(' '.join(example[FeatureNames.SNIPPETS]).split())
    yield ' '.join(example[FeatureNames.QUESTION].split())
    yield ' '.join(example[FeatureNames.ANSWER].split())


def _prepare_serchqa_data(tmp_dir):
  file_path = generator_utils.maybe_download_from_drive(tmp_dir,
                                                        _FILENAME + '.zip',
                                                        _DRIVE_URL)
  try:
    zip_ref = zipfile.ZipFile(file_path, 'r')
    zip_ref.extractall(tmp_dir)
    zip_ref.close()

  except zipfile.BadZipfile:
    tf.logging.error("Please dowload the file 'SearchQA.zip' to the tmp_dir "
                     "through address: "
                     "https://drive.google.com/open?id=0B51lBZ1gs1XTR3BIVTJQWkREQU0")
    raise zipfile.BadZipfile

  return os.path.join(tmp_dir, _FILENAME)

@registry.register_problem
class SearchQa(text_problems.QuestionAndContext2TextProblem):
  """Base class for SearchQa question answering problem."""

  @property
  def is_generate_per_split(self):
    return True

  @property
  def dataset_splits(self):
    return [{
        'split': problem.DatasetSplit.TRAIN,
        'shards': 10,
    },{
        'split': problem.DatasetSplit.EVAL,
        'shards': 1,
    },{
        'split': problem.DatasetSplit.TEST,
        'shards': 1,
      }
    ]

  @property
  def vocab_type(self):
    return text_problems.VocabType.SUBWORD

  # if VocabType is SUBWORD
  @property
  def approx_vocab_size(self):
    return 2 ** 16  # ~65k


  # if VocabType is Token
  @property
  def vocab_size(self):
    #
    return 2 ** 16  # ~65k

  # if VocabType is Token
  @property
  def oov_token(self):
    return _UNK


  def generate_samples(self, data_dir, tmp_dir, dataset_split):

    raw_data_dir = _prepare_serchqa_data(tmp_dir)

    train_file = os.path.join(raw_data_dir, "train.txt")
    eval_file = os.path.join(raw_data_dir, "val.tx")
    test_file = os.path.join(raw_data_dir, "test.txt")

    if dataset_split == problem.DatasetSplit.TRAIN:
      dataset_file = train_file
      _build_vocab(generate_text_for_vocab(dataset_file),
        data_dir, self.vocab_filename, self.vocab_size)

    elif dataset_split == problem.DatasetSplit.EVAL:
      dataset_file = eval_file

    elif dataset_split == problem.DatasetSplit.TEST:
      dataset_file = test_file

    def _generator():
      for example in _parse_and_generate_searchqa_sampls(dataset_file):
        yield {
          'input': example[FeatureNames.QUESTION],
          'target': example[FeatureNames.ANSWER],
          'context' : ' '.join(example[FeatureNames.SNIPPETS]),
        }

    return _generator()




@registry.register_problem
class SearchQaConcat(SearchQa):
  """Searchqa with snipptes and question concatenated together in inputs."""

  def dataset_filename(self):
    return 'search_qa'

  def preprocess_example(self, example, unused_mode, unused_model_hparams):
    sep = tf.convert_to_tensor([self.QUESTION_SEPARATOR_ID],
                               dtype=example[FeatureNames.QUESTION].dtype)
    example['inputs'] = tf.concat(
        [example[FeatureNames.SNIPPETS], sep, example[FeatureNames.QUESTION]], 0)
    return example

  def hparams(self, defaults, unused_model_hparams):
    (super(SearchQa, self)
     .hparams(defaults, unused_model_hparams))
    p = defaults
    del p.input_modality[FeatureNames.SNIPPETS]


@registry.register_problem
class SearchQaSnippets(text_problems.Text2TextProblem):
  """Base class for SearchQa question answering problems."""
  def __init__(self, *args, **kwargs):

    super(SearchQaSnippets, self).__init__(*args, **kwargs)

    self.max_snippet_length = None
    self.max_search_results_length = None
    self.max_question_length = None
    self.max_answer_length = None

    data_dir = os.path.expanduser(FLAGS.data_dir)
    metadata_path = os.path.join(data_dir, "meta_data.json")

    if tf.gfile.Exists(metadata_path):
      with tf.gfile.GFile(metadata_path, mode='r') as f:
        metadata = json.load(f)

      self.max_snippet_length = metadata['max_snippet_length']
      self.max_search_results_length = metadata['max_search_results_length']
      self.max_question_length = metadata['max_question_length']
      self.max_answer_length = metadata['max_answer_length']

    assert not self._was_reversed, 'This problem is not reversible!'
    assert not self._was_copy, 'This problem is not copyable!'

  @property
  def is_generate_per_split(self):
    return True

  @property
  def dataset_splits(self):
    return [{
      'split': problem.DatasetSplit.TRAIN,
      'shards': 10,
    }, {
      'split': problem.DatasetSplit.EVAL,
      'shards': 1,
    }, {
      'split': problem.DatasetSplit.TEST,
      'shards': 1,
    }
    ]

  @property
  def vocab_type(self):
    return text_problems.VocabType.TOKEN

  @property
  def vocab_size(self):
    return 2 ** 16  # ~65k


  @property
  def oov_token(self):
    return _UNK


  @property
  def truncated_search_results(self):
      return 300


  @property
  def num_train_shards(self):
    return self.dataset_splits[0]["shards"]


  @property
  def num_dev_shards(self):
    return self.dataset_splits[1]["shards"]


  @property
  def vocab_filename(self):
      return "vocab.%s.%s" % (self.dataset_filename(),
                              text_problems.VocabType.TOKEN)


  def generate_data(self, data_dir, tmp_dir, task_id=-1):

    raw_data_dir = _prepare_serchqa_data(tmp_dir)
    metadata_path = os.path.join(data_dir, "meta_data.json")

    train_file = os.path.join(raw_data_dir, "train.txt")
    dev_file = os.path.join(raw_data_dir, "val.txt")
    test_file = os.path.join(raw_data_dir, "test.txt")

    _build_vocab(generate_text_for_vocab(train_file),
                 data_dir, self.vocab_filename, self.vocab_size)

    encoder = self.get_or_create_vocab(data_dir, tmp_dir)
    self._extract_searchqa_metadata(encoder,
                                    [train_file, dev_file, test_file],
                                    metadata_path)

    filepath_fns = {
        problem.DatasetSplit.TRAIN: self.training_filepaths,
        problem.DatasetSplit.EVAL: self.dev_filepaths,
        problem.DatasetSplit.TEST: self.test_filepaths,
    }

    split_paths = [(split["split"], filepath_fns[split["split"]](
        data_dir, split["shards"], shuffled=False))
                   for split in self.dataset_splits]
    all_paths = []
    for _, paths in split_paths:
      all_paths.extend(paths)

    for split, paths in split_paths:
      generator_utils.generate_files(
          self._maybe_pack_examples(
              self.generate_encoded_samples(data_dir, tmp_dir, split,
                                            encoder)), paths)


  def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split, encoder):
    """Reads examples and encodes them using the given encoders.

    Args:
      examples: all the examples in the data parsed by the dataset parser

    Yields:
      tf_examples that are encoded based ont the given encoders
    """

    def pad_input(snippets, question, answer):
      'Pad snippets, stories, and queries to a consistence length.'
      for snippet in snippets:
        for _ in range(self.max_snippet_length - len(snippet)):
          snippet.append(PAD)
        assert len(snippet) == self.max_snippet_length

      for _ in range(self.max_search_results_length - len(snippets)):
        snippets.append([PAD for _ in range(self.max_snippet_length)])

      for _ in range(self.max_question_length - len(question)):
        question.append(PAD)

      for _ in range(self.max_answer_length - len(answer)):
        answer.append(PAD)

      assert len(snippets) == self.max_search_results_length
      assert len(question) == self.max_question_length
      assert len(answer) == self.max_answer_length

      return snippets, question, answer


    raw_data_dir = _prepare_serchqa_data(tmp_dir)

    train_file = os.path.join(raw_data_dir, "train.txt")
    eval_file = os.path.join(raw_data_dir, "val.tx")
    test_file = os.path.join(raw_data_dir, "test.txt")

    if dataset_split == problem.DatasetSplit.TRAIN:
      dataset_file = train_file

    elif dataset_split == problem.DatasetSplit.EVAL:
      dataset_file = eval_file

    elif dataset_split == problem.DatasetSplit.TEST:
      dataset_file = test_file

    for example in _parse_and_generate_searchqa_sampls(dataset_file):
      example = self.encode_example(example, encoder)
      snippets, question, answer = pad_input(example[FeatureNames.SNIPPETS],
                                             example[FeatureNames.QUESTION],
                                             example[FeatureNames.ANSWER])

      snippets_flat = [token_id for snippet in snippets for token_id in
                       snippet]

      yield {
        FeatureNames.SNIPPETS: snippets_flat,
        FeatureNames.QUESTION: question,
        FeatureNames.ANSWER: answer
      }

  def searchqa_generate_encoded(self,
                                 sample_generator,
                                 vocab,
                                 targets_vocab=None,
                                 has_inputs=True):
    """Encode Text2Text samples from the generator with the vocab."""
    targets_vocab = targets_vocab or vocab
    for sample in sample_generator:
      if has_inputs:
        sample["inputs"] = vocab.encode(sample["inputs"])
        sample["inputs"].append(text_encoder.EOS_ID)
      sample["targets"] = targets_vocab.encode(sample["targets"])
      sample["targets"].append(text_encoder.EOS_ID)
      yield sample



  def encode_example(self,
                       example,
                       encoder,
                       snippet_length=None,
                       search_results_length=None,
                       question_length=None,
                       answer_length=None):
    def truncate_results(results):
      'Truncate snippets to the specified maximum length.'
      return results[-self.truncated_search_results:]

    snippets = [encoder.encode(snippet) for snippet in
                truncate_results(example[FeatureNames.SNIPPETS])]
    question = encoder.encode(example[FeatureNames.QUESTION])
    answer = encoder.encode(example[FeatureNames.ANSWER])
    example[FeatureNames.SNIPPETS] = snippets
    example[FeatureNames.QUESTION] = question
    example[FeatureNames.ANSWER] = answer

    if not (snippet_length is None or search_results_length is None or
        question_length is None or answer_length is None):
      # The function is called for metadata extraction.
      snippet_length.extend([len(snippet) for snippet in snippets])
      search_results_length.append(len(snippets))
      question_length.append(len(question))
      answer_length.append(len(answer))
      return (snippet_length, search_results_length,
              question_length, answer_length)

    return example

  def _extract_searchqa_metadata(self, encoder, dataset_files, metadata_path):

    snippet_length = []
    search_results_length = []
    question_length = []
    answer_length = []

    for dataset_file in dataset_files:
      for example in _parse_and_generate_searchqa_sampls(dataset_file):
        (snippet_length, search_results_length,
         question_length, answer_length) = self.encode_example(example, encoder,
                                                           snippet_length,
                                                           search_results_length,
                                                           question_length,
                                                           answer_length)
    self.max_snippet_length = max(snippet_length)
    self.max_search_results_length = max(search_results_length)
    self.max_question_length = max(question_length)
    self.max_answer_length = max(answer_length)

    with tf.gfile.Open(metadata_path, 'w') as f:
      f.write(json.dumps({
        'max_snippet_length': self.max_snippet_length,
        'max_search_results_length': self.max_search_results_length,
        'max_question_length': self.max_question_length,
        'max_answer_length': self.max_answer_length,
      }))


  def example_reading_spec(self):
    """Specify the names and types of the features on disk.

    Returns:
      The names and type of features.
    """
    data_fields = {
      FeatureNames.SNIPPETS: tf.FixedLenFeature(
        shape=[self.max_search_results_length, self.max_snippet_length], dtype=tf.int64),
      FeatureNames.QUESTION: tf.FixedLenFeature(
        shape=[1, self.max_question_length], dtype=tf.int64),
      FeatureNames.ANSWER: tf.FixedLenFeature(
        shape=[1, self.max_answer_length], dtype=tf.int64)}

    data_items_to_decoders = None
    return (data_fields, data_items_to_decoders)


  def preprocess_example(self, example, mode, unused_hparams):
    """Preprocesses the example feature dict.

    Args:
      example: input example
      mode: training, eval, and inference
      unused_hparams: -

    Returns:
      The processed example
    """
    # add feature 'targets' to the example which is equal to Answer
    example['targets'] = example[FeatureNames.ANSWER]
    # In T2T, features are supposed to enter the pipeline as 3d tensors.
    # "inputs" and "targets" will be expended to 3d if they're not,
    # and we should expand other features if we define any
    example[FeatureNames.SNIPPETS] = tf.expand_dims(
      example[FeatureNames.SNIPPETS], -1)
    example[FeatureNames.QUESTION] = tf.expand_dims(
        example[FeatureNames.QUESTION], -1)
    tf.logging.info(example[FeatureNames.QUESTION])
    return example

  def feature_encoders(self, data_dir):
    """Determines how features from each example should be encoded.

    Args:
      data_dir: The base directory where data and vocab files are stored.

    Returns:
      A dict of <feature name, Encoder> for encoding and decoding inference
       input/output.
    """
    assert self.vocab_type == text_problems.VocabType.TOKEN

    vocab_filename = os.path.join(data_dir, self.vocab_filename)
    encoder = text_encoder.TokenTextEncoder(vocab_filename,
                                            replace_oov=self.oov_token)


    return {FeatureNames.SNIPPETS: encoder,
            FeatureNames.QUESTION: encoder,
            'targets': encoder}

  def hparams(self, defaults, unused_model_hparams):
    """Defines model hyperparameters.

    Args:
      defaults: default hparams
      unused_model_hparams: -

    """
    p = defaults
    p.stop_at_eos = int(True)

    snippets_vocab_size = self._encoders[FeatureNames.SNIPPETS].vocab_size
    question_vocab_size = self._encoders[FeatureNames.QUESTION].vocab_size

    p.input_modality = {
      FeatureNames.SNIPPETS: (registry.Modalities.SYMBOL, snippets_vocab_size),
      FeatureNames.QUESTION: (
      registry.Modalities.SYMBOL, question_vocab_size)}

    target_vocab_size = self._encoders["targets"].vocab_size
    p.target_modality = (registry.Modalities.SYMBOL, target_vocab_size)
