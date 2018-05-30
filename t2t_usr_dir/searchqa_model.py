"""Models for SearchQA tasks.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import common_attention
from tensor2tensor.models.transformer import features_to_nonpadding
from tensor2tensor.utils import beam_search
from tensor2tensor.models import transformer
from . import searchqa_problem

import tensorflow as tf

FLAGS = tf.flags.FLAGS


# ============================================================================
# Transformer-base models
# ============================================================================
@registry.register_model
class SearchqaTransformer(transformer.Transformer):
  
  @property
  def has_input(self):
    return True

  def estimator_spec_predict(self, features):
    """Construct EstimatorSpec for PREDICT mode."""
    decode_hparams = self._decode_hparams
    infer_out = self.infer(features, beam_size=decode_hparams.beam_size,
                           top_beams=(
                             decode_hparams.beam_size if decode_hparams.return_beams else 1),
                           alpha=decode_hparams.alpha,
                           decode_length=decode_hparams.extra_length)
    if isinstance(infer_out, dict):
      outputs = infer_out["outputs"]
      scores = infer_out["scores"]
    else:
      outputs = infer_out
      scores = None

    batch_size = common_layers.shape_list(
      features[searchqa_problem.FeatureNames.SNIPPETS])[0]
    batched_problem_choice = (features["problem_choice"] * tf.ones(
      (batch_size,), dtype=tf.int32))
    predictions = {
      "outputs": outputs,
      "scores": scores,
      searchqa_problem.FeatureNames.SNIPPETS: features.get(searchqa_problem.FeatureNames.SNIPPETS),
      searchqa_problem.FeatureNames.QUESTION: features.get(
        searchqa_problem.FeatureNames.QUESTION),
      "targets": features.get("infer_targets"),
      "problem_choice": batched_problem_choice,
    }
    t2t_model._del_dict_nones(predictions)

    export_out = {"outputs": predictions["outputs"]}
    if "scores" in predictions:
      export_out["scores"] = predictions["scores"]

    return tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.PREDICT,
                                      predictions=predictions,
                                      export_outputs={
                                        "output": tf.estimator.export.PredictOutput(
                                          export_out)})

  def _slow_greedy_infer(self, features, decode_length):
    """A slow greedy inference method.

    Quadratic time in decode_length.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": None
          "logits": `Tensor` of shape [batch_size, time, 1, 1, vocab_size].
          "losses": a dictionary: {loss-name (string): floating point `Scalar`}
      }
    """
    if not features:
      features = {}

    # todo(dehghani): remove dim-expansion and check
    story_old = None
    if len(features[searchqa_problem.FeatureNames.SNIPPETS].shape) < 4:
      story_old = features[searchqa_problem.FeatureNames.SNIPPETS]
      features[searchqa_problem.FeatureNames.SNIPPETS] = tf.expand_dims(
        features[searchqa_problem.FeatureNames.SNIPPETS], 2)

    question_old = None
    if len(features[searchqa_problem.FeatureNames.QUESTION].shape) < 4:
      question_old = features[searchqa_problem.FeatureNames.QUESTION]
      features[searchqa_problem.FeatureNames.QUESTION] = tf.expand_dims(
        features[searchqa_problem.FeatureNames.QUESTION], 2)

    targets_old = features.get("targets", None)
    target_modality = self._problem_hparams.target_modality

    def infer_step(recent_output, recent_logits, unused_loss):
      """Inference step."""
      if not tf.contrib.in_eager_mode():
        recent_output.set_shape([None, None, None, 1])
      padded = tf.pad(recent_output, [[0, 0], [0, 1], [0, 0], [0, 0]])
      features["targets"] = padded
      # This is inefficient in that it generates samples at all timesteps,
      # not just the last one, except if target_modality is pointwise.
      samples, logits, losses = self.sample(features)
      # Concatenate the already-generated recent_output with last timestep
      # of the newly-generated samples.
      if target_modality.top_is_pointwise:
        cur_sample = samples[:, -1, :, :]
      else:
        cur_sample = samples[:, common_layers.shape_list(recent_output)[1], :,
                     :]
      cur_sample = tf.to_int64(tf.expand_dims(cur_sample, axis=1))
      samples = tf.concat([recent_output, cur_sample], axis=1)
      if not tf.contrib.in_eager_mode():
        samples.set_shape([None, None, None, 1])

      # Assuming we have one shard for logits.
      logits = tf.concat([recent_logits, logits[:, -1:]], 1)
      loss = sum([l for l in losses.values() if l is not None])
      return samples, logits, loss

    # Create an initial output tensor. This will be passed
    # to the infer_step, which adds one timestep at every iteration.
    if "partial_targets" in features:
      initial_output = tf.to_int64(features["partial_targets"])
      while len(initial_output.get_shape().as_list()) < 4:
        initial_output = tf.expand_dims(initial_output, 2)
      batch_size = common_layers.shape_list(initial_output)[0]
    else:
      batch_size = common_layers.shape_list(
        features[searchqa_problem.FeatureNames.SNIPPETS])[0]

      initial_output = tf.zeros((batch_size, 0, 1, 1), dtype=tf.int64)
    # Hack: foldl complains when the output shape is less specified than the
    # input shape, so we confuse it about the input shape.
    initial_output = tf.slice(initial_output, [0, 0, 0, 0],
                              common_layers.shape_list(initial_output))
    target_modality = self._problem_hparams.target_modality
    if target_modality.is_class_modality:
      decode_length = 1
    else:
      decode_length = (common_layers.shape_list(
        features[searchqa_problem.FeatureNames.SNIPPETS])[1] +
                       common_layers.shape_list(
                         features[searchqa_problem.FeatureNames.QUESTION])[1] +
                       decode_length)
    # Initial values of result, logits and loss.
    result = initial_output
    # tensor of shape [batch_size, time, 1, 1, vocab_size]
    logits = tf.zeros((batch_size, 0, 1, 1, target_modality.top_dimensionality))
    if not tf.contrib.in_eager_mode():
      logits.set_shape([None, None, None, None, None])
    loss = 0.0

    def while_exit_cond(result, logits,
                        loss):  # pylint: disable=unused-argument
      """Exit the loop either if reach decode_length or EOS."""
      length = common_layers.shape_list(result)[1]

      not_overflow = length < decode_length

      if self._problem_hparams.stop_at_eos:
        def fn_not_eos():
          return tf.not_equal(  # Check if the last predicted element is a EOS
            tf.squeeze(result[:, -1, :, :]), text_encoder.EOS_ID)

        not_eos = tf.cond(
          # We only check for early stoping if there is at least 1 element (
          # otherwise not_eos will crash)
          tf.not_equal(length, 0), fn_not_eos, lambda: True, )

        return tf.cond(tf.equal(batch_size, 1),
                       # If batch_size == 1, we check EOS for early stoping
                       lambda: tf.logical_and(not_overflow, not_eos),
                       # Else, just wait for max length
                       lambda: not_overflow)
      return not_overflow

    result, logits, loss = tf.while_loop(while_exit_cond, infer_step,
                                         [result, logits, loss],
                                         shape_invariants=[tf.TensorShape(
                                           [None, None, None, None]),
                                                           tf.TensorShape(
                                                             [None, None, None,
                                                              None, None]),
                                                           tf.TensorShape(
                                                             []), ],
                                         back_prop=False, parallel_iterations=1)
    if story_old is not None:  # Restore to not confuse Estimator.
      features[searchqa_problem.FeatureNames.SNIPPETS] = story_old
    if question_old is not None:  # Restore to not confuse Estimator.
      features[searchqa_problem.FeatureNames.QUESTION] = question_old
    # Reassign targets back to the previous value.
    if targets_old is not None:
      features["targets"] = targets_old
    losses = {"training": loss}
    if "partial_targets" in features:
      partial_target_length = \
        common_layers.shape_list(features["partial_targets"])[1]
      result = tf.slice(result, [0, partial_target_length, 0, 0],
                        [-1, -1, -1, -1])
    return {"outputs": result, "scores": None, "logits": logits,
            "losses": losses, }

  def _beam_decode_slow(self, features, decode_length, beam_size, top_beams,
                        alpha):
    """Slow version of Beam search decoding.

    Quadratic time in decode_length.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for slonger translations.

    Returns:
       samples: an integer `Tensor`. Top samples from the beam search
    """
    batch_size = common_layers.shape_list(
      features[searchqa_problem.FeatureNames.SNIPPETS])[0]

    def symbols_to_logits_fn(ids):
      """Go from ids to logits."""
      ids = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
      ids = tf.pad(ids[:, 1:], [[0, 0], [0, 1], [0, 0], [0, 0]])
      if "partial_targets" in features:
        pt = features["partial_targets"]
        pt_length = common_layers.shape_list(pt)[1]
        pt = tf.tile(pt, [1, beam_size])
        pt = tf.reshape(pt, [batch_size * beam_size, pt_length, 1, 1])
        ids = tf.concat([pt, ids], axis=1)

      features["targets"] = ids
      self._coverage = None
      logits, _ = self(features)  # pylint: disable=not-callable
      # now self._coverage is a coverage tensor for the first datashard.
      # it has shape [batch_size] and contains floats between 0 and
      # source_length.
      if self._problem_hparams:
        modality = self._problem_hparams.target_modality
        if modality.top_is_pointwise:
          return tf.squeeze(logits, axis=[1, 2, 3])
      # -1 due to the pad above.
      current_output_position = common_layers.shape_list(ids)[1] - 1
      logits = logits[:, current_output_position, :, :]
      return tf.squeeze(logits, axis=[1, 2])

    initial_ids = tf.zeros([batch_size], dtype=tf.int32)

    if self.has_input:
      story_old = features[searchqa_problem.FeatureNames.SNIPPETS]
      question_old = features[searchqa_problem.FeatureNames.QUESTION]

      features[searchqa_problem.FeatureNames.SNIPPETS] = tf.expand_dims(
        features[searchqa_problem.FeatureNames.SNIPPETS], 1)

      features[searchqa_problem.FeatureNames.QUESTION] = tf.expand_dims(
        features[searchqa_problem.FeatureNames.QUESTION], 1)

      if len(features[searchqa_problem.FeatureNames.SNIPPETS].shape) < 5:
        features[searchqa_problem.FeatureNames.SNIPPETS] = tf.expand_dims(
          features[searchqa_problem.FeatureNames.SNIPPETS], 4)

      if len(features[searchqa_problem.FeatureNames.QUESTION].shape) < 5:
        features[searchqa_problem.FeatureNames.QUESTION] = tf.expand_dims(
          features[searchqa_problem.FeatureNames.QUESTION], 4)

      # Expand the inputs in to the beam size.
      features[searchqa_problem.FeatureNames.SNIPPETS] = tf.tile(
        features[searchqa_problem.FeatureNames.SNIPPETS], [1, beam_size, 1, 1, 1])

      features[searchqa_problem.FeatureNames.QUESTION] = tf.tile(
        features[searchqa_problem.FeatureNames.QUESTION], [1, beam_size, 1, 1, 1])

      s = common_layers.shape_list(features[searchqa_problem.FeatureNames.SNIPPETS])
      features[searchqa_problem.FeatureNames.SNIPPETS] = tf.reshape(
        features[searchqa_problem.FeatureNames.SNIPPETS], [s[0] * s[1], s[2], s[3], s[4]])

      s = common_layers.shape_list(features[searchqa_problem.FeatureNames.QUESTION])
      features[searchqa_problem.FeatureNames.QUESTION] = tf.reshape(
        features[searchqa_problem.FeatureNames.QUESTION],
        [s[0] * s[1], s[2], s[3], s[4]])

    target_modality = self._problem_hparams.target_modality
    vocab_size = target_modality.top_dimensionality
    # Setting decode length to input length + decode_length
    decode_length = tf.constant(decode_length)
    if "partial_targets" not in features:
      decode_length += common_layers.shape_list(
        features[searchqa_problem.FeatureNames.SNIPPETS])[1] + common_layers.shape_list(
        features[searchqa_problem.FeatureNames.QUESTION])[1]

    ids, scores = beam_search.beam_search(
      symbols_to_logits_fn,
      initial_ids,
      beam_size,
      decode_length,
      vocab_size,
      alpha,
      stop_early=(top_beams == 1))

    # Set inputs back to the unexpanded inputs to not to confuse the Estimator!
    if self.has_input:
      features[searchqa_problem.FeatureNames.SNIPPETS] = story_old
      features[searchqa_problem.FeatureNames.QUESTION] = question_old

    # Return `top_beams` decodings (also remove initial id from the beam search)
    if top_beams == 1:
      samples = ids[:, 0, 1:]
    else:
      samples = ids[:, :top_beams, 1]

    return {"outputs": samples, "scores": scores}

  def _fast_decode(self, features, decode_length, beam_size=1, top_beams=1,
                   alpha=1.0):
    """Fast decoding.

    Implements both greedy and beam search decoding, uses beam search iff
    beam_size > 1, otherwise beam search related arguments are ignored.

    Args:
      features: a map of string to model  features.
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha,
      stronger
        the preference for slonger translations.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }

    Raises:
      NotImplementedError: If there are multiple data shards.
    """
    if self._num_datashards != 1:
      raise NotImplementedError("Fast decoding only supports a single shard.")
    dp = self._data_parallelism
    hparams = self._hparams
    target_modality = self._problem_hparams.target_modality

    story = features[searchqa_problem.FeatureNames.SNIPPETS]
    question = features[searchqa_problem.FeatureNames.QUESTION]

    if target_modality.is_class_modality:
      decode_length = 1
    else:
      decode_length = (common_layers.shape_list(story)[1] +
                       common_layers.shape_list(question)[1] + decode_length)

    story = tf.expand_dims(story, axis=1)
    question = tf.expand_dims(question, axis=1)

    if len(story.shape) < 5:
      story = tf.expand_dims(story, axis=4)

    if len(question.shape) < 5:
      question = tf.expand_dims(question, axis=4)

    s = common_layers.shape_list(story)
    batch_size = s[0]
    story = tf.reshape(story, [s[0] * s[1], s[2], s[3], s[4]])

    s = common_layers.shape_list(question)
    batch_size = s[0]

    question = tf.reshape(question, [s[0] * s[1], s[2], s[3], s[4]])

    # _shard_features called to ensure that the variable names match
    story = self._shard_features({searchqa_problem.FeatureNames.SNIPPETS: story}
                                 )[searchqa_problem.FeatureNames.SNIPPETS]

    question = self._shard_features({searchqa_problem.FeatureNames.QUESTION: question}
                                    )[searchqa_problem.FeatureNames.QUESTION]

    story_modality = self._problem_hparams.input_modality[
      searchqa_problem.FeatureNames.SNIPPETS]
    question_modality = self._problem_hparams.input_modality[
      searchqa_problem.FeatureNames.QUESTION]

    with tf.variable_scope(story_modality.name):
      story = story_modality.bottom_sharded(story, dp)

    with tf.variable_scope(question_modality.name,
                           reuse=(
                               story_modality.name == question_modality.name)):
      question = question_modality.bottom_sharded(question, dp)

    with tf.variable_scope("body"):
      if target_modality.is_class_modality:
        encoder_output = dp(self.encode, story, question,
                            features["target_space_id"], hparams)
      else:
        encoder_output, encoder_decoder_attention_bias = dp(self.encode, story,
                                                            question, features[
                                                              "target_space_id"],
                                                            hparams,
                                                            features=features)
        encoder_decoder_attention_bias = encoder_decoder_attention_bias[0]

      encoder_output = encoder_output[0]

    if hparams.pos == "timing":
      timing_signal = common_attention.get_timing_signal_1d(decode_length + 1,
                                                            hparams.hidden_size)

    def preprocess_targets(targets, i):
      """Performs preprocessing steps on the targets to prepare for the
      decoder.

      This includes:
        - Embedding the ids.
        - Flattening to 3D tensor.
        - Optionally adding timing signals.

      Args:
        targets: inputs ids to the decoder. [batch_size, 1]
        i: scalar, Step number of the decoding loop.

      Returns:
        Processed targets [batch_size, 1, hidden_dim]
      """
      # _shard_features called to ensure that the variable names match
      targets = self._shard_features({"targets": targets})["targets"]
      with tf.variable_scope(target_modality.name):
        targets = target_modality.targets_bottom_sharded(targets, dp)[0]
      targets = common_layers.flatten4d3d(targets)

      targets = tf.cond(tf.equal(i, 0), lambda: tf.zeros_like(targets),
                        lambda: targets)

      if hparams.pos == "timing":
        targets += timing_signal[:, i:i + 1]
      return targets

    decoder_self_attention_bias = (
      common_attention.attention_bias_lower_triangle(decode_length))
    if hparams.proximity_bias:
      decoder_self_attention_bias += common_attention.attention_bias_proximal(
        decode_length)

    def symbols_to_logits_fn(ids, i, cache):
      """Go from ids to logits for next symbol."""
      ids = ids[:, -1:]
      targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
      targets = preprocess_targets(targets, i)

      bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]

      with tf.variable_scope("body"):
        body_outputs = dp(self.decode, targets, cache.get("encoder_output"),
                          cache.get("encoder_decoder_attention_bias"), bias,
                          hparams, cache,
                          nonpadding=features_to_nonpadding(features, "targets")
                          )

      with tf.variable_scope(target_modality.name):
        logits = target_modality.top_sharded(body_outputs, None, dp)[0]

      ret = tf.squeeze(logits, axis=[1, 2, 3])
      return ret, cache

    def labels_to_logits_fn(unused_ids, unused_i, cache):
      """Go from labels to logits"""
      with tf.variable_scope("body"):
        body_outputs = dp(tf.expand_dims, cache.get("encoder_output"), 2)

      with tf.variable_scope(target_modality.name):
        logits = target_modality.top_sharded(body_outputs, None, dp)[0]

      ret = tf.squeeze(logits, axis=[1, 2, 3])
      return ret, cache

    if target_modality.is_class_modality:
      ret = transformer.fast_decode(encoder_output=encoder_output,
                                    encoder_decoder_attention_bias=None,
                                    symbols_to_logits_fn=labels_to_logits_fn,
                                    hparams=hparams,
                                    decode_length=decode_length,
                                    vocab_size=target_modality.top_dimensionality,
                                    beam_size=beam_size,
                                    top_beams=top_beams, alpha=alpha,
                                    batch_size=batch_size)

    else:
      ret = transformer.fast_decode(encoder_output=encoder_output,
                                    encoder_decoder_attention_bias=encoder_decoder_attention_bias,
                                    symbols_to_logits_fn=symbols_to_logits_fn,
                                    hparams=hparams,
                                    decode_length=decode_length,
                                    vocab_size=target_modality.top_dimensionality,
                                    beam_size=beam_size,
                                    top_beams=top_beams, alpha=alpha,
                                    batch_size=batch_size)

    return ret


  def model_fn(self, features):
    with tf.variable_scope(tf.get_variable_scope(), use_resource=True):
      transformed_features = self.bottom(features)

      if self.hparams.activation_dtype == "bfloat16":
        for k, v in sorted(six.iteritems(transformed_features)):
          if v.dtype == tf.float32:
            transformed_features[k] = tf.cast(v, tf.bfloat16)

      with tf.variable_scope("body"):
        t2t_model.log_info("Building model body")
        body_out = self.body(transformed_features, features)
      output, losses = self._normalize_body_output(body_out)

      if "training" in losses:
        t2t_model.log_info("Skipping T2TModel top and loss because training loss "
                 "returned from body")
        logits = output
      else:
        logits = self.top(output, features)
        losses["training"] = 0.0
        if self._hparams.mode != tf.estimator.ModeKeys.PREDICT:
          losses["training"] = self.loss(logits, features)

      return logits, losses

  def snippet_encoding(self, input, original_input,
                      initializer=None, scope=None):
    """
    Implementation of the learned multiplicative mask from Section 2.1,
    Equation 1.
    This module is also described in [End-To-End Memory Networks](
    https://arxiv.org/abs/1502.01852)
    as Position Encoding (PE). The mask allows the ordering of words in a
    sentence to affect the encoding.    """
    with tf.variable_scope(scope, 'encode_input', initializer=initializer):

      _, _, max_sentence_length, embedding_size = input.get_shape().as_list()


      pad_mask = tf.to_float(tf.not_equal(original_input,
                             tf.constant(searchqa_problem.PAD, dtype=tf.int32)))
      input_masked = input * pad_mask
      positional_mask = tf.get_variable(name='positional_mask',
        shape=[max_sentence_length, embedding_size])
      # batch_size * len * emb_size
      encoded_input = tf.reduce_sum(tf.multiply(input_masked, positional_mask)
                                    , axis=2)

      return encoded_input


  def body(self, features, original_features):
    """Transformer main model_fn.
    Args:
      features: Map of features to the model. Should contain the following:
          "inputs": Transformer inputs [batch_size, input_length, hidden_dim]
          "targets": Target decoder outputs.
              [batch_size, decoder_length, hidden_dim]
          "target_space_id"
    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
    hparams = self._hparams

    snippets = features.get(searchqa_problem.FeatureNames.SNIPPETS)
    questions = features.get(searchqa_problem.FeatureNames.QUESTION)
    target_space = features["target_space_id"]

    with tf.variable_scope('input'):
      # [batch_size, search_results_len, embed_sz]
      encoded_story = self.inputs_encoding(input=snippets,
                                           original_input=original_features.get(
                                           searchqa_problem.FeatureNames.SNIPPETS),
                                           initializer=tf.constant_initializer(
                                             1.0), scope='story_encoding')

      # [batch_size, 1, embed_sz]
      encoded_question = self.inputs_encoding(input=questions,
                                              original_input=original_features.get(
                                              searchqa_problem.FeatureNames.QUESTION),
                                              initializer=tf.constant_initializer(
                                                1.0), scope='question_encoding')

    # Concat snippets and questions to creat the inputs
    inputs = tf.concat([snippets, questions], axis=1)
    # the input is 4D by default and it gets squeezed from 4D to 3D in the
    # encode function, so we need to make it 4D by inserting channel dim.
    # inputs = tf.expand_dims(inputs, axis = 2)

    losses = []
    encoder_output, encoder_decoder_attention_bias = self.encode(
      inputs, target_space, hparams, features=features, losses=losses)

    targets = features["targets"]
    targets_shape = common_layers.shape_list(targets)
    targets = common_layers.flatten4d3d(targets)

    decoder_input, decoder_self_attention_bias = transformer.transformer_prepare_decoder(
      targets, hparams, features=features)

    decoder_output = self.decode(
      decoder_input,
      encoder_output,
      encoder_decoder_attention_bias,
      decoder_self_attention_bias,
      hparams,
      nonpadding=features_to_nonpadding(features, "targets"),
      losses=losses)

    ret = tf.reshape(decoder_output, targets_shape)
    if losses:
      return ret, {"extra_loss": tf.add_n(losses)}
    else:
      return ret
