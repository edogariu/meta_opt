from collections import defaultdict, Counter
import math
import re
import sys
import unicodedata

import numpy as np
import jax
import jax.numpy as jnp
from flax.training import common_utils

from ._models import Transformer as _Transformer
from ._decode import EOS_ID, flat_batch_beam_expand, beam_search

@jax.jit
def encode(tstate, batch):
    yhat = tstate.apply_fn({'params': tstate.params}, batch['x'], train=False,)

def predict_step(tstate, batch, beam_size: int = 4, max_decode_len: int = 256, alpha=0.6):
# def predict_step(
    # inputs, params, cache, max_decode_len, config, beam_size=4
# ):
  """Predict translation with fast decoding beam search on a batch."""
  # Prepare transformer fast-decoder call for beam search: for beam search, we
  # need to set up our decoder model to handle a batch size equal to
  # batch_size * beam_size, where each batch item"s data is expanded in-place
  # rather than tiled.
  # i.e. if we denote each batch element subtensor as el[n]:
  # [el0, el1, el2] --> beamsize=2 --> [el0,el0,el1,el1,el2,el2]
  encoded_inputs = flat_batch_beam_expand(
      tstate.apply_fn(
          {"params": tstate.params}, batch['x']['inputs'], method=_Transformer.encode, train=False,
      ),
      beam_size,
  )
  raw_inputs = flat_batch_beam_expand(batch['x']['inputs'], beam_size)

  def tokens_ids_to_logits(flat_ids):
    """Token slice to logits from decoder model."""
    # --> [batch * beam, 1, vocab]
    flat_logits = tstate.apply_fn(
        {"params": tstate.params},
        encoded_inputs,
        raw_inputs,  # only needed for input padding mask
        flat_ids,
        method=_Transformer.decode,
        train=False,
    )
    # Remove singleton sequence-length dimension:
    # [batch * beam, 1, vocab] --> [batch * beam, vocab]
    flat_logits = flat_logits.squeeze(axis=1)
    return flat_logits

  # Using the above-defined single-step decoder function, run a
  # beam search over possible sequences given input encoding.
  beam_seqs, _ = beam_search(
      batch['x']['inputs'],
      tokens_ids_to_logits,
      beam_size=beam_size,
      alpha=alpha,
      eos_id=EOS_ID,
      max_decode_len=max_decode_len,
  )

  # Beam search returns [n_batch, n_beam, n_length + 1] with beam dimension
  # sorted in increasing order of log-probability.
  # Return the highest scoring beam sequence, drop first dummy 0 token.
  return beam_seqs[:, -1, 1:]



class UnicodeRegex:
  """Ad-hoc hack to recognize all punctuation and symbols."""

  def __init__(self):
    punctuation = self.property_chars("P")
    self.nondigit_punct_re = re.compile(r"([^\d])([" + punctuation + r"])")
    self.punct_nondigit_re = re.compile(r"([" + punctuation + r"])([^\d])")
    self.symbol_re = re.compile("([" + self.property_chars("S") + "])")

  def property_chars(self, prefix):
    return "".join(
        chr(x)
        for x in range(sys.maxunicode)
        if unicodedata.category(chr(x)).startswith(prefix)
    )


uregex = UnicodeRegex()


def bleu_tokenize(string):
  r"""Tokenize a string following the official BLEU implementation.

  See https://github.com/moses-smt/mosesdecoder/'
           'blob/master/scripts/generic/mteval-v14.pl#L954-L983
  In our case, the input string is expected to be just one line
  and no HTML entities de-escaping is needed.
  So we just tokenize on punctuation and symbols,
  except when a punctuation is preceded and followed by a digit
  (e.g. a comma/dot as a thousand/decimal separator).

  Note that a number (e.g. a year) followed by a dot at the end of sentence
  is NOT tokenized, i.e. the dot stays with the number because
  `s/(\p{P})(\P{N})/ $1 $2/g` does not match this case (unless we add a
  space after each sentence). However, this error is already in the
  original mteval-v14.pl and we want to be consistent with it.

  Args:
    string: the input string

  Returns:
    a list of tokens
  """
  string = uregex.nondigit_punct_re.sub(r"\1 \2 ", string)
  string = uregex.punct_nondigit_re.sub(r" \1 \2", string)
  string = uregex.symbol_re.sub(r" \1 ", string)
  return string.split()


def _get_ngrams(segment, max_order):
  """Extracts all n-grams up to a given maximum order from an input segment.

  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this methods.

  Returns:
    The Counter containing all n-grams up to max_order in segment
    with a count of how many times each n-gram occurred.
  """
  ngram_counts = Counter()
  for order in range(1, max_order + 1):
    for i in range(0, len(segment) - order + 1):
      ngram = tuple(segment[i : i + order])
      ngram_counts[ngram] += 1
  return ngram_counts


def compute_bleu_matches(reference_corpus, translation_corpus, max_order=4):
  """Computes BLEU match stats of translations against one or more references.

  Args:
    reference_corpus: list of references for each translation. Each reference
      should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation should
      be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.

  Returns:
    Aggregated n-gram stats for BLEU calculation.
  """
  reference_length = 0
  translation_length = 0
  bp = 1.0
  geo_mean = 0

  matches_by_order = [0] * max_order
  possible_matches_by_order = [0] * max_order
  precisions = []

  for references, translations in zip(reference_corpus, translation_corpus):
    reference_length += len(references)
    translation_length += len(translations)
    ref_ngram_counts = _get_ngrams(references, max_order)
    translation_ngram_counts = _get_ngrams(translations, max_order)

    overlap = {
        ngram: min(count, translation_ngram_counts[ngram])
        for ngram, count in ref_ngram_counts.items()
    }

    for ngram in overlap:
      matches_by_order[len(ngram) - 1] += overlap[ngram]
    for ngram in translation_ngram_counts:
      possible_matches_by_order[len(ngram) - 1] += translation_ngram_counts[
          ngram
      ]

  return (
      np.array(matches_by_order),
      np.array(possible_matches_by_order),
      np.array(reference_length),
      np.array(translation_length),
  )


def bleu_partial(ref_lines, hyp_lines, case_sensitive=False):
  """Compute n-gram statistics for two lists of references and translations."""
  if len(ref_lines) != len(hyp_lines):
    raise ValueError(
        "Reference and translation lists have different numbers of lines."
    )
  if not case_sensitive:
    ref_lines = [x.lower() for x in ref_lines]
    hyp_lines = [x.lower() for x in hyp_lines]
  ref_tokens = [bleu_tokenize(x) for x in ref_lines]
  hyp_tokens = [bleu_tokenize(x) for x in hyp_lines]
  return compute_bleu_matches(ref_tokens, hyp_tokens)


def complete_bleu(
    matches_by_order,
    possible_matches_by_order,
    reference_length,
    translation_length,
    max_order=4,
    use_bp=True,
):
  """Compute BLEU score from aggregated n-gram statistics."""
  precisions = [0] * max_order
  smooth = 1.0
  geo_mean = 0.0
  for i in range(0, max_order):
    if possible_matches_by_order[i] > 0:
      precisions[i] = matches_by_order[i] / possible_matches_by_order[i]
      if matches_by_order[i] > 0:
        precisions[i] = matches_by_order[i] / possible_matches_by_order[i]
      else:
        smooth *= 2
        precisions[i] = 1.0 / (smooth * possible_matches_by_order[i])
    else:
      precisions[i] = 0.0

  if max(precisions) > 0:
    p_log_sum = sum(math.log(p) for p in precisions if p)
    geo_mean = math.exp(p_log_sum / max_order)

  if use_bp:
    if not reference_length:
      bp = 1.0
    else:
      ratio = translation_length / reference_length
      if ratio <= 0.0:
        bp = 0.0
      elif ratio >= 1.0:
        bp = 1.0
      else:
        bp = math.exp(1 - 1.0 / ratio)
  bleu = geo_mean * bp
  return float(bleu) * 100.0


def bleu_local(ref_lines, hyp_lines, case_sensitive=False):
  """Compute BLEU for two lists of reference and hypothesis translations."""
  stats = bleu_partial(ref_lines, hyp_lines, case_sensitive=case_sensitive)
  return complete_bleu(*stats) * 100

def per_host_sum_pmap(in_tree):
  """Execute psum on in_tree"s leaves over one device per host."""
  host2devices = defaultdict(list)
  for d in jax.devices():
    host2devices[d.process_index].append(d)
  devices = [host2devices[k][0] for k in host2devices]
  host_psum = jax.pmap(lambda x: jax.lax.psum(x, "i"), "i", devices=devices)

  def pre_pmap(xs):
    return jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (1,) + x.shape), xs
    )

  def post_pmap(xs):
    return jax.tree_util.tree_map(lambda x: x[0], xs)

  return post_pmap(host_psum(pre_pmap(in_tree)))

def pad_examples(x, desired_batch_size):
  """Expand batch to desired size by repeating last slice."""
  batch_pad = desired_batch_size - x.shape[0]
  return np.concatenate([x, np.tile(x[-1], (batch_pad, 1))], axis=0)

def initialize_cache(inputs, max_decode_len, config):
  """Initialize a cache for a given input shape and max decode length."""
  target_shape = (inputs.shape[0], max_decode_len) + inputs.shape[2:]
  initial_variables = _Transformer(config).init(
      jax.random.key(0),
      jnp.ones(inputs.shape, config.dtype),
      jnp.ones(target_shape, config.dtype),
  )
  return initial_variables["cache"]

def translate_and_calculate_bleu(
    tstate,
    dataset,
    tokenizer,
    max_predict_length: int = 256,
):
  """Translates the `dataset` and calculates the BLEU score."""

  def decode_tokens(toks):
    valid_toks = toks[: np.argmax(toks == EOS_ID) + 1].astype(np.int32)
    return tokenizer.detokenize(valid_toks).numpy().decode("utf-8")
  
  sources, references, predictions = [], [], []
  for batch in dataset:
    pred = predict_step(tstate, batch)
    # Iterate through non-padding examples of batch.
    inputs, targets = batch['x']['inputs'], batch['y']
    for i, s in enumerate(pred):
      sources.append(decode_tokens(inputs[i]))
      references.append(decode_tokens(targets[i]))
      predictions.append(decode_tokens(s))

  # Calculate BLEU score for translated eval corpus against reference.
  bleu_matches = bleu_partial(references, predictions)
  all_bleu_matches = per_host_sum_pmap(bleu_matches)
  bleu_score = complete_bleu(*all_bleu_matches)
  # Save translation samples for tensorboard.
  exemplars = ""
  for n in np.random.choice(np.arange(len(predictions)), 8):
    exemplars += f"{sources[n]}\n\n{references[n]}\n\n{predictions[n]}\n\n"
  return exemplars, bleu_score
