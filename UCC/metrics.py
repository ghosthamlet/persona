
import collections

from torchtext.data.metrics import bleu_score
from pytorch_lightning.metrics.functional import f1_score


def distinct_score(output, gram_type=1):
    """Paper: https://arxiv.org/pdf/1510.03055.pdf

    Degree of diversity by calculating the number of distinct unigrams 
    and bigrams in generated responses.  
    The value is scaled by total number of generated tokens to avoid 
    favoring long sentences.
    """
    n_token = len(output)
    grams = list(_get_ngrams(output, 2))

    n_gram = 0
    for v in grams:
        if len(v) == gram_type:
            n_gram += 1

    return n_gram / n_token


def _get_ngrams(segment, max_order):
  """Extracts all n-grams upto a given maximum order from an input segment.
  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.
  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
  ngram_counts = collections.Counter()
  for order in range(1, max_order + 1):
    for i in range(0, len(segment) - order + 1):
      ngram = tuple(segment[i:i+order])
      ngram_counts[ngram] += 1
  return ngram_counts

