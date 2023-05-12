
import numpy as np
import logging
from swig_decoders import TrieVector, ctc_beam_search_decoder_batch, \
                            map_sent, map_batch, \
                            PathTrie, \
                            Scorer, HotWordsBoosting, BatchHotWordsScorer
import multiprocessing

logging.basicConfig(filename='out.log', level=logging.INFO)

def test_prefix_beam_search(batch_log_ctc_probs, batch_lens, beam_size, blank_id, space_id, cutoff_prob=0.999):
    """
    Prefix beam search
    Params:
        batch_log_probs: B x T x V, the log probabilities of a sequence
        batch_lens: B, the actual length of each sequence
    Return:
        hyps: a batch of beam candidates for each sequence
        [[(score, cand_list1), (score, cand_list2), ....(score, cand_list_beam)],
         [(score, cand_list1), (score, candi_list2), ...],
         ...
         []]
    """
    #batch_log_probs_seq, batch_log_probs_idx = torch.topk(batch_log_ctc_probs, beam_size, dim=-1)
    batch_log_probs_idx = np.argsort(batch_log_ctc_probs, axis=-1)[:, :, ::-1]
    batch_log_probs_seq = np.sort(batch_log_ctc_probs, axis=-1)[:, :, ::-1]
    batch_log_probs_seq_list = batch_log_probs_seq.tolist()
    batch_log_probs_idx_list = batch_log_probs_idx.tolist()
    batch_len_list = batch_lens.tolist()
    batch_log_probs_seq = []
    batch_log_probs_ids = []
    batch_start = []
    batch_root = TrieVector()
    root_dict = {}
    for i in range(len(batch_len_list)):
        num_sent = batch_len_list[i]
        batch_log_probs_seq.append(batch_log_probs_seq_list[i][0:num_sent])
        batch_log_probs_ids.append(batch_log_probs_idx_list[i][0:num_sent])
        root_dict[i] = PathTrie()
        batch_root.append(root_dict[i])
        batch_start.append(True)
    num_processes = min(multiprocessing.cpu_count()-1, len(batch_log_probs_seq))
    score_hyps = ctc_beam_search_decoder_batch(batch_log_probs_seq,
                                            batch_log_probs_ids,
                                            batch_root,
                                            batch_start,
                                            beam_size,
                                            num_processes,
                                            blank_id,
                                            space_id,
                                            cutoff_prob)
    return score_hyps

def test_batch_greedy_search(batch_log_ctc_probs, batch_lens, vocab_list, blank_id):
    """
    Greedy search
    Params:
        batch_log_ctc_probs: B x T x V
        batch_lens: B
        vocab_list: a list of symbols, of size V
        blank_id: id for blank symbol
    Return:
        batch of decoded string sentences
    """
    
    sort_ids = np.argsort(batch_log_ctc_probs, axis=-1)[:, :, ::-1]
    batch_greedy_ids = sort_ids[:, :, 0].tolist()
    batch_len_list = batch_lens.tolist()
    batch_ids = []
    for seq_ids, seq_len in zip(batch_greedy_ids, batch_len_list):
        batch_ids.append(seq_ids[0: seq_len])
    num_processes = min(multiprocessing.cpu_count()-1, len(batch_ids))
    greedy = True
    result = map_batch(batch_ids, vocab_list, num_processes, greedy, blank_id)
    return result


def test_map_batch(batch_sent_list, vocab_list, blank_id):
    """
    Map a batch of sent ids to string
    Prams:
        batch_sent_list: a list of list of ids
        vocab_list: a list of symbols, of size V
        blank_id: id for blank symbol
    """
    num_processes = min(multiprocessing.cpu_count()-1, len(batch_sent_list))
    greedy = False # this is not used for greedy search so we set it to false
    results = map_batch(batch_sent_list, vocab_list, num_processes, greedy, blank_id)
    return results

def test_map_sent(sent_ids, vocab_list, greedy, blank_id):
    """
    Map one sentence ids to string
    greedy: False, just map. True, use ctc greedy search
    """
    return map_sent(sent_ids, vocab_list, greedy, blank_id)
    

def load_vocab(vocab_file):
    vocab = []
    with open(vocab_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().split()
            vocab.append(line[0])
    return vocab

def test_prefix_beam_search_hotwords(batch_log_ctc_probs, batch_lens, beam_size, blank_id, space_id, lm_path, vocab_list, cutoff_prob=0.999):
    """
    Prefix beam search
    Params:
        batch_log_probs: B x T x V, the log probabilities of a sequence
        batch_lens: B, the actual length of each sequence
    Return:
        hyps: a batch of beam candidates for each sequence
        [[(score, cand_list1), (score, cand_list2), ....(score, cand_list_beam)],
         [(score, cand_list1), (score, candi_list2), ...],
         ...
         []]
    """
    # batch_log_probs_seq, batch_log_probs_idx = torch.topk(batch_log_ctc_probs, beam_size, dim=-1)
    batch_log_probs_idx = np.argsort(batch_log_ctc_probs, axis=-1)[:, :, ::-1]
    batch_log_probs_seq = np.sort(batch_log_ctc_probs, axis=-1)[:, :, ::-1]
    batch_log_probs_seq_list = batch_log_probs_seq.tolist()
    batch_log_probs_idx_list = batch_log_probs_idx.tolist()
    batch_len_list = batch_lens.tolist()
    batch_log_probs_seq = []
    batch_log_probs_ids = []
    batch_start = []
    batch_root = TrieVector()
    root_dict = {}
    scorer = Scorer(0.5, 0.5, lm_path, vocab_list)
    batch_hotwords_scorer = BatchHotWordsScorer()
    #In the first badcase, there is a big difference in scoring between the optimal path and other paths.
    hot_words1 = {'换一': -3.40282e+38, '首歌': -100, '换歌': 3.40282e+38}
    hot_words2 = {'极点': 5}
    #hot_words = {}
    hotwords_scorer_dict = {}
    for i in range(len(batch_len_list)):
        num_sent = batch_len_list[i]
        batch_log_probs_seq.append(batch_log_probs_seq_list[i][0:num_sent])
        batch_log_probs_ids.append(batch_log_probs_idx_list[i][0:num_sent])
        root_dict[i] = PathTrie()
        batch_root.append(root_dict[i])
        batch_start.append(True)
        if i == 0:
            hotwords_scorer = HotWordsBoosting(hot_words1, vocab_list)
        else:
            hotwords_scorer = HotWordsBoosting(hot_words2, vocab_list)
        hotwords_scorer_dict[i] = hotwords_scorer
        # don't use hotwords. set batch_hotwords_scorer=None or
        # batch_hotwords_scorer.append(None) or batch_hotwords_scorer.append(HotWordsBoosting({}, vocab_list))
        batch_hotwords_scorer.append(hotwords_scorer)
    num_processes = min(multiprocessing.cpu_count()-1, len(batch_log_probs_seq))

    score_hyps = ctc_beam_search_decoder_batch(batch_log_probs_seq,
                                               batch_log_probs_ids,
                                               batch_root,
                                               batch_start,
                                               beam_size,
                                               num_processes,
                                               blank_id,
                                               space_id,
                                               cutoff_prob,
                                               scorer,
                                               batch_hotwords_scorer)
    return score_hyps


if __name__ == "__main__":
    input = "data/test.npz"
    word = "data/words.txt"
    beam_size = 10
    blank_id = 0
    space_id = 45

    vocab_list = load_vocab(word)
    input = np.load(input)
    batch_log_ctc_probs = input['batch_log_ctc_probs']
    batch_len = input["batch_len"]
    # ctc prefix beam search
    logging.info("Testing ctc prefix beam search")
    score_hyps = test_prefix_beam_search(batch_log_ctc_probs,
                            batch_len,
                            beam_size,
                            blank_id,
                            space_id,
                            cutoff_prob=0.999)
    # map the most probable cand ids to string
    batch_ids = [score_hyps[0][0][1], score_hyps[1][0][1]]
    map_sents = test_map_batch(batch_ids, vocab_list, blank_id)
    logging.info(map_sents)

    logging.info("Testing greedy search")
    # greedy search
    greedy_sents = test_batch_greedy_search(batch_log_ctc_probs,
                                            batch_len,
                                            vocab_list,
                                            blank_id)
    logging.info(greedy_sents)

    logging.info("Test one sentence")
    sent_ids = score_hyps[0][0][1]
    one_sent = test_map_sent(sent_ids, vocab_list, False, blank_id)
    logging.info(one_sent)

    batch_ids = []
    logging.info("Test ctc prefix beam search all hyps")
    for i in range(len(score_hyps)):
        for j in range(len(score_hyps[i])):
            batch_ids.append(score_hyps[i][j][1])

    batch_map_sents = test_map_batch(batch_ids, vocab_list, blank_id)
    logging.info(batch_map_sents)


    logging.info("Test hotwords boosting with character-level language models during ctc prefix beam search")
    character_lm="lm/character_corpus.arpa"
    score_hyps = test_prefix_beam_search_hotwords(batch_log_ctc_probs,
                                         batch_len,
                                         beam_size,
                                         blank_id,
                                         space_id,
                                         character_lm,
                                         vocab_list,
                                         cutoff_prob=0.999,
                                        )
    batch_ids = []
    for i in range(len(score_hyps)):
        for j in range(len(score_hyps[i])):
            batch_ids.append(score_hyps[i][j][1])

    batch_map_sents = test_map_batch(batch_ids, vocab_list, blank_id)
    logging.info(batch_map_sents)


    logging.info("Test hotwords boosting with word-level language models during ctc prefix beam search")
    word_lm = "lm/segmented_corpus.arpa"
    score_hyps = test_prefix_beam_search_hotwords(batch_log_ctc_probs,
                                         batch_len,
                                         beam_size,
                                         blank_id,
                                         space_id,
                                         word_lm,
                                         vocab_list,
                                         cutoff_prob=0.999,
                                         )
    batch_ids = []
    for i in range(len(score_hyps)):
        for j in range(len(score_hyps[i])):
            batch_ids.append(score_hyps[i][j][1])

    batch_map_sents = test_map_batch(batch_ids, vocab_list, blank_id)
    logging.info(batch_map_sents)