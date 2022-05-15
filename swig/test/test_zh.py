
import numpy as np
import logging
from swig_decoders import TrieVector, ctc_beam_search_decoder_batch, \
                            map_sent, map_batch, \
                            PathTrie, TrieVector
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