# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import swig_decoders as decoder
import numpy as np

probs_seq = [[
            0.06390443, 0.21124858, 0.27323887, 0.06870235, 0.0361254,
            0.18184413, 0.16493624
        ], [
            0.03309247, 0.22866108, 0.24390638, 0.09699597, 0.31895462,
            0.0094893, 0.06890021
        ], [
            0.218104, 0.19992557, 0.18245131, 0.08503348, 0.14903535,
            0.08424043, 0.08120984
        ], [
            0.12094152, 0.19162472, 0.01473646, 0.28045061, 0.24246305,
            0.05206269, 0.09772094
        ], [
            0.1333387, 0.00550838, 0.00301669, 0.21745861, 0.20803985,
            0.41317442, 0.01946335
        ], [
            0.16468227, 0.1980699, 0.1906545, 0.18963251, 0.19860937,
            0.04377724, 0.01457421
        ]]

vocab_list = ["\'", " ", "a", "b", "c", "d"]

log_prob_seq = np.log(np.array(probs_seq, dtype=np.float32))
log_probs_idx = np.argsort(log_prob_seq, axis=-1)[:, ::-1]
log_prob_seq = np.sort(log_prob_seq, axis=-1)[:, ::-1]

root = decoder.PathTrie()
root.score = root.log_prob_b_prev = 0.0
beam_size=20

chunk_log_prob_seq = [li.tolist() for li in log_prob_seq]
chunk_log_probs_idx = [li.tolist() for li in log_probs_idx]

alpha = 0.5
beta = 0.5
lm_path = '../kenlm/lm/test.arpa'
scorer = decoder.Scorer(alpha, beta, lm_path, vocab_list)

root2 = decoder.TrieVector()
temp_dict = {}
for i in range(2):
    root = decoder.PathTrie()
    temp_dict[i] = root
    root2.push_back(root)


batch_chunk_log_prob_seq = [chunk_log_prob_seq, chunk_log_prob_seq]
batch_chunk_log_probs_idx = [chunk_log_probs_idx, chunk_log_probs_idx]
batch_chunk_length = [6, 6]
batch_start = [True, True]

result1 =  decoder.ctc_beam_search_decoder_batch(batch_chunk_log_prob_seq, 
                                                 batch_chunk_log_probs_idx,
                                                 root2,
                                                 batch_start,
                                                 beam_size, 1, 6, 1, 0.9999, scorer)
# print single sentence result
print(decoder.map_sent(result1[0][0][1], vocab_list))
print(result1[0])

# Test stateful decoder
# continue decoding
batch_start = [False, False]
result2 = decoder.ctc_beam_search_decoder_batch(batch_chunk_log_prob_seq, 
                                                 batch_chunk_log_probs_idx,
                                                 root2,
                                                 batch_start,
                                                 beam_size, 1, 6, 1, 0.9999, scorer)

print(decoder.map_batch([result1[0][0][1], result1[1][0][1]], vocab_list, 1))
