## Installation

We adapted this ctc decoder from [here](https://github.com/PaddlePaddle/DeepSpeech/tree/develop/deepspeech/decoders/swig).
This decoder can only run on cpu.

* continuous decoding for streaming asr
* support kenlm language model
* multiprocessing

To install the decoder:
```bash
git clone https://github.com/Slyne/ctc_decoder.git
apt-get update
apt-get install swig
apt-get install python3-dev 
cd ctc_decoder/swig && bash setup.sh
```

## Usage

Please refer to ```swig/test/test_en.py``` and ```swig/test/test_zh.py``` for how to do streaming decoding and offline decoding w/o language model.

### Adding language model
How to build the language model ?
You may refer to [kenlm](https://github.com/kpu/kenlm).
For Mandarin, the input text for language model should be like:
```
好 好 学 习 ，天 天 向 上 ！
再 接 再 厉
...
```
There's a space between two characters.

For English, the input text is just like the normal text.
```
Share Market Today - Stock Market and Share Market Live Updates
```

How to add language model:
```
alpha = 0.5
beta = 0.5
lm_path = '../kenlm/lm/test.arpa'
scorer = decoder.Scorer(alpha, beta, lm_path, vocab_list)
......
result1 =  decoder.ctc_beam_search_decoder_batch(batch_chunk_log_prob_seq, 
                                                 batch_chunk_log_probs_idx,
                                                 batch_root_trie,
                                                 batch_start,
                                                 beam_size, num_processes,
                                                 blank_id, space_id,
                                                 cutoff_prob, scorer)
```
How language model in called in this implementation of ctc prefix beam search ?

If the language model is char based (like the Mandarin lm), it will call the language model scorer all the times.
If the language model is word based (like the English lm), it will only call the scorer whenever `space_id` is detected.
