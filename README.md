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
