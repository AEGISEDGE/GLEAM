#GLEAM#
The source code of global generative topic fusion with local sentence generation encoding for abstractive summarization.  
**Requirement:   Tensorflow ; Bazel**

#Description:#
## Global representation generation ##
Before running GLEAM, the VAE topic model need to be trained on corpus and export the corresponding document vectors.   
1.	**vocab_headline.py** generate the vocabulary of corpus.  
usage: **python vocab_headline.py <souce_dir> <target_dir>**
<source_dir> : directory of corpus.  
<target_dir> : output directory of vocabulary file.   
2.	**batch_text_vector.py** process corpus for training and evaluating.  
usage: **python batch_text_vector.py <souce_dir> <target_dir> <vocab_path>**  
<source_dir> : directory of corpus.  
<target_dir> : output directory of vocabulary file.   
<vocab_path> : vocabulary file path.  
3. **batch_nvdm.py** training NVDM on the corpus processed above (which needs to be placed at 'train' folder).  
Training : **python NVDM.py**  
Testing and generate the document vectors: **Python NVDM.py --test_data_dir <test_dir> --store_data_dir <store_dir> --test True**  
<test_dir> : test corpuse directory.  
<store_dir>: directory where to store the document vectors.  
**【The above 1-4 procedure need to run on both training set and test set】**  

## Generating summarization ##   
Recommending runing with GPU.   
5. selecting GPU device to run the model:   
**export CUDA_VISIBLE_DEVICES = <gpu_id>**  
<gpu_id> : Your cuda gpu id.  
6. Build the project:  
bazel build -c opt --config=cuda textsum/...  
7. Training:  
bazel-bin/textsum/seq2seq_attention \
  --mode=train \
  --article_key=article \
  --abstract_key=abstract \
  --data_path=data/train/* \
  --vocab_path=data/vocabulary/vocab \
  --log_root=textsum/log_root \
  --train_dir=textsum/log_root/train**  
8. Decoding:  
bazel-bin/textsum/seq2seq_attention \
  --mode=decode \
  --article_key=article \
  --abstract_key=abstract \
  --data_path=data/duc_test_two_vector/* \
  --vocab_path=data/vocabulary/vocab \
  --log_root=textsum/log_root \
  --decode_dir=textsum/log_root/decode \
  --beam_size=8**
