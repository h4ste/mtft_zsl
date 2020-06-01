# Zero-shot Learning (ZSL) for consumer health answer summarization

To install related dependencies:
```bash
pip install -r requirements.txt
```

To train the model:
```bash
export CHECKPOINT_DIR = <path to save model checkpoints>
export PREDICTION_DIR = <path to save predictions>
python -m fslks.run_experiment \
                --training_tasks=\"medinfo bioasq/single_doc bioasq/multi_doc pubmed_summ medlineplus_references super_glue/copa scientific_papers/arxiv scientific_papers/pubmed cochrane_summ cnn_dailymail ebm/answer ebm/justify squad movie_rationales evi_conv cosmos_qa::validation mctaco qa4mre/2011.main.EN qa4mre/2012.main.EN qa4mre/2013.main.EN qa4mre/2012.alzheimers.EN qa4mre/2013.alzheimers.EN\" \
                --validation_tasks=\"mctaco::test\" \
                --testing_tasks=\"chiqa/section2answer_single_extractive duc/2004 duc/2007 tac/2009 tac/2010\" \
                --do_train=True \
                --checkpoint_dir=$CHECKPOINT_DIR \
                --do_predict=True \
                --prediction_dir=$PREDICTION_DIR \
                --do_test=True \
                --init_checkpoint=t5-base \
                --num_epochs=10 \
                --max_seq_len=512 \
                --cache_dir=MEMORY \
                --batch_size=8 \
                --eval_batch_size=16 \
                --eval_batches=10 \
                --steps_per_epoch=1000 \
                --warmup_epochs=3 \
                --prefetch_size=-1 \
                --cache_dir=MEMORY \
                --implementation='pytorch' \
                --use_amp=False \
                --temperature=2
```

To use BART-Large, replace "t5-base" with "bart-large". 
To do temperature-scaling, set `--temperature=<T>`.
To do adaptive mixing set `--dynamic_mixing=True`
To do self-adaptive mixing set `--dynamic_mixing=True --mix_from_validation=False`

To do testing only:
```bash
python -m fslks.run_experiment \
                --testing_tasks=\"chiqa/section2answer_single_extractive duc/2004 duc/2007 tac/2009 tac/2010\" \
                --do_predict=True \
                --prediction_dir=$PREDICTION_DIR \
                --do_test=True \
                --init_checkpoint=$CHECKPOINT_DIR \
                --max_seq_len=512 \
                --cache_dir=MEMORY \
                --eval_batch_size=16 \
                --eval_batches=10 \
                --prefetch_size=-1 \
                --cache_dir=MEMORY \
                --implementation='pytorch' \
                --use_amp=False
```

Note: `--init_checkpoint` can be the name of any hugging face model, or the path to any saved checkpoint created by `--do_train=True`

