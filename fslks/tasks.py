from fslks import sink

# noinspection PyUnresolvedReferences
from fslks.datasets.summarization import *

# noinspection PyUnresolvedReferences
from fslks.datasets.argumentation import *

sink.register('bioasq/single-doc',
              prompt=sink.Constant('summarize'),
              input=sink.Sequence([
                  sink.Feature('question'),
                  sink.Feature('article')
              ]),
              output=sink.Feature('summary'))

sink.register('bioasq/multi-doc',
              prompt=sink.Constant('summarize'),
              input=sink.Sequence([
                  sink.Feature('question'),
                  sink.Sequence('article')
              ]),
              output=sink.Feature('summary'))

sink.register('chiqa/section2answer_multi_abstractive',
              prompt=sink.Constant('summarize'),
              input=sink.Sequence([
                  sink.Feature('question'),
                  sink.Sequence('article')
              ]),
              output=sink.Feature('summary'))

sink.register('chiqa/page2answer_multi_abstractive',
              prompt=sink.Constant('summarize'),
              input=sink.Sequence([
                  sink.Feature('question'),
                  sink.Sequence('article')
              ]),
              output=sink.Feature('summary'))

sink.register('chiqa/section2answer_multi_extractive',
              prompt=sink.Constant('summarize'),
              input=sink.Sequence([
                  sink.Feature('question'),
                  sink.Sequence('article')
              ]),
              output=sink.Feature('summary'))

sink.register('chiqa/page2answer_multi_extractive',
              prompt=sink.Constant('summarize'),
              input=sink.Sequence([
                  sink.Feature('question'),
                  sink.Sequence('article')
              ]),
              output=sink.Feature('summary'))

sink.register('chiqa/section2answer_single_abstractive',
              prompt=sink.Constant('summarize'),
              input=sink.Sequence([
                  sink.Feature('question'),
                  sink.Feature('article')
              ]),
              output=sink.Feature('summary'))

sink.register('chiqa/page2answer_single_abstractive',
              prompt=sink.Constant('summarize'),
              input=sink.Sequence([
                  sink.Feature('question'),
                  sink.Feature('article')
              ]),
              output=sink.Feature('summary'))

sink.register('chiqa/section2answer_single_extractive',
              prompt=sink.Constant('summarize'),
              input=sink.Sequence([
                  sink.Feature('question'),
                  sink.Feature('article')
              ]),
              output=sink.Feature('summary'))

sink.register('chiqa/page2answer_single_extractive',
              prompt=sink.Constant('summarize'),
              input=sink.Sequence([
                  sink.Feature('question'),
                  sink.Feature('article')
              ]),
              output=sink.Feature('summary'))

sink.register('ebm/answer',
              prompt=sink.Constant('summarize'),
              input=sink.Sequence([
                  sink.Feature('question'),
                  sink.Sequence('article')
              ]),
              output=sink.Feature('summary'))

sink.register('ebm/justify',
              prompt=sink.Constant('summarize'),
              input=sink.Sequence([
                  sink.Feature('question'),
                  sink.Feature('answer'),
                  sink.Sequence('article')
              ]),
              output=sink.Feature('summary'))

sink.register('medlineplus_references',
              prompt=sink.Constant('summarize'),
              input=sink.Sequence('article'),
              output=sink.Feature('summary'))

sink.register('medlineplus_reviews',
              prompt=sink.Constant('summarize'),
              input=sink.Sequence('article'),
              output=sink.Feature('summary'))

sink.register('medinfo',
              prompt=sink.Constant('summarize'),
              input=sink.Sequence([
                  sink.Feature('question'),
                  sink.Feature('article')
              ]),
              output=sink.Feature('summary'))

sink.register('pubmed_summ',
              prompt=sink.Constant('summarize'),
              input=sink.Sequence([
                  sink.Feature('title'),
                  sink.Feature('article')
              ]),
              output=sink.Feature('summary'))

sink.register('cochrane_summ',
              prompt=sink.Constant('summarize'),
              input=sink.Feature('article'),
              output=sink.Feature('summary'))

sink.register('scientific_papers',
              prompt=sink.Constant('summarize'),
              input=sink.Feature('article'),
              output=sink.Feature('abstract'))

sink.register('movie_rationales',
              prompt=sink.Constant('summarize'),
              input=sink.Sequence('evidences'),
              output=sink.Feature('review'))

sink.register('cnn_dailymail',
              prompt=sink.Constant('summarize'),
              input=sink.Feature('article'),
              output=sink.Feature('highlights'))

sink.register('super_glue/copa',
              prompt=sink.Sequence([
                  sink.Constant('choose'),
                  sink.Feature('question'),
              ]),
              input=sink.Sequence([
                  sink.Feature('premise'),
                  sink.Feature('choice1'),
                  sink.Feature('choice2'),
              ]),
              output=sink.LabelMapping('label', {
                  0: sink.Feature('choice1'),
                  1: sink.Feature('choice2')
              }))

_eviconv_stance_mapping = sink.LabelMapping('stance', {
    0: sink.Constant('PRO:'),
    1: sink.Constant('CON:'),
})
sink.register('evi_conv',
              prompt=sink.Constant('argue'),
              input=sink.Sequence([
                  sink.DictEntry('evidence_1', _eviconv_stance_mapping),
                  sink.DictEntry('evidence_1', sink.Feature('text')),
                  sink.DictEntry('evidence_2', _eviconv_stance_mapping),
                  sink.DictEntry('evidence_2', sink.Feature('text')),
              ]),
              output=sink.LabelMapping('label', {
                  0: sink.DictEntry('evidence_1', sink.Feature('text')),
                  1: sink.DictEntry('evidence_2', sink.Feature('text'))
              }))
