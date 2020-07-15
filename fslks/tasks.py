from fslks import sink
# noinspection PyUnresolvedReferences
from fslks.datasets.argumentation import *
# noinspection PyUnresolvedReferences
from fslks.datasets.summarization import *


def register_task_mappings():
    sink.register('bioasq/single_doc',
                  input=sink.Join([
                      sink.Constant('bioasq single'),
                      sink.Constant('question:'),
                      sink.Feature('question'),
                      sink.Constant('summarize:'),
                      sink.Feature('article')
                  ]),
                  target=sink.Feature('summary'))

    sink.register('bioasq/multi_doc',
                  input=sink.Join([
                      sink.Constant('bioasq multi'),
                      sink.Constant('question:'),
                      sink.Feature('question'),
                      sink.Constant('summarize:'),
                      sink.Sequence('article')
                  ]),
                  target=sink.Feature('summary'))

    sink.register('chiqa/section2answer_multi_abstractive',
                  input=sink.Join([
                      sink.Constant('chiqa s2a abstract'),
                      sink.Constant('question:'),
                      sink.Feature('question'),
                      sink.Constant('summarize:'),
                      sink.Sequence('article')
                  ]),
                  target=sink.Feature('summary'))

    sink.register('chiqa/page2answer_multi_abstractive',
                  input=sink.Join([
                      sink.Constant('chiqa p2a abstract'),
                      sink.Constant('question:'),
                      sink.Feature('question'),
                      sink.Constant('summarize:'),
                      sink.Sequence('article')
                  ]),
                  target=sink.Feature('summary'))

    sink.register('chiqa/section2answer_multi_extractive',
                  input=sink.Join([
                      sink.Constant('chiqa s2a extract'),
                      sink.Constant('question:'),
                      sink.Feature('question'),
                      sink.Constant('summarize:'),
                      sink.Sequence('article')
                  ]),
                  target=sink.Feature('summary'))

    sink.register('chiqa/page2answer_multi_extractive',
                  input=sink.Join([
                      sink.Constant('chiqa p2a extract'),
                      sink.Constant('question:'),
                      sink.Feature('question'),
                      sink.Constant('summarize:'),
                      sink.Sequence('article')
                  ]),
                  target=sink.Feature('summary'))

    sink.register('chiqa/section2answer_single_abstractive',
                  input=sink.Join([
                      sink.Constant('chiqa s2a abstract'),
                      sink.Constant('question:'),
                      sink.Feature('question'),
                      sink.Constant('summarize:'),
                      sink.Feature('article')
                  ]),
                  target=sink.Feature('summary'))

    sink.register('chiqa/page2answer_single_abstractive',
                  input=sink.Join([
                      sink.Constant('chiqa p2a abstract'),
                      sink.Constant('question:'),
                      sink.Feature('question'),
                      sink.Constant('summarize:'),
                      sink.Feature('article')
                  ]),
                  target=sink.Feature('summary'))

    sink.register('chiqa/section2answer_single_extractive',
                  input=sink.Join([
                      sink.Constant('chiqa s2a extract'),
                      sink.Constant('question:'),
                      sink.Feature('question'),
                      sink.Constant('summarize:'),
                      sink.Feature('article')
                  ]),
                  target=sink.Feature('summary'))

    sink.register('chiqa/page2answer_single_extractive',
                  input=sink.Join([
                      sink.Constant('chiqa p2a extract'),
                      sink.Constant('question:'),
                      sink.Feature('question'),
                      sink.Constant('summarize:'),
                      sink.Feature('article')
                  ]),
                  target=sink.Feature('summary'))

    sink.register('ebm/answer',
                  input=sink.Join([
                      sink.Constant('ebm answer'),
                      sink.Constant('question:'),
                      sink.Feature('question'),
                      sink.Constant('summarize:'),
                      sink.Sequence('article')
                  ]),
                  target=sink.Feature('summary'))

    sink.register('ebm/justify',
                  input=sink.Join([
                      sink.Constant('ebm justify'),
                      sink.Constant('question:'),
                      sink.Feature('question'),
                      sink.Constant('answer:'),
                      sink.Feature('answer'),
                      sink.Constant('summarize:'),
                      sink.Sequence('article')
                  ]),
                  target=sink.Feature('summary'))

    sink.register('medlineplus_references',
                  input=sink.Join([
                      sink.Constant('mpref'),
                      sink.Constant('summarize:'),
                      sink.Sequence('article')
                  ]),
                  target=sink.Feature('summary'))

    sink.register('medlineplus_reviews',
                  input=sink.Join([
                      sink.Constant('mprev'),
                      sink.Constant('summarize:'),
                      sink.Sequence('article')
                  ]),
                  target=sink.Feature('summary'))

    sink.register('medinfo',
                  input=sink.Join([
                      sink.Constant('medinfo'),
                      sink.Constant('question:'),
                      sink.Feature('question'),
                      sink.Constant('summarize:'),
                      sink.Feature('article')
                  ]),
                  target=sink.Feature('summary'))

    sink.register('pubmed_summ',
                  input=sink.Join([
                      sink.Constant('pubmed'),
                      sink.Constant('title:'),
                      sink.Feature('title'),
                      sink.Constant('summarize:'),
                      sink.Feature('article')
                  ]),
                  target=sink.Feature('summary'))

    sink.register('cochrane_summ/summary',
                  input=sink.Join([
                      sink.Constant('cochrane summary'),
                      sink.Constant('summarize:'),
                      sink.Feature('article')
                  ]),
                  target=sink.Feature('summary'))

    sink.register('cochrane_summ/clinical_answer',
                  input=sink.Join([
                      sink.Constant('cochrane question'),
                      sink.Constant('question:'),
                      sink.Feature('question'),
                      sink.Constant('summarize:'),
                      sink.Feature('article')
                  ]),
                  target=sink.Feature('answer'))

    sink.register('duc/2004',
                  input=sink.Join([
                      sink.Constant('duc 2004'),
                      sink.Constant('summarize:'),
                      sink.Feature('document')
                  ]),
                  target=sink.Feature('summary'))

    sink.register('duc/2007',
                  input=sink.Join([
                      sink.Constant('duc 2004'),
                      sink.Constant('summarize:'),
                      sink.Constant('question:'),
                      sink.Feature('question'),
                      sink.Sequence('document')
                  ]),
                  target=sink.Feature('summary'))

    sink.register('scientific_papers/arxiv',
                  input=sink.Join([
                      sink.Constant('arxiv'),
                      sink.Constant('summarize:'),
                      sink.Feature('article')
                  ]),
                  target=sink.Feature('abstract'))

    sink.register('scientific_papers/pubmed',
                  input=sink.Join([
                      sink.Constant('pubmed'),
                      sink.Constant('summarize:'),
                      sink.Feature('article')
                  ]),
                  target=sink.Feature('abstract'))

    sink.register('movie_rationales',
                  input=sink.Join([
                      sink.Constant('movies'),
                      sink.Constant('summarize:'),
                      sink.Sequence('evidences')
                  ]),
                  target=sink.Feature('review'))

    sink.register('cnn_dailymail',
                  input=sink.Join([
                      sink.Constant('cnn'),
                      sink.Constant('summarize:'),
                      sink.Feature('article')
                  ]),
                  target=sink.Feature('highlights'))

    sink.register('squad',
                  input=sink.Join([
                      sink.Constant('squad'),
                      sink.Constant('question:'),
                      sink.Feature('question'),
                      sink.Constant('context:'),
                      sink.Feature('context')
                  ]),
                  target=sink.Sequence(sink.DictEntry('answers', sink.Feature('text'))))

    sink.register('super_glue/copa',
                  input=sink.Join([
                      sink.Constant('binary copa'),
                      sink.Constant('choice1:'),
                      sink.Feature('choice1'),
                      sink.Constant('choice2:'),
                      sink.Feature('choice2'),
                      sink.Constant('premise:'),
                      sink.Feature('premise'),
                      sink.Constant('question:'),
                      sink.Feature('question')
                  ]),
                  target=sink.LabelSwitch('label', {
                      0: sink.Constant('True'),
                      1: sink.Constant('False')
                  }))

    _eviconv_stance_mapping = sink.LabelSwitch('stance', {
        0: sink.Constant('pro:'),
        1: sink.Constant('con:'),
    })
    sink.register('evi_conv',
                  input=sink.Join([
                      sink.Constant('binary argue'),
                      sink.Constant('choice1:'),
                      sink.DictEntry('evidence_1', _eviconv_stance_mapping),
                      sink.DictEntry('evidence_1', sink.Feature('text')),
                      sink.Constant('choice2:'),
                      sink.DictEntry('evidence_2', _eviconv_stance_mapping),
                      sink.DictEntry('evidence_2', sink.Feature('text')),
                  ]),
                  target=sink.LabelSwitch('label', {
                      0: sink.Constant('True'),
                      1: sink.Constant('False')
                  }))

    sink.register('tac/2009',
                  input=sink.Join([
                      sink.Constant('tac update'),
                      sink.Constant('question:'),
                      sink.Feature('topic'),
                      sink.Constant('summarize:'),
                      sink.Sequence('articles')
                  ]),
                  target=sink.Feature('summary'))

    sink.register('tac/2010',
                  input=sink.Join([
                      sink.Constant('tac update'),
                      sink.Constant('question:'),
                      sink.Feature('topic'),
                      sink.Constant('summarize:'),
                      sink.Sequence('articles')
                  ]),
                  target=sink.Feature('summary'))

    for config in [f'{d:d}.main.EN' for d in (2011, 2012, 2013)] + \
                  [f'{d:d}.alzheimers.EN' for d in (2012, 2013)]:
        sink.register(f'qa4mre/{config}',
                      input=sink.Join([
                          sink.Constant('qa4mre'),
                          sink.Feature('topic_name'),
                          sink.Constant('question:'),
                          sink.Feature('question_str'),
                          # Answer_options is a sequence of dictionary features, so we (1) create a Sequence mapping
                          sink.Sequence(
                              # ...and (2) indicate that the sequence is a Dictionary named 'answer_options'
                              sink.DictEntry(
                                  'answer_options',
                                  # ...and (3) indicate that we want to convert 'answer' options by prepending "choice:"
                                  # before each answer string
                                  sink.Join([
                                      sink.Constant('choice:'),
                                      sink.Feature('answer_str')
                                  ]))
                          ),
                          # Document goes last because they are crazy long
                          sink.Constant('context:'),
                          sink.Feature('document_str'),
                      ]),
                      target=sink.Feature('correct_answer_str'))

    sink.register('mctaco',
                  input=sink.Join([
                      sink.Constant('binary mctaco'),
                      sink.LabelSwitch('category', {
                          0: sink.Constant('duration'),
                          1: sink.Constant('ordering'),
                          2: sink.Constant('when'),
                          3: sink.Constant('frequency'),
                          4: sink.Constant('state')
                      }),
                      sink.Constant('question:'),
                      sink.Feature('question'),
                      sink.Constant('context:'),
                      sink.Feature('sentence'),
                      sink.Constant('answer:'),
                      sink.Feature('answer')
                  ]),
                  target=sink.LabelSwitch('label', {
                      0: sink.Constant('True'),
                      1: sink.Constant('False')
                  }))

    sink.register('cosmos_qa',
                  input=sink.Join([
                      sink.Constant('cosmos'),
                      sink.Constant('question:'),
                      sink.Feature('question'),
                      sink.Constant('context:'),
                      sink.Feature('context'),
                      *[mapping
                        for d in range(4)
                        for mapping in [sink.Constant(f'choice{d:d}:'), sink.Feature(f'answer{d:d}')]],
                  ]),
                  target=sink.LabelSwitch('label', {d: sink.Feature(f'answer{d:d}') for d in range(4)}))
