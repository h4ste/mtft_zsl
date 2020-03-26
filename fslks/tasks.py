from fslks import sink

sink.register('bioasq',
              prompt=sink.Constant('summarize:'),
              input=sink.Sequence([
                  sink.Feature('question'),
                  sink.Feature('article')
              ]),
              output=sink.Feature('summary'))

sink.register('scientific_papers',
              prompt=sink.Constant('summarize:'),
              input=sink.Feature('article'),
              output=sink.Feature('abstract'))

sink.register('movie_rationales',
              prompt=sink.Constant('summarize:'),
              input=sink.Sequence('evidences'),
              output=sink.Feature('review'))

sink.register('cnn_dailymail',
              prompt=sink.Constant('summarize:'),
              input=sink.Feature('article'),
              output=sink.Feature('highlights'))

sink.register('super_glue/copa',
              prompt=sink.Constant('choose:'),
              input=sink.Sequence([
                  sink.Feature('question'),
                  sink.Feature('premise'),
                  sink.Feature('choice1'),
                  sink.Feature('choice2'),
              ]),
              output=sink.LabelMapping('label', {
                  0: sink.Feature('choice1'),
                  1: sink.Feature('choice2')
              }))
