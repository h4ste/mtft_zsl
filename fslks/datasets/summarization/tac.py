"""
Module to create MEDIQA-Answer Summarization for Consumer Health Information Question Answering tensorflow dataset. 
"""
import glob
import os
from xml.etree import ElementTree

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
import tensorflow_datasets.public_api as tfds


_DESCRIPTION = """TBD."""

_CITATION = """This work is 100% plagiarized"""

_TAC_DOWNLOAD_INSTRUCTIONS = """Temp until upload"""

_ACQUAINT_HEADER = """<?xml version="1.0" encoding="ISO-8859-1" standalone="yes" ?>
<!DOCTYPE DOC [

<!-- fields of DOC -->
<!ELEMENT  DOC         (DOCNO*, DOCTYPE*, DATE_TIME*, HEADER*, BODY*, TRAILER*) >

<!-- fields of "DOC" -->
<!ELEMENT  DOCNO       (#PCDATA)                             >
<!ELEMENT  DOCTYPE     (#PCDATA)                             >
<!ELEMENT  TXTTYPE     (#PCDATA)                             >
<!ELEMENT  DATE_TIME   (#PCDATA)                             >
<!ELEMENT  HEADER      (#PCDATA)                             >
<!ELEMENT  BODY        (#PCDATA | SLUG | CATEGORY | HEADLINE | TEXT)* >
<!ELEMENT  TRAILER     (#PCDATA)                             >

<!-- fields of "BODY" -->
<!ELEMENT  SLUG        (#PCDATA)                             >
<!ELEMENT  HEADLINE    (#PCDATA)                             >
<!ELEMENT  CATEGORY    (#PCDATA)                             >
<!ELEMENT  TEXT        (#PCDATA | P | SUBHEAD | ANNOTATION)* >

<!--fields of "TEXT" -->  
<!ELEMENT  P           (#PCDATA)                             >
<!ELEMENT  SUBHEAD     (#PCDATA)                             >
<!ELEMENT  ANNOTATION  (#PCDATA)                             >

<!--Entities -->
<!ENTITY   AMP    "&amp;" >
<!ENTITY   Cx14   "" >
<!ENTITY   Cx13   "" >
<!ENTITY   Cx12   "" >
<!ENTITY   Cx11   "" >
<!ENTITY   Cx1f   "" >
<!ENTITY   HT     "" >
<!ENTITY   QL     "" >
<!ENTITY   QR     "" >
<!ENTITY   LR     "" >
<!ENTITY   UR     "" >
<!ENTITY   QC     "" >
]>
"""


class TacConfig(tfds.core.BuilderConfig):
    """Builder config for MEDIQA-Answer Summarization"""

    @tfds.core.disallow_positional_args
    def __init__(self,
                 base_dir,
                 **kwargs):
        """Config for MEDIQA-Ans.

        Args:
          single_doc: `bool`, for single or multi-doc summarization
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(version=tfds.core.Version("1.0.0"), **kwargs)
        self.base_dir = base_dir


class Tac(tfds.core.GeneratorBasedBuilder):
    """MEDIQA-AnS dataset builder"""

    MANUAL_DOWNLOAD_INSTRUCTIONS = _TAC_DOWNLOAD_INSTRUCTIONS

    BUILDER_CONFIGS = [
            TacConfig(name='2009', base_dir='update_2009', description="multi-document abstractive summarization"),
            TacConfig(name='2010', base_dir='guided_2010', description="multi-document abstractive summarization"),
    ]

    def __init__(self, include_update=False, **kwargs):
        super().__init__(**kwargs)
        self.include_update = include_update

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'topic': tfds.features.Text(),
                'articles': tfds.features.Sequence(tfds.features.Text()),
                'summary': tfds.features.Text(),
            }),
            supervised_keys=('articles', 'summary'),
            citation=_CITATION
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        path = os.path.join(dl_manager.manual_dir, self.name, self.builder_config.base_dir)
        return [
                tfds.core.SplitGenerator(name=tfds.Split.TEST,
                                         gen_kwargs={
                                             'topics_path': os.path.join(path, 'test_topics.xml'),
                                             'summaries_path': os.path.join(path, 'models'),
                                             'documents_path': os.path.join(path, 'test_docs')
                                         }),
        ]

    def _generate_examples(self, topics_path: str, summaries_path: str, documents_path: str):
        docsets = {'docsetA'}
        if self.include_update:
            docsets.add('docsetB')

        vectorizer = TfidfVectorizer(min_df=1, stop_words="english")
        """Parse and yield TAC collection"""
        with tf.io.gfile.GFile(topics_path) as f:
            try:
                tree = ElementTree.parse(f)
            except IOError as ioe:
                print('Failed to parse', topics_path)
                raise ioe
            root = tree.getroot()
            assert root.tag == 'TACtaskdata'

            for topic_ in root.findall('topic'):
                topic = topic_.findtext('narrative') or topic_.findtext('title')
                topic_id = topic_.attrib['id']

                for docset in docsets:
                    docset_ = topic_.find(docset)
                    docset_id = docset_.attrib['id']

                    # For some reason, the topics in the topic xml file all end with A, but in the summaries the A
                    # mysteriously disappears
                    adjusted_docset_id = docset_id.replace(topic_id, topic_id[:-1])
                    summary_files = os.path.join(summaries_path, '%s.M.100.[A-H].[A-H]' % adjusted_docset_id)
                    summaries = []
                    for summary_file in glob.glob(summary_files):
                        with tf.io.gfile.GFile(summary_file, 'rb') as g:
                            try:
                                summary = g.read().decode('utf-8', errors='ignore')
                            except UnicodeDecodeError as ude:
                                print('Failed to read file', summary_file)
                                raise ude
                            summaries.append(summary)

                    documents = []
                    for doc_ in docset_.findall('doc'):
                        doc_id = doc_.attrib['id']
                        doc_path = os.path.join(documents_path, topic_id, docset_id, doc_id)
                        with tf.io.gfile.GFile(doc_path) as h:
                            xml_str = h.read()
                        xml_str = _ACQUAINT_HEADER + xml_str
                        try:
                            doc_root = ElementTree.fromstring(xml_str)
                        except ElementTree.ParseError as ioe:
                            print('Failed to parse', doc_path)
                            raise ioe
                        # doc_root = doc_tree.getroot()
                        assert doc_root.tag == 'DOC'
                        text_root = doc_root.find('BODY') or doc_root
                        doc_text = '\n'.join(text.strip() for text in (text_root.find('TEXT').itertext()))
                        documents.append(doc_text)

                    summary_vecs = vectorizer.fit_transform(summaries)
                    doc_vecs = vectorizer.transform(documents)
                    similarities = (summary_vecs * doc_vecs.T).A
                    for sid, (summary, doc_similarities) in enumerate(zip(summaries, similarities), start=1):
                        _, sorted_documents = zip(*sorted(zip(doc_similarities, documents), reverse=True))
                        yield ('%s-s%d' % (docset_id, sid), {
                            'topic': topic,
                            'articles': sorted_documents,
                            'summary': summary
                        })
