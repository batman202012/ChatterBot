import string
from chatterbot import languages


class Tagger:

    def __init__(self, language=None):
        self.language = language or languages.ENG

    def get_text_index_string(self, text):
        return text

    def get_text_index_string_multi(self, texts):
        return texts


class LowercaseTagger(Tagger):
    """
    Returns the text in lowercase.
    """

    def get_text_index_string(self, text):
        return text.lower()


class PosLemmaTagger(Tagger):

    def __init__(self, language=None):
        super().__init__(language)
        import spacy

        self.punctuation_table = str.maketrans(dict.fromkeys(string.punctuation))

        self.nlp = spacy.load(self.language.ISO_639_1.lower(), disable=["transformer", "parser", "ner"])

    def punctuation_check(self, text):
        if len(text) <= 2:
            text_without_punctuation = text.translate(self.punctuation_table)
            if len(text_without_punctuation) >= 1:
                return text_without_punctuation
        return text

    def get_text_index_string_multi(self, texts):
        new_texts = [self.punctuation_check(text) for text in texts]

        return [self._process_document(doc) for doc in self.nlp.pipe(new_texts)]

    def get_text_index_string(self, text):
        """
        Return a string of text containing part-of-speech, lemma pairs.
        """
        text = self.punctuation_check(text)
        document = self.nlp(text)
        return self._process_document(document)

    def _process_document(self, document):
        bigram_pairs = []
        text = document.text

        if len(text) <= 2:
            bigram_pairs = [
                token.lemma_.lower() for token in document
            ]
        else:
            tokens = [
                token for token in document if token.is_alpha and not token.is_stop
            ]

            if len(tokens) < 2:
                tokens = [
                    token for token in document if token.is_alpha
                ]

            if len(tokens) > 512:
                tokens = tokens[:512]

            for index in range(1, len(tokens)):
                bigram_pairs.append('{}:{}'.format(
                    tokens[index - 1].pos_,
                    tokens[index].lemma_.lower()
                ))

        if not bigram_pairs:
            bigram_pairs = [
                token.lemma_.lower() for token in document
            ]

        return ' '.join(bigram_pairs)
