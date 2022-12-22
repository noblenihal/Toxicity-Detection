

class BaseTokenizer(object):
    def process_text(self, text):
        raise NotImplemented

    def process(self, texts):
        for text in texts:
            yield self.process_text(text)


def read_lines_from_model(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f]
        return lines