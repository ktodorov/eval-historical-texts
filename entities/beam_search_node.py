class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length, decoder_output):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.previous_node = previousNode
        self.word_id = wordId
        self.log_probability = logProb
        self.length = length
        self.decoder_output = decoder_output

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.log_probability# / float(self.length - 1 + 1e-6) + alpha * reward

    def __lt__(self, other):
        if other is None:
            return False

        return self.eval() < other.eval()

    def __eq__(self, other):
        if other is None:
            return False

        return self.eval() == other.eval()