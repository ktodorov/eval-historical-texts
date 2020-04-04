import operator
import numpy as np
import torch
import random

from queue import PriorityQueue

from entities.beam_search_node import BeamSearchNode

from services.arguments.postocr_arguments_service import PostOCRArgumentsService
from services.vocabulary_service import VocabularyService


class DecodingService():
    def __init__(
            self,
            arguments_service: PostOCRArgumentsService,
            vocabulary_service: VocabularyService):

        self._arguments_service = arguments_service
        self._vocabulary_service = vocabulary_service

    def beam_decode(
            self,
            target_tensor,
            encoder_contexts,
            decoder_function):
        '''
        :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
        :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
        :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
        :return: decoded_batch
        '''

        beam_width = self._arguments_service.beam_width
        topk = 1  # how many sentence we want to generate
        decoded_batch = []
        batch_size = target_tensor.size(0)

        # decoding goes sentence by sentence
        for idx in range(batch_size):
            encoder_context = encoder_contexts[idx].unsqueeze(0)
            decoder_hidden = encoder_context

            # Start with the start of the target sentence
            decoder_input = target_tensor[idx][0].unsqueeze(0)

            # Number of sentence to generate
            endnodes = []
            number_required = 1  # min((topk + 1), topk - len(endnodes))

            # starting node
            node = BeamSearchNode(decoder_hidden, None,
                                  decoder_input, 0, 1, None)
            nodes = PriorityQueue()

            # start the queue
            nodes.put((-node.eval(), node))
            qsize = 1

            # start beam search
            while True:
                # give up when decoding takes too long
                if qsize > 1000:
                    break

                # fetch the best node
                score, n = nodes.get()
                decoder_input = n.wordid
                decoder_hidden = n.h

                if n.wordid.item() == self._vocabulary_service.eos_token and n.prevNode != None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                # decode for one step using decoder
                decoder_output, decoder_hidden = decoder_function(
                    decoder_input, decoder_hidden, encoder_context)

                # PUT HERE REAL BEAM SEARCH OF TOP
                log_prob, indexes = torch.topk(decoder_output, beam_width)
                nextnodes = []

                for new_k in range(beam_width):
                    decoded_t = indexes[0][new_k].reshape(1)
                    log_p = log_prob[0][new_k].item()

                    node = BeamSearchNode(
                        decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1, decoder_output)
                    score = -node.eval()
                    nextnodes.append((score, node))

                # put them into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
                    # increase qsize
                qsize += len(nextnodes) - 1

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            utterances = []
            score, n = endnodes[0]
            utterance = []
            utterance.append(n.decoder_output)
            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.decoder_output)

            utterance = utterance[::-1]

            decoded_batch.append(utterance[1:])

        sequence_lengths = [len(x) for x in decoded_batch]
        padded_decoded_batch = torch.zeros((batch_size, max(sequence_lengths), self._vocabulary_service.vocabulary_size(
        )), dtype=torch.float, device=self._arguments_service.device)
        for i, sequence_length in enumerate(sequence_lengths):
            decoded_tensor = torch.cat(decoded_batch[i])
            padded_decoded_batch[i][:sequence_length] = decoded_tensor

        return padded_decoded_batch, target_tensor[:, 1:]

    def greedy_decode(
            self,
            targets,
            encoder_context,
            decoder_function):

        trg_vocab_size = self._vocabulary_service.vocabulary_size()
        batch_size, trg_len = targets.shape
        teacher_forcing_ratio = self._arguments_service.teacher_forcing_ratio
        hidden = encoder_context

        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len,
                              trg_vocab_size, device=self._arguments_service.device)

        input = targets[:, 0]

        for t in range(0, trg_len):

            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden = decoder_function(input, hidden, encoder_context)

            outputs[:, t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            if teacher_force:
                input = targets[:, t]
            else:
                # get the highest predicted token from our predictions
                top1 = output.argmax(1)
                input = top1

        return outputs, targets
