import operator
import numpy as np
import torch
from torch.functional import F
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
        batch_size, target_length = target_tensor.shape

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
            node = BeamSearchNode(
                hiddenstate=decoder_hidden,
                previousNode=None,
                wordId=decoder_input,
                logProb=0,
                length=1,
                decoder_output=None)

            nodes:PriorityQueue[BeamSearchNode] = PriorityQueue()

            # start the queue
            nodes.put((node.eval(), node))
            qsize = 1

            # start beam search
            while True:
                # give up when decoding takes too long
                if qsize > 2000:
                    break

                # fetch the best node
                current_node_score, current_node = nodes.get()
                if current_node.length > target_length:
                    break

                decoder_input = current_node.word_id
                decoder_hidden = current_node.h

                if current_node.word_id.item() == self._vocabulary_service.eos_token and current_node.previous_node != None:
                    endnodes.append((current_node_score, current_node))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                # decode for one step using decoder
                decoder_output, decoder_hidden = decoder_function(
                    decoder_input, decoder_hidden, encoder_context)

                # PUT HERE REAL BEAM SEARCH OF TOP
                log_output = F.log_softmax(decoder_output, dim=1) + current_node.log_probability
                log_probabilities, indexes = torch.topk(log_output, beam_width)
                for new_k in range(beam_width):
                    decoded_t = indexes[0][new_k].reshape(1)
                    log_probability = log_probabilities[0][new_k].item()

                    node = BeamSearchNode(
                        hiddenstate=decoder_hidden,
                        previousNode=current_node,
                        wordId=decoded_t,
                        logProb=log_probability,
                        length=current_node.length + 1,
                        decoder_output=decoder_output)

                    current_node_score = node.eval()
                    nodes.put((current_node_score, node))

                # increase qsize
                qsize += beam_width - 1

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            current_node_score, current_node = endnodes[0]
            utterance = torch.zeros((1, current_node.length - 1, current_node.decoder_output.shape[1]), device=self._arguments_service.device)
            utterance[0][-1] = current_node.decoder_output
            counter = 2
            # back trace
            while current_node.previous_node != None:
                current_node = current_node.previous_node
                if current_node.decoder_output is not None:
                    utterance[0][-counter] = current_node.decoder_output

                counter += 1

            decoded_batch.append(utterance)

        sequence_lengths = [x.shape[1] for x in decoded_batch]
        padded_decoded_batch = torch.zeros((batch_size, max(sequence_lengths), self._vocabulary_service.vocabulary_size(
        )), dtype=torch.float, device=self._arguments_service.device)
        for i, sequence_length in enumerate(sequence_lengths):
            padded_decoded_batch[i][:sequence_length] = decoded_batch[i]

        return padded_decoded_batch, target_tensor[:, 1:]

    def greedy_decode(
            self,
            targets,
            encoder_context,
            decoder_function):

        trg_vocab_size = self._vocabulary_service.vocabulary_size()
        batch_size, trg_len = targets.shape
        teacher_forcing_ratio = self._arguments_service.teacher_forcing_ratio
        hidden = encoder_context.permute(1, 0, 2)

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
