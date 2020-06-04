import torch
from torch import nn
from torch.functional import F
import random

from overrides import overrides
from queue import PriorityQueue

from enums.embedding_type import EmbeddingType
from entities.batch_representation import BatchRepresentation
from entities.options.embedding_layer_options import EmbeddingLayerOptions
from entities.options.pretrained_representations_options import PretrainedRepresentationsOptions
from entities.beam_search_node import BeamSearchNode

from models.embedding.embedding_layer import EmbeddingLayer
from models.rnn_encoder_decoder.sequence_attention import SequenceAttention

from services.file_service import FileService
from models.model_base import ModelBase


class SequenceDecoder(ModelBase):
    def __init__(
            self,
            file_service: FileService,
            device: str,
            embedding_size: int,
            attention_dimension: int,
            encoder_hidden_dimension: int,
            decoder_hidden_dimension: int,
            number_of_layers: int,
            output_dimension: int,
            vocabulary_size: int,
            pad_idx: int,
            dropout: float = 0,
            use_own_embeddings: bool = True,
            shared_embedding_layer: EmbeddingLayer = None,
            use_beam_search: bool = False,
            beam_width: int = None,
            teacher_forcing_ratio: float = None,
            eos_token: int = None):
        super().__init__()

        self._device = device
        self._use_beam_search = use_beam_search
        self._beam_width = beam_width
        self._vocabulary_size = vocabulary_size
        self._teacher_forcing_ratio = teacher_forcing_ratio

        self._eos_token = eos_token
        self._pad_idx = pad_idx

        if not use_own_embeddings:
            if shared_embedding_layer is None:
                raise Exception('Shared embeddings not supplied')
            self._embedding_layer = shared_embedding_layer
        else:
            self._embedding_layer = EmbeddingLayer(
                file_service,
                EmbeddingLayerOptions(
                    device=device,
                    pretrained_representations_options=PretrainedRepresentationsOptions(
                        include_pretrained_model=False),
                    learn_character_embeddings=True,
                    include_pretrained_model=False,
                    vocabulary_size=output_dimension,
                    character_embeddings_size=embedding_size,
                    dropout=dropout,
                    output_embedding_type=EmbeddingType.Character))

        self._rnn_layer = nn.GRU(
            embedding_size + decoder_hidden_dimension,
            encoder_hidden_dimension,
            number_of_layers,
            batch_first=True)

        self._attention_layer = SequenceAttention(
            hidden_size=attention_dimension)

        # to initialize from the final encoder state
        self._bridge_layer = nn.Linear(
            decoder_hidden_dimension, encoder_hidden_dimension, bias=True)

        self._pre_output_layer = nn.Linear(
            encoder_hidden_dimension + decoder_hidden_dimension + embedding_size,
            encoder_hidden_dimension,
            bias=False)

        self.dropout_layer = nn.Dropout(dropout)

    @overrides
    def forward(self, targets, encoder_hidden, encoder_final, src_mask, hidden=None):
        if self._use_beam_search:
            outputs, targets = self._beam_decode(encoder_final, targets)
        else:
            out, hidden, pre_output = self._greedy_decode(
                encoder_hidden, encoder_final, targets, src_mask, hidden=hidden)

        # return outputs, targets
        return pre_output, hidden

    def _beam_decode(
            self,
            encoder_contexts,
            target_tensor):
        '''
        :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
        :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
        :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
        :return: decoded_batch
        '''

        beam_width = self._beam_width
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

            nodes: PriorityQueue[BeamSearchNode] = PriorityQueue()

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

                if current_node.word_id.item() == self._eos_token and current_node.previous_node != None:
                    endnodes.append((current_node_score, current_node))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                # decode for one step using decoder
                decoder_output, decoder_hidden = self._internal_forward(
                    decoder_input,
                    decoder_hidden,
                    encoder_context)

                # PUT HERE REAL BEAM SEARCH OF TOP
                log_output = F.log_softmax(
                    decoder_output, dim=1) + current_node.log_probability
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
            utterance = torch.zeros(
                (1, current_node.length - 1, current_node.decoder_output.shape[1]), device=self._device)
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
        padded_decoded_batch = torch.zeros((batch_size, max(
            sequence_lengths), self._vocabulary_size), dtype=torch.float, device=self._device)
        for i, sequence_length in enumerate(sequence_lengths):
            padded_decoded_batch[i][:sequence_length] = decoded_batch[i]

        return padded_decoded_batch, target_tensor[:, 1:]

    def init_hidden(self, encoder_final):
        """Returns the initial decoder state,
        conditioned on the final encoder state."""

        if encoder_final is None:
            return None  # start with zeros

        return torch.tanh(self._bridge_layer.forward(encoder_final))

    def _greedy_decode(self, encoder_hidden, encoder_final, targets, src_mask, hidden=None):
        # the maximum number of steps to unroll the RNN
        batch_size, max_len = targets.shape

        # initialize decoder hidden state
        if hidden is None:
            hidden = self.init_hidden(encoder_final)

        input_batch = BatchRepresentation(
            device=self._device,
            batch_size=batch_size,
            character_sequences=targets)

        target_embeddings = self._embedding_layer.forward(input_batch)

        # pre-compute projected encoder hidden states
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        proj_key = self._attention_layer.key_layer(encoder_hidden)

        # here we store all intermediate hidden states and pre-output vectors
        decoder_states = []
        pre_output_vectors = []

        # unroll the decoder RNN for max_len steps
        for i in range(max_len):
            prev_embed = target_embeddings[:, i].unsqueeze(1)
            output, hidden, pre_output = self._internal_forward(
                prev_embed,
                encoder_hidden,
                src_mask,
                proj_key,
                hidden)

            decoder_states.append(output)
            pre_output_vectors.append(pre_output)

        decoder_states = torch.cat(decoder_states, dim=1)
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        return decoder_states, hidden, pre_output_vectors  # [B, N, D]

    def _internal_forward(self, prev_embed, encoder_hidden, src_mask, proj_key, hidden):
        """Perform a single decoder step (1 character)"""

        # compute context vector using attention mechanism
        query = hidden[-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]
        context, attn_probs = self._attention_layer.forward(
            query=query, proj_key=proj_key,
            value=encoder_hidden, mask=src_mask)

        # update rnn hidden state
        rnn_input = torch.cat([prev_embed, context], dim=2)
        output, hidden = self._rnn_layer.forward(rnn_input, hidden)

        pre_output = torch.cat([prev_embed, output, context], dim=2)
        pre_output = self.dropout_layer(pre_output)
        pre_output = self._pre_output_layer(pre_output)

        return output, hidden, pre_output
