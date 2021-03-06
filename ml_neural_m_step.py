import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import utils
import numpy as np
from numpy import linalg as LA
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from allennlp.modules.elmo import Elmo, batch_to_ids



class m_step_model(nn.Module):
    def __init__(self, tag_num, word_num, lan_num, words, options):
        super(m_step_model, self).__init__()
        self.tag_num = tag_num
        self.word_num = word_num
        self.options = options
        self.cvalency = options.c_valency
        self.dvalency = options.d_valency
        self.drop_out = options.drop_out
        self.child_only = options.child_only
        self.gpu = options.gpu
        self.pembedding_dim = options.pembedding_dim
        self.valency_dim = options.valency_dim
        self.hid_dim = options.hid_dim
        self.pre_output_dim = options.pre_output_dim  # child token dimention
        self.pre_output_word_dim = options.pre_output_word_dim  # elmo dimention: 1024
        self.unified_network = options.unified_network
        self.decision_pre_output_dim = options.decision_pre_output_dim
        self.drop_out = options.drop_out
        self.lan_num = lan_num
        self.ml_comb_type = options.ml_comb_type  # options.ml_comb_type = 0(no_lang_id)/1(id embeddings)/2(classify-tags)
        self.stc_model_type = options.stc_model_type  # 1  lstm   2 lstm with atten   3 variational
        self.non_dscrm_iter = options.non_dscrm_iter

        self.full_lex = options.full_lex
        self.input_word_dim = options.input_word_dim
        self.options_file = '/home/hanwj/Code/l_dmv/data/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json'  # "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        self.weight_file = '/home/hanwj/Code/l_dmv/data/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'  # "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        self.elmo = Elmo(self.options_file, self.weight_file, 2, dropout=0, requires_grad=False)
        self.i2word = {p: i for i, p in words.items()}

        if self.ml_comb_type == 1:
            self.lang_dim = options.lang_dim  # options.lang_dim = 10(default)

        elif self.ml_comb_type == 2:
            self.lang_dim = options.lang_dim
            self.lstm_layer_num = options.lstm_layer_num  # 1
            self.lstm_hidden_dim = options.lstm_hidden_dim  # 10
            self.bidirectional = options.bidirectional  # True
            self.lstm_direct = 2 if self.bidirectional else 1
            self.hidden = self.init_hidden(1)  # 1 here is just for init, will be changed in forward process
            self.lstm = nn.LSTM(self.pembedding_dim, self.lstm_hidden_dim, num_layers=self.lstm_layer_num,
                                bidirectional=self.bidirectional,
                                batch_first=True)  # hidden_dim // 2, num_layers=1, bidirectional=True
            self.lang_classifier = nn.Linear(self.lstm_direct * self.lstm_hidden_dim, self.lan_num)
            if self.stc_model_type == 2:
                self.max_length = 40  # TODO, we train or test on len40
                self.nhid = self.pembedding_dim # for attention
                self.hvds_dim = self.nhid + self.lstm_hidden_dim * self.lstm_direct * self.lstm_layer_num  # for sts
                self.linear_hvds = nn.Linear(self.hvds_dim, self.max_length)

            if self.stc_model_type == 3:
                self.variational_mu = nn.Linear(self.lstm_hidden_dim * self.lstm_direct * self.lstm_layer_num, self.lstm_direct * self.lstm_hidden_dim)
                self.variational_logvar = nn.Linear(self.lstm_hidden_dim * self.lstm_direct * self.lstm_layer_num,
                                                    self.lstm_direct * self.lstm_hidden_dim)  # log var.pow(2)
        self.plookup = nn.Embedding(self.tag_num, self.pembedding_dim)
        if self.full_lex:
            self.p_word_lookup = self.elmo#nn.Embedding(self.word_num, self.pre_output_word_dim)  # pembedding_dim of elmo
            # self.p_word_lookup.weight.requires_grad = False  # TODO:hanwj
            self.m_word_lookup = nn.Embedding(self.word_num, self.pre_output_word_dim)  # self.p_word_lookup  # pembedding_dim of elmo
            self.m_word_lookup.weight.requires_grad = False  # TODO:hanwj
            self.hid_before_dir = nn.Linear(self.pembedding_dim + self.valency_dim + self.input_word_dim, self.pembedding_dim + self.valency_dim)
        self.dplookup = nn.Embedding(self.tag_num, self.pembedding_dim)
        self.vlookup = nn.Embedding(self.cvalency, self.valency_dim)
        self.dvlookup = nn.Embedding(self.dvalency, self.valency_dim)
        self.head_lstm_embeddings = self.plookup
        if self.ml_comb_type == 1:
            self.llookup = nn.Embedding(self.lan_num, self.lang_dim)

        self.dropout_layer = nn.Dropout(p=self.drop_out)

        self.dir_embed = options.dir_embed
        self.dir_dim = options.dir_dim
        if self.dir_embed:
            self.dlookup = nn.Embedding(2, self.dir_dim)
        if not self.dir_embed:
            if self.ml_comb_type == 0:
                self.left_hid = nn.Linear((self.pembedding_dim + self.valency_dim), self.hid_dim)
                self.right_hid = nn.Linear((self.pembedding_dim + self.valency_dim), self.hid_dim)
            elif self.ml_comb_type == 1:
                self.left_hid = nn.Linear((self.pembedding_dim + self.valency_dim + self.lang_dim), self.hid_dim)
                self.right_hid = nn.Linear((self.pembedding_dim + self.valency_dim + self.lang_dim), self.hid_dim)
            elif self.ml_comb_type == 2:
                self.left_hid = nn.Linear(
                    (self.pembedding_dim + self.valency_dim + self.lstm_direct * self.lstm_hidden_dim), self.hid_dim)
                self.right_hid = nn.Linear(
                    (self.pembedding_dim + self.valency_dim + self.lstm_direct * self.lstm_hidden_dim),
                    self.hid_dim)


        else:
            self.hid = nn.Linear((self.pembedding_dim + self.valency_dim + self.dir_dim), self.hid_dim)

        if self.full_lex:
            self.linear_chd_hid = nn.Linear(self.hid_dim, self.pre_output_dim)
            self.linear_chd_word_hid = nn.Linear(self.hid_dim, self.pre_output_word_dim)
            self.pre_word_output = nn.Linear(self.pre_output_word_dim, self.word_num)
            self.pre_word_output.weight = self.m_word_lookup.weight
            self.pre_output = nn.Linear(self.pre_output_dim, self.tag_num)
            # lang_output = torch.mm(lang_output_before, torch.transpose(self.llookup.weight, 0, 1))
        else:
            self.linear_chd_hid = nn.Linear(self.hid_dim, self.pre_output_dim)
            self.pre_output = nn.Linear(self.pre_output_dim, self.tag_num)

        if not self.dir_embed:
            self.left_decision_hid = nn.Linear((self.pembedding_dim + self.valency_dim), self.hid_dim)
            self.right_decision_hid = nn.Linear((self.pembedding_dim + self.valency_dim), self.hid_dim)
        else:
            self.decision_hid = nn.Linear((self.pembedding_dim + self.valency_dim + self.dir_dim), self.hid_dim)
        self.linear_decision_hid = nn.Linear(self.hid_dim, self.decision_pre_output_dim)
        self.decision_pre_output = nn.Linear(self.decision_pre_output_dim, 2)
        # self.decision_pre_output = nn.Linear(self.hid_dim, 2)
        self.em_type = options.em_type
        self.param_smoothing = options.param_smoothing

        self.optim_type = options.optim_type
        self.lr = options.learning_rate
        if self.optim_type == 'sgd':
            self.optim = optim.SGD(self.parameters(), lr=self.lr)
        elif self.optim_type == 'adam':
            self.optim = optim.Adam(self.parameters(), lr=self.lr)
        elif self.optim_type == 'adagrad':
            self.optim = optim.Adagrad(self.parameters(), lr=self.lr)

    def init_hidden(self, batch_init):
        return (
            torch.zeros(self.lstm_layer_num * self.lstm_direct, batch_init, self.lstm_hidden_dim),
            # num_layers * bi-direction
            torch.zeros(2 * 1, batch_init, self.lstm_hidden_dim))

    def reparameterize(self, training, mu, logvar):
        if training:
            std = torch.exp(0.5 * logvar)
            eps = torch.autograd.Variable(torch.randn(std.size()), requires_grad=False)  # torch.randn_like(std)
            temp = torch.mul(std, eps) + mu
            return temp  # eps.mul(std).add_(mu)
        else:
            return mu

    def lang_loss(self, stc_representation, batch_target):

        loss = torch.nn.CrossEntropyLoss()
        lang_output = self.lang_classifier(stc_representation)
        lang_loss = loss(lang_output, batch_target)
        return lang_loss


    def stc_representation(self, sentences, sentences_len, hid_tensor):
        batch_size = len(sentences)
        sentences_maxlen = len(sentences[0])
        if self.stc_model_type == 1:
            embeds = self.head_lstm_embeddings(sentences)
            lstm_out, self.hidden = self.lstm(embeds)
            sentences_all_lstm = torch.transpose(self.hidden[0], 0, 1)
            sentences_all_lstm = sentences_all_lstm.contiguous().view(sentences_all_lstm.size()[0], -1)
            return self.dropout_layer(sentences_all_lstm)
        elif self.stc_model_type == 2:
            embeds = self.head_lstm_embeddings(torch.autograd.Variable(torch.LongTensor(sentences)))
            lstm_out, self.hidden = self.lstm(embeds)  # [0]# sentence_in.view(BATCH_SIZE, BATCH_SIZE, -1)
            sentences_lstm = torch.transpose(self.hidden[0], 0, 1).contiguous().view(batch_size,
                                                                                     -1)  # batch_size* (num_layer*direct*hiddensize) #use h not c
            atten_weight = F.softmax(self.linear_hvds(torch.cat((hid_tensor, sentences_lstm), 1)))[:, 0:sentences_maxlen]
            attn_applied = torch.bmm(torch.transpose(atten_weight.unsqueeze(2), 1, 2), lstm_out)  # 1*1*6
            return attn_applied.squeeze(1)
        elif self.stc_model_type == 3:
            embeds = self.dropout_layer(self.head_lstm_embeddings(torch.autograd.Variable(torch.LongTensor(sentences))))
            lstm_out, self.hidden = self.lstm(embeds)  # [0]# sentence_in.view(BATCH_SIZE, BATCH_SIZE, -1)
            sentences_all_lstm = torch.transpose(self.hidden[0], 0, 1)
            sentences_all_lstm = sentences_all_lstm.contiguous().view(sentences_all_lstm.size()[0], -1)
            mu = self.variational_mu(sentences_all_lstm)
            logvar = self.variational_logvar(sentences_all_lstm)
            var_out = self.reparameterize(self.training, mu, logvar)
            return var_out, -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def forward_(self, batch_pos, batch_word, batch_index, batch_dir, batch_valence, batch_target, batch_target_word, batch_target_count,
                 is_prediction, type, em_type, batch_lang_id, sentences_mlist, sentences, sentences_len, epoch):
        p_embeds = self.plookup(batch_pos)
        if self.full_lex:
            sentences_mlist = [[self.i2word[j] for j in i] for i in sentences_mlist]
            batch_ws_id = batch_to_ids(sentences_mlist)
            batch_ws_emb = self.elmo(batch_ws_id)
            pw_embeds = batch_ws_emb['elmo_representations'][0][range(len(batch_index)), batch_index, :]
            # pw_embeds = self.p_word_lookup(batch_word)
        if type == 'child':
            v_embeds = self.vlookup(batch_valence)
        else:
            v_embeds = self.dvlookup(batch_valence)
        if self.dir_embed:
            d_embeds = self.dlookup(batch_dir)
        if not self.dir_embed:
            left_mask, right_mask = self.construct_mask(batch_dir)
            if self.ml_comb_type == 0:
                input_embeds = torch.cat((p_embeds, v_embeds), 1)
            elif self.ml_comb_type == 1:
                lang_embeds = self.llookup(batch_lang_id)
                input_embeds = torch.cat((p_embeds, v_embeds, lang_embeds), 1)
            elif self.ml_comb_type == 2:
                stc_representation_and_vae_loss = self.stc_representation(sentences, sentences_len, p_embeds)
                stc_representation = stc_representation_and_vae_loss[0] if isinstance(stc_representation_and_vae_loss,
                                                                                      tuple) else stc_representation_and_vae_loss
                vae_loss = stc_representation_and_vae_loss[1] if isinstance(stc_representation_and_vae_loss,
                                                                            tuple) else 0
                stc_representation = stc_representation * 0 if epoch < self.non_dscrm_iter else stc_representation
                vae_loss = vae_loss * 0 if epoch < self.non_dscrm_iter else vae_loss
                input_embeds = torch.cat((p_embeds, v_embeds, stc_representation), 1)
                if not is_prediction:
                    lang_cls_loss = self.lang_loss(stc_representation, batch_lang_id)
            if self.full_lex:
                input_embeds = torch.cat((input_embeds, pw_embeds), 1)
                input_embeds = F.relu(self.hid_before_dir(input_embeds))
            input_embeds = self.dropout_layer(input_embeds)
            left_v = self.left_hid(input_embeds)
            left_v = F.relu(left_v)
            right_v = self.right_hid(input_embeds)
            right_v = F.relu(right_v)
            left_v = left_v.masked_fill(left_mask, 0.0)
            right_v = right_v.masked_fill(right_mask, 0.0)
            hidden_v = left_v + right_v
        else:
            input_embeds = torch.cat((p_embeds, v_embeds, d_embeds), 1)
            hidden_v = self.hid(input_embeds)
        if type == 'child':
            pre_output_v = self.pre_output(F.relu(self.linear_chd_hid(hidden_v)))
            if self.full_lex:
                pre_output_word_v = self.pre_word_output(F.relu(self.linear_chd_word_hid(hidden_v)))
        else:
            pre_output_v = self.decision_pre_output(F.relu(self.linear_decision_hid(hidden_v)))
        if not is_prediction:
            if em_type == 'viterbi':
                loss = torch.nn.CrossEntropyLoss()
                batch_loss = loss(pre_output_v, batch_target)
                return batch_loss
            else:
                predicted_prob = F.log_softmax(pre_output_v, dim=1)
                batch_target = batch_target.view(len(batch_target), 1)
                target_prob = torch.gather(predicted_prob, 1, batch_target)
                batch_target_count = batch_target_count.view(len(batch_target_count), 1)
                batch_loss = -torch.sum(batch_target_count * target_prob)
                if self.full_lex:
                    predicted_word_prob = F.log_softmax(pre_output_word_v, dim=1)
                    batch_target_word = batch_target_word.view(len(batch_target_word), 1)
                    target_word_prob = torch.gather(predicted_word_prob, 1, batch_target_word)
                    batch_loss += -torch.sum(batch_target_count * target_word_prob)

                if self.ml_comb_type == 2:
                    batch_loss += lang_cls_loss
                if self.stc_model_type == 3:
                    batch_loss += vae_loss
                return batch_loss
        else:
            predicted_param = F.softmax(pre_output_v, dim=1)
            predicted_word_param = F.softmax(pre_output_word_v, dim=1) if self.full_lex else None
            return predicted_param, predicted_word_param

    def forward_decision(self, batch_decision_pos, batch_decision_dir, batch_dvalence, batch_target_decision,
                         batch_target_decision_count, is_prediction, em_type):
        p_embeds = self.dplookup(batch_decision_pos)
        v_embeds = self.dvlookup(batch_dvalence)

        if self.dir_embed:
            d_embeds = self.dlookup(batch_decision_dir)
        if not self.dir_embed:

            left_mask, right_mask = self.construct_mask(batch_decision_dir)
            input_embeds = torch.cat((p_embeds, v_embeds), 1)

            left_v = self.left_decision_hid(input_embeds)
            left_v = F.relu(left_v)
            right_v = self.right_decision_hid(input_embeds)
            right_v = F.relu(right_v)
            left_v = left_v.masked_fill(left_mask, 0.0)
            right_v = right_v.masked_fill(right_mask, 0.0)
            hidden_v = left_v + right_v
        else:
            input_embeds = torch.cat((p_embeds, v_embeds, d_embeds), 1)
            hidden_v = self.decision_hid(input_embeds)
        pre_output_v = self.decision_pre_output(F.relu(self.linear_decision_hid(hidden_v)))
        if not is_prediction:
            if em_type == 'viterbi':
                loss = torch.nn.CrossEntropyLoss()
                batch_loss = loss(pre_output_v, batch_target_decision)
                return batch_loss
            else:
                predicted_prob = F.log_softmax(pre_output_v, dim=1)
                batch_target = batch_target_decision.view(len(batch_target_decision), 1)
                target_prob = torch.gather(predicted_prob, 1, batch_target)
                batch_target_count = batch_target_decision_count.view(len(batch_target_decision_count), 1)
                batch_loss = -torch.sum(batch_target_count * target_prob)
                return batch_loss
        else:
            predicted_param = F.softmax(pre_output_v, dim=1)
            return predicted_param

    def construct_mask(self, batch_zero_one):
        batch_size = len(batch_zero_one)
        left_compare = torch.ones(batch_size, dtype=torch.long)
        right_compare = torch.zeros(batch_size, dtype=torch.long)
        left_mask = torch.eq(batch_zero_one, left_compare)
        right_mask = torch.eq(batch_zero_one, right_compare)
        left_mask = left_mask.view(batch_size, 1)
        right_mask = right_mask.view(batch_size, 1)
        left_mask = left_mask.expand(-1, self.hid_dim)
        right_mask = right_mask.expand(-1, self.hid_dim)
        return left_mask, right_mask

    # estimate parameters of e-step
    def predict(self, sentence_trans_param, root_param, decision_param, batch_size, trans_counter, root_counnter, decision_counter,
                sentence_map, sentence_word_map, language_map, languages, epoch):
        s_len, _, cvalency = sentence_trans_param[0].shape  # h c v
        input_decision_pos_num, decision_dir_num, dvalency, target_decision_num = decision_param.shape
        # input_trans_list = [[p, cv] for p in range(input_pos_num) for cv in range(cvalency)]
        # batched_input_trans = utils.construct_update_batch_data(input_trans_list, batch_size)
        # trans_batch_num = len(batched_input_trans)

        for s in range(len(sentence_map)):
            # Update transition parameters
            batch_target_lan_v = torch.LongTensor([languages[language_map[s]]]).expand(len(sentence_map[s])**2)  # TODO hanwj
            batch_input_len = torch.LongTensor([len(sentence_map[s])]).expand(len(sentence_map[s])**2)
            batch_input_sen_v = torch.LongTensor([sentence_map[s]]).expand(len(sentence_map[s])**2, len(sentence_map[s]))
            batch_input_sen_v_mlist = torch.LongTensor([sentence_word_map[s]]).expand(len(sentence_word_map[s])**2, len(sentence_word_map[s]))
            batch_input_sen_word_v = torch.LongTensor([sentence_word_map[s]]).expand(len(sentence_map[s])**2, len(sentence_word_map[s]))
            one_batch_input_pos = torch.LongTensor([sentence_map[s][h] for h in range(len(sentence_map[s])) for _ in range(len(sentence_map[s])) for v in range(cvalency)])
            one_batch_input_word = torch.LongTensor([sentence_word_map[s][h] for h in range(len(sentence_word_map[s])) for c in range(len(sentence_word_map[s])) for v in range(cvalency)])
            one_batch_output_pos = torch.LongTensor([sentence_map[s][c] for h in range(len(sentence_map[s])) for c in range(len(sentence_map[s])) for v in range(cvalency)])
            one_batch_output_word = torch.LongTensor([sentence_word_map[s][c] for h in range(len(sentence_word_map[s])) for c in range(len(sentence_word_map[s])) for v in range(cvalency)])
            one_batch_dir = torch.LongTensor([1 if h<c else 0 for h in range(len(sentence_map[s])) for c in range(len(sentence_map[s])) for v in range(cvalency)])
            one_batch_cvalency = torch.LongTensor([v for h in range(len(sentence_map[s])) for c in range(len(sentence_map[s])) for v in range(cvalency)])
            one_batch_input_tag_index = np.array([h for h in range(len(sentence_map[s])) for c in range(len(sentence_map[s])) for v in range(cvalency)])
            one_batch_input_word_index = np.array([h for h in range(len(sentence_word_map[s])) for c in range(len(sentence_word_map[s])) for v in range(cvalency)])
            one_batch_output_tag_index = np.array([c for h in range(len(sentence_map[s])) for c in range(len(sentence_map[s])) for v in range(cvalency)])
            one_batch_output_word_index = np.array([c for h in range(len(sentence_word_map[s])) for c in range(len(sentence_word_map[s])) for v in range(cvalency)])
            one_batch_dir_index = np.array([1 if h<c else 0 for h in range(len(sentence_map[s])) for c in range(len(sentence_map[s])) for v in range(cvalency)])
            one_batch_cvalency_index = np.array([v for h in range(len(sentence_map[s])) for c in range(len(sentence_map[s])) for v in range(cvalency)])
            predicted_trans_param, predicted_trans_param_word = self.forward_(one_batch_input_pos, one_batch_input_word, one_batch_input_word_index, one_batch_dir, one_batch_cvalency,
                                                  None, None, None, True, 'child',
                                                  self.em_type, batch_target_lan_v, batch_input_sen_v_mlist, batch_input_sen_v,
                                                  batch_input_len, epoch=epoch)
            if self.full_lex:
                sentence_trans_param[s][one_batch_input_word_index, one_batch_output_word_index, one_batch_cvalency_index] = predicted_trans_param_word.detach().numpy()[range((len(sentence_word_map[s])**2)*cvalency), one_batch_output_word]  # .reshape(one_batch_size, target_pos_num, 1, 1)
            else:
                sentence_trans_param[s][one_batch_input_tag_index, one_batch_output_tag_index, one_batch_cvalency_index] = predicted_trans_param.detach().numpy()[range((len(sentence_map[s])**2)*cvalency), one_batch_output_pos]  # .reshape(one_batch_size, target_pos_num, 1, 1)


        # _, input_pos_num, target_pos_num, dir_num, cvalency = sentence_trans_param.shape
        # input_decision_pos_num, decision_dir_num, dvalency, target_decision_num = decision_param.shape
        # input_trans_list = [[p, d, cv] for p in range(input_pos_num) for d in range(dir_num) for cv in range(cvalency)]
        # input_decision_list = [[p, d, dv] for p in range(input_decision_pos_num) for d in range(dir_num) for dv in
        #                        range(dvalency)]
        #
        # batched_input_trans = utils.construct_update_batch_data(input_trans_list, batch_size)
        # batched_input_decision = utils.construct_update_batch_data(input_decision_list, batch_size)
        # trans_batch_num = len(batched_input_trans)
        # decision_batch_num = len(batched_input_decision)
        # for s in range(len(sentence_map)):
        #     for i in range(trans_batch_num):
        #         # Update transition parameters
        #         one_batch_size = len(batched_input_trans[i])
        #         batch_target_lan_v = torch.LongTensor([languages[language_map[s]]]).expand(one_batch_size)  # TODO hanwj
        #         batch_input_len = torch.LongTensor([len(sentence_map[s])]).expand(one_batch_size)
        #         batch_input_sen_v = torch.LongTensor([sentence_map[s]]).expand(one_batch_size, len(sentence_map[s]))
        #         one_batch_input_pos = torch.LongTensor(batched_input_trans[i])[:, 0]
        #         one_batch_dir = torch.LongTensor(batched_input_trans[i])[:, 1]
        #         one_batch_cvalency = torch.LongTensor(batched_input_trans[i])[:, 2]
        #         one_batch_input_pos_index = np.array(batched_input_trans[i])[:, 0]
        #         one_batch_dir_index = np.array(batched_input_trans[i])[:, 1]
        #         one_batch_cvalency_index = np.array(batched_input_trans[i])[:, 2]
        #         predicted_trans_param = self.forward_(one_batch_input_pos, one_batch_dir, one_batch_cvalency,
        #                                               None, None, None, True, 'child',
        #                                               self.em_type, batch_target_lan_v, batch_input_sen_v,
        #                                               batch_input_len, epoch=epoch)
        #         sentence_trans_param[s][one_batch_input_pos_index, :, one_batch_dir_index, one_batch_cvalency_index] = predicted_trans_param.detach().numpy()#.reshape(one_batch_size, target_pos_num, 1, 1)
        # TODO:
        # if not child_only:
        #     for i in range(decision_batch_num):
        #         # Update decision parameters
        #         one_batch_size = len(batched_input_decision[i])
        #         if self.unified_network:
        #             one_batch_input_decision_pos = torch.LongTensor(
        #                 map(lambda p: from_decision[p], np.array(batched_input_decision[i])[:, 0]))
        #         else:
        #             one_batch_input_decision_pos = torch.LongTensor(batched_input_decision[i])[:, 0]
        #         one_batch_decision_dir = torch.LongTensor(batched_input_decision[i])[:, 1]
        #         one_batch_dvalency = torch.LongTensor(batched_input_decision[i])[:, 2]
        #         if self.unified_network:
        #             one_batch_input_decision_pos_index = np.array(one_batch_input_decision_pos).tolist()
        #             one_batch_input_decision_pos_index = np.array(
        #                 map(lambda p: to_decision[p], one_batch_input_decision_pos_index))
        #         else:
        #             one_batch_input_decision_pos_index = np.array(batched_input_decision[i])[:, 0]
        #         one_batch_decision_dir_index = np.array(batched_input_decision[i])[:, 1]
        #         one_batch_dvalency_index = np.array(batched_input_decision[i])[:, 2]
        #         if self.unified_network:
        #             predicted_decision_param = self.forward_(one_batch_input_decision_pos, one_batch_decision_dir,
        #                                                      one_batch_dvalency, None, None, True, 'decision',
        #                                                      self.em_type)
        #         else:
        #             predicted_decision_param = self.forward_decision(one_batch_input_decision_pos,
        #                                                              one_batch_decision_dir, one_batch_dvalency,
        #                                                              None, None, True, self.em_type)
        #         decision_param[one_batch_input_decision_pos_index, :, one_batch_decision_dir_index,
        #         one_batch_dvalency_index, :] = predicted_decision_param.detach().numpy().reshape(one_batch_size, 1,
        #                                                                                          target_decision_num)
        # if child_only:
        decision_counter = decision_counter + self.param_smoothing
        decision_sum = np.sum(decision_counter, axis=3, keepdims=True)
        decision_param = decision_counter / decision_sum

        root_counnter = root_counnter + self.param_smoothing
        root_sum = np.sum(root_counnter)
        root_param = root_counnter / root_sum

        trans_counter = trans_counter + self.param_smoothing
        child_sum = np.sum(trans_counter, axis=1, keepdims=True)
        trans_param = trans_counter / child_sum

        # decision_counter = decision_counter + self.param_smoothing
        # decision_sum = np.sum(decision_counter, axis=3, keepdims=True)
        # decision_param_compare = decision_counter / decision_sum
        # decision_difference = decision_param_compare - decision_param
        # if not self.child_only:
        #     print 'distance for decision in this iteration ' + str(LA.norm(decision_difference))
        # trans_counter = trans_counter + self.param_smoothing
        # child_sum = np.sum(trans_counter, axis=(1, 3), keepdims=True)
        # trans_param_compare = trans_counter / child_sum
        # trans_difference = trans_param_compare - trans_param
        # print 'distance for trans in this iteration ' + str(LA.norm(trans_difference))
        return sentence_trans_param, trans_param, root_param, decision_param
