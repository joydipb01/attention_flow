import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer

import numpy as np

import seaborn as sns
import itertools
import matplotlib as mpl

from matplotlib import pyplot as plt

rc = {'font.size': 10, 'axes.labelsize': 10, 'legend.fontsize': 10.0,
      'axes.titlesize': 32, 'xtick.labelsize': 20, 'ytick.labelsize': 16}

plt.rcParams.update(**rc)

mpl.rcParams['axes.linewidth'] = .5  # set the value globally

from attention_graph_util import *

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def plot_attention_heatmap(att, s_position, t_positions, sentence):
    cls_att = np.flip(att[:,s_position, t_positions], axis=0)
    input_tokens= list(itertools.compress(['<cls>']+sentence.split(),
                                          [i in t_positions for i in np.arange(len(sentence)+1)]))
    xticklb = input_tokens
    yticklb = [str(i) if not i % 2 else '' for i in np.arange(att.shape[0], 0, -1)]
    ax = sns.heatmap(cls_att, xticklabels=xticklb, yticklabels=yticklb, cmap="YlOrRd")
    return ax


def convert_adjmat_tomats(adjmat, n_layers, l):
    mats = np.zeros((n_layers, l, l))

    for i in np.arange(n_layers):
        mats[i] = adjmat[(i+1)*l:(i+2)*l,i*l:(i+1)*l]

    return mats


def get_model_tokenizer():
    pretrained_weights = 'bert-base-uncased'
    model = AutoModelForMaskedLM.from_pretrained(pretrained_weights,
                                                 output_hidden_states=True,
                                                 output_attentions=True)
    model.zero_grad()
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(pretrained_weights, use_fast=True)
    return model, tokenizer


# TODO: 1. See if first sentence is missing a MASK token
#       2. What is src?
def get_data(tokenizer):
    sentences = {}
    src = {}
    targets = {}

    mask = tokenizer.mask_token
    sentences = ["He talked to her about his book",
                 f"She asked the doctor about {mask} backache",
                 f"The author talked to Sara about {mask} book",
                 f"John tried to convince Mary of his love and brought flowers for {mask}",
                 f"Mary convinced John of {mask} love"]
    src = [6, 6, 7, 13, 5]
    targets = [(1, 4), (1, 4), (2, 5), (1, 5), (1, 3)]

    # sentences[1] = "She asked the doctor about "+tokenizer.mask_token+" backache"
    # src[1] = 6
    # targets[1] = (1,4)
    # sentences[0] = "He talked to her about his book"
    # src[0] = 6
    # targets[0] = (1,4)
    # sentences[2] = "The author talked to Sara about "+tokenizer.mask_token+" book"
    # src[2] = 7
    # targets[2] = (2,5)
    # sentences[3] = "John tried to convince Mary of his love and brought flowers for "+tokenizer.mask_token
    # src[3] = 13
    # targets[3] = (1,5)
    # sentences[4] = "Mary convinced John of "+tokenizer.mask_token+" love"
    # src[4] = 5
    # targets[4] = (1,3)
    return sentences, src, targets


def get_attentions(model, sentences, tokenizer, ex_id=2):
    sentence = sentences[ex_id]
    tokens = ['[cls]']+tokenizer.tokenize(sentence)+['[sep]']
    print(len(tokens), tokens)
    tf_input_ids = tokenizer.encode(sentence)
    print(tokenizer.decode(tf_input_ids))
    input_ids = torch.tensor([tf_input_ids])
    model_outputs = model(input_ids)
    all_hidden_states, all_attentions = model_outputs['hidden_states'], model_outputs['attentions']
    _attentions = [att.detach().numpy() for att in all_attentions]
    attentions_mat = np.asarray(_attentions)[:,0]
    print(attentions_mat.shape)
    return attentions_mat, tokens, tf_input_ids, input_ids


def get_targets(model, tokenizer, src, targets, tf_input_ids, input_ids, ex_id=2):
    output = model(input_ids)[0]
    predicted_target = torch.nn.Softmax()(output[0,src[ex_id]])
    print(np.argmax(output.detach().numpy()[0], axis=-1))
    print(tokenizer.decode(np.argmax(output.detach().numpy()[0], axis=-1)))
    print(tf_input_ids[src[ex_id]], tokenizer.decode([tf_input_ids[src[ex_id]]]))
    print(tf_input_ids[targets[ex_id][0]], tokenizer.decode([tf_input_ids[targets[ex_id][0]]]),
          predicted_target[tf_input_ids[targets[ex_id][0]]])
    print(tf_input_ids[targets[ex_id][1]], tokenizer.decode([tf_input_ids[targets[ex_id][1]]]),
          predicted_target[tf_input_ids[targets[ex_id][1]]])
    his_id = tokenizer.encode('his')[1]
    her_id = tokenizer.encode('her')[1]
    print(his_id, her_id)
    print("his prob:", predicted_target[his_id], "her prob:", predicted_target[her_id], "her?",
          predicted_target[her_id] > predicted_target[his_id])
    return predicted_target, his_id, her_id


def plot_predicted_target(predicted_target, his_id, her_id, ex_id=2):
    fig = plt.figure(1, figsize=(2, 6))
    ax = sns.barplot(['his', 'her'], [float(predicted_target[his_id].detach()), float(
        predicted_target[her_id].detach())], linewidth=0, palette='Set1')
    sns.despine(fig=fig, ax=None, top=True, right=True,
                left=True, bottom=False, offset=None, trim=False)
    ax.set_yticks([])
    plt.savefig('rat_bert_bar_{}.png'.format(ex_id), format='png',
                transparent=True, dpi=360, bbox_inches='tight')


mod, tok=get_model_tokenizer()
sentences, src, targets=get_data(tok)
ex_id=2
sentence=sentences[ex_id]
attentions_mat, tokens, tf_input_ids, input_ids=get_attentions(mod, sentences, tok, ex_id)
#print(sentence)

plt.figure(1,figsize=(3,6))
plot_attention_heatmap(attentions_mat.sum(axis=1)/attentions_mat.shape[1], src[ex_id], t_positions=targets[ex_id], sentence=sentence)
plt.savefig('rat_bert_att_{}.png'.format(ex_id), format='png', transparent=True, dpi=360, bbox_inches='tight')


# In[57]:


res_att_mat = attentions_mat.sum(axis=1)/attentions_mat.shape[1]
res_att_mat = res_att_mat + np.eye(res_att_mat.shape[1])[None,...]
res_att_mat = res_att_mat / res_att_mat.sum(axis=-1)[...,None]

res_adj_mat, res_labels_to_index = get_adjmat(mat=res_att_mat, input_tokens=tokens)

res_G = draw_attention_graph(res_adj_mat,res_labels_to_index, n_layers=res_att_mat.shape[0], length=res_att_mat.shape[-1])


# In[58]:


output_nodes = []
input_nodes = []
for key in res_labels_to_index:
    if 'L24' in key:
        output_nodes.append(key)
    if res_labels_to_index[key] < attentions_mat.shape[-1]:
        input_nodes.append(key)

flow_values = compute_flows(res_G, res_labels_to_index, input_nodes, length=attentions_mat.shape[-1])
#flow_G = draw_attention_graph(flow_values,res_labels_to_index, n_layers=attentions_mat.shape[0], length=attentions_mat.shape[-1])


# In[59]:


flow_att_mat = convert_adjmat_tomats(flow_values, n_layers=attentions_mat.shape[0], l=attentions_mat.shape[-1])

plt.figure(1,figsize=(3,6))
plot_attention_heatmap(flow_att_mat, src[ex_id], t_positions=targets[ex_id], sentence=sentence)
plt.savefig('res_fat_bert_att_{}.png'.format(ex_id), format='png', transparent=True,dpi=360, bbox_inches='tight')


# In[60]:


joint_attentions = compute_joint_attention(res_att_mat, add_residual=False)
joint_att_adjmat, joint_labels_to_index = get_adjmat(mat=joint_attentions, input_tokens=tokens)

#G = draw_attention_graph(joint_att_adjmat,joint_labels_to_index, n_layers=joint_attentions.shape[0], length=joint_attentions.shape[-1])


# In[61]:


plt.figure(1,figsize=(3,6))
plot_attention_heatmap(joint_attentions, src[ex_id], t_positions=targets[ex_id], sentence=sentence)
plt.savefig('res_jat_bert_att_{}.png'.format(ex_id), format='png', transparent=True, dpi=360, bbox_inches='tight')

