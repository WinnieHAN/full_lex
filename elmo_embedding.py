from allennlp.modules.elmo import Elmo, batch_to_ids

options_file = '/home/hanwj/Code/l_dmv/data/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json'#"https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = '/home/hanwj/Code/l_dmv/data/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'#"https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, 2, dropout=0)

# use batch_to_ids to convert sentences to character ids
sentences = [['First', 'sentence', '.'], ['Another', '.']]
character_ids = batch_to_ids(sentences)

embeddings = elmo(character_ids)
print(embeddings)
print(embeddings['elmo_representations'].shape())
# embeddings['elmo_representations'] is length two list of tensors.
# Each element contains one layer of ELMo representations with shape
# (2, 3, 1024).
#   2    - the batch size
#   3    - the sequence length of the batch
#   1024 - the length of each ELMo vector