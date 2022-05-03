import torch

from torch.utils import data
from transformers import AutoTokenizer

# map lm name to huggingface's pre-trained model names
lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased'}

def get_tokenizer(lm):
    if lm in lm_mp:
        return AutoTokenizer.from_pretrained(lm_mp[lm])
    else:
        return AutoTokenizer.from_pretrained(lm)


class DittoDataset(data.Dataset):
    """EM dataset"""

    def __init__(self,
                 path,
                 max_len=256,
                 size=None,
                 lm='roberta'):
        self.tokenizer = get_tokenizer(lm)
        self.pairs = []
        self.labels = []
        self.ids = []
        self.max_len = max_len
        self.size = size

        if isinstance(path, list):
            lines = path
        else:
            lines = open(path)

        for line in lines:
            s1, s2, label, id1, id2 = line.strip().split('\t')
            self.pairs.append((s1, s2))
            self.labels.append(int(label))
            self.ids.append((int(id1), int(id2)))

        self.pairs = self.pairs[:size]
        self.labels = self.labels[:size]
        self.ids = self.ids[:size]
        print(self.ids)


    def __len__(self):
        """Return the size of the dataset."""
        return len(self.pairs)

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            List of int: token ID's of the two entities
            int: the label of the pair (0: unmatch, 1: match)
        """
        left = self.pairs[idx][0]
        right = self.pairs[idx][1]

        # left + right
        x = self.tokenizer.encode(text=left,
                                  text_pair=right,
                                  max_length=self.max_len,
                                  truncation=True)


        return x, self.labels[idx], self.ids[idx][0], self.ids[idx][1]


    @staticmethod
    def pad(batch):
        """Merge a list of dataset items into a train/test batch
        Args:
            batch (list of tuple): a list of dataset items

        Returns:
            LongTensor: x1 of shape (batch_size, seq_len)
            LongTensor: x2 of shape (batch_size, seq_len).
                        Elements of x1 and x2 are padded to the same length
            LongTensor: a batch of labels, (batch_size,)
        """
        if len(batch[0]) == 5:
            x1, x2, y, id1, id2 = zip(*batch)

            maxlen = max([len(x) for x in x1+x2])
            x1 = [xi + [0]*(maxlen - len(xi)) for xi in x1]
            x2 = [xi + [0]*(maxlen - len(xi)) for xi in x2]
            return torch.LongTensor(x1), \
                   torch.LongTensor(x2), \
                   torch.LongTensor(y), \
                   torch.LongTensor(id1), \
                   torch.LongTensor(id2)
        else:
            x12, y, id1, id2 = zip(*batch)
            maxlen = max([len(x) for x in x12])
            x12 = [xi + [0]*(maxlen - len(xi)) for xi in x12]
            return torch.LongTensor(x12), \
                   torch.LongTensor(y), \
                   torch.LongTensor(id1), \
                   torch.LongTensor(id2)
