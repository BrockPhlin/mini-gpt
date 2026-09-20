import torch
import matplotlib.pyplot as plt

words = open("names.txt").read().splitlines()

chars = sorted(list(set(''.join(words))))
N = torch.zeros((28, 28), dtype=torch.int32)
stoi = {s:i for i, s in enumerate(chars)}
stoi['<S>'] = 26
stoi['<E>'] = 27

# b = {}
for w in words:
    # <S>和<E>是首尾标记
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1
        # bigram = (ch1, ch2)
        # N[stoi[ch1], stoi[ch2]] += 1

# print(sorted(b.items(), key = lambda kv: -kv[1]))

# plt.imshow(N) 
# for jupyter notebook