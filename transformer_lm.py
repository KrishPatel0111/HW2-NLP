import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import TensorDataset, DataLoader


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)


class NeuralLanguageModel(LanguageModel):
    def __init__(self, model, vocab_index, seq_len, device):
        #initialize with trained model and vocab index
        self.model = model
        self.vocab_index = vocab_index
        self.seq_len = seq_len
        self.device = device
        self.space = vocab_index.index_of(' ') if ' ' in vocab_index.objs_to_ints else 0
        

    def get_next_char_log_probs(self, context):
        # Get log probabilities from the model
        self.model.eval()
        indexes = [self.vocab_index.index_of(ch) for ch in context][-self.seq_len+1:]
        indexes = [self.space] + indexes
        x = torch.tensor([indexes], device=self.device)
        log_probs = self.model(x)[:, -1, :]
        return log_probs.squeeze(0).detach().cpu().numpy().astype(float)
        

class CharacterTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, seq_len=128):
        #layers and model definition
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # Forward pass
        T = x.shape[1]
        x = self.emb(x) + self.pos(torch.arange(T, device=x.device)).unsqueeze(0)
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        out = self.encoder(x, mask=mask)
        return self.log_softmax(self.head(out))


def train_lm(args, train_text, dev_text, vocab_index):
    V = len(vocab_index)
    seq_len = 128
    batch_size = 256 if torch.cuda.is_available() else 128
    epochs = 10
    stride = 32
    lr = 0.01

    #dataset with stride
    indexes = torch.tensor([vocab_index.index_of(ch) for ch in train_text])
    starts = torch.arange(0, indexes.numel() - seq_len, step=stride)
    space = vocab_index.index_of(' ') if ' ' in vocab_index.objs_to_ints else 0

    # creates batches of (X, Y) where X is input sequence, Y is target sequence (next char)
    body = torch.stack([indexes[s : s + seq_len - 1] for s in starts])
    sos_col = torch.full((body.size(0), 1), space)
    X = torch.cat([sos_col, body], dim=1)
    Y = torch.empty_like(X)
    Y[:, :-1] = X[:, 1:]
    Y[:, -1]  = Y[:, -2]

    #DataLoader for batching
    ds = TensorDataset(X, Y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=False)

    #initialize model, optimizer, loss
    model = CharacterTransformerLM(V, seq_len=seq_len).to('cuda' if torch.cuda.is_available() else 'cpu')
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.NLLLoss()

    #training loop
    model.train()
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu',enabled=True)
    for ep in range(epochs):
        total_loss = 0.0
        total_tok  = 0

        for xb_cpu, yb_cpu in dl:
            xb = xb_cpu.to('cuda' if torch.cuda.is_available() else 'cpu', non_blocking=True) 
            yb = yb_cpu.to('cuda' if torch.cuda.is_available() else 'cpu', non_blocking=True) 

            with torch.amp.autocast('cuda',enabled=True):
                #Forward pass and loss computation
                log_probs = model(xb)                  
                B, T, V = log_probs.shape
                loss = loss_fn(log_probs.view(B*T, V), yb.view(B*T))

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total_loss += loss.item() * (B * T)
            total_tok  += (B * T)

        avg_nll = total_loss / max(1, total_tok)
        print(f"epoch {ep}: train_nll={avg_nll:.4f}, approx_ppl={math.exp(avg_nll):.2f}")

    model.eval()
    return NeuralLanguageModel(model, vocab_index, seq_len=seq_len, device='cuda' if torch.cuda.is_available() else 'cpu')
