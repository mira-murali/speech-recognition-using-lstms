import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader

# from preprocessing import cifar_10_preprocess
from ctcdecode import CTCBeamDecoder
from warpctc_pytorch import CTCLoss
# from data.phoneme_list import PHONEME_MAP
import Levenshtein as L
from torch.autograd import Variable
# from weight_drop import WeightDrop

from collections import Counter
import numpy as np
import math
import csv
import argparse

import sys
import os
import time

parser = argparse.ArgumentParser(description='LSTM Training for Speech to Text')
parser.add_argument('--epochs', default=10, type=int, help='number of epochs for training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate for optimizer')
parser.add_argument('--print-freq', default=100, type=int, help='How often to print progress within every epoch')
parser.add_argument('--weight-decay', default=1e-6, type=float, help='weight decay for optimizer')
parser.add_argument('--foldername', default='./', type=str, help='folder to save model')
parser.add_argument('--eval', default=0, type=int, help='if code is only being run to get test predictions')
parser.add_argument('--batch-size', default=32, type=int, help='Batch size to load in data')
parser.add_argument('--load-folder', default='./', type=str, help='folder from which to load saved model for finetuning')
parser.add_argument('--finetune', default=0, type=int, help='if code is being used to finetune an already trained model')
parser.add_argument('--path-to-data', default='data', type=str, help='path to folder in which data is stored')
args = parser.parse_args()

PHONEME_MAP = [

    '_',  # "+BREATH+"
    '+',  # "+COUGH+"
    '~',  # "+NOISE+"
    '!',  # "+SMACK+"
    '-',  # "+UH+"
    '@',  # "+UM+"
    'a',  # "AA"
    'A',  # "AE"
    'h',  # "AH"
    'o',  # "AO"
    'w',  # "AW"
    'y',  # "AY"
    'b',  # "B"
    'c',  # "CH"
    'd',  # "D"
    'D',  # "DH"
    'e',  # "EH"
    'r',  # "ER"
    'E',  # "EY"
    'f',  # "F"
    'g',  # "G"
    'H',  # "HH"
    'i',  # "IH"
    'I',  # "IY"
    'j',  # "JH"
    'k',  # "K"
    'l',  # "L"
    'm',  # "M"
    'n',  # "N"
    'G',  # "NG"
    'O',  # "OW"
    'Y',  # "OY"
    'p',  # "P"
    'R',  # "R"
    's',  # "S"
    'S',  # "SH"
    '.',  # "SIL"
    't',  # "T"
    'T',  # "TH"
    'u',  # "UH"
    'U',  # "UW"
    'v',  # "V"
    'W',  # "W"
    '?',  # "Y"
    'z',  # "Z"
    'Z',  # "ZH"
]

dir_path = os.path.realpath('lstm_cnn.py')
os.environ['CURRENT'] = dir_path[:dir_path.find('lstm')]
if not os.path.isdir(os.path.join(os.environ['CURRENT'], args.foldername)):
    os.mkdir(os.path.join(os.environ['CURRENT'], args.foldername))

phoneme_freq = None

if args.eval:
    outfile = open(os.path.join(args.foldername, 'output_eval.txt'), 'w')
else:
    outfile = open(os.path.join(args.foldername, 'output.txt'), 'w')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
outfile.write(str(device))
outfile.write('\n')



def write_to_csv(predictions):
    output_file = open(os.path.join(args.foldername, "submission.csv"), "w")
    pred_writer = csv.writer(output_file, delimiter=',')
    pred_writer.writerow(['id', 'Predicted'])
    for i in range(len(predictions)):
        pred_writer.writerow([str(i), predictions[i]])
#
def getLengths(data):
    seq_lengths = [len(d) for d in data]
    return seq_lengths

class framesDataset(Dataset):
    def __init__(self, frames, labels):
        self.frames = [torch.tensor(frames[i]) for i in range(frames.shape[0])]
        if labels is not None:
            self.labels = [torch.LongTensor(labels[i]+1) for i in range(labels.shape[0])]
        else:
            self.labels = None

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        if self.labels is not None:
            targets = self.labels[idx]
            return frame.to(device), targets.to(device)
        return frame.to(device)

def collate_frames(utterance_list):
    utterances, targets = zip(*utterance_list)
    seq_lengths = [len(utterance) for utterance in utterances]
    seq_order = sorted(range(len(seq_lengths)), key=seq_lengths.__getitem__, reverse=True)
    sorted_utterances = [utterances[i] for i in seq_order]
    sorted_targets = [targets[i] for i in seq_order]
    return sorted_utterances, sorted_targets

class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x

class baselineLSTM(nn.Module):
    def __init__(self, hidden_size = 384, input_size=64, num_classes=47, nlayers=3):
        super(baselineLSTM, self).__init__()
        self.phoneme_size = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.embed =  nn.Sequential(
            nn.Conv1d(40, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ELU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ELU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ELU(),
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ELU()
        )
        # self.rnns = nn.ModuleList()
        # hidden0 = torch.zeros(self.nlayers*2, 1, self.hidden_size)
        # cell0 = torch.zeros(self.nlayers*2, 1, self.hidden_size)
        # self.hidden0 = nn.Parameter(hidden0, requires_grad=True).to(device)
        # self.cell0 = nn.Parameter(cell0, requires_grad=True).to(device)
        self.dropout = LockedDropout()
        self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, bidirectional=True, dropout=0.4, num_layers=self.nlayers)
        # weight_list = []
        # for name in self.rnn.named_parameters():
        #     if 'weight_hh' in name[0]:
        #         weight_list.append(name[0])
        # self.drop_connect = WeightDrop(self.rnn, weight_list, dropout=0.5)

        self.output_layer = nn.Linear(in_features=2*self.hidden_size, out_features=self.phoneme_size)

    def forward(self, utterance_list, hidden=None):
        max_length = len(utterance_list[0])
        batch_size = len(utterance_list)
        padded_utterances = torch.stack([torch.t(F.pad(u, (0, 0, 0, max_length-len(u)))) for u in utterance_list])
        # print(padded_utterances.shape)
        seq_lengths = [len(u) for u in utterance_list]
        embedding = self.embed(padded_utterances)
        n, c, h = embedding.size()
        embedding = embedding.permute(2, 0, 1)
        scaled_length = max_length
        # this is what h should be
        for l in range(0, len(self.embed), 2):
            n = self.embed[l]
            # print(scaled_length, h)
            scaled_length = int((scaled_length + 2*n.padding[0] - n.dilation[0]*(n.kernel_size[0]-1)-1)/n.stride[0] + 1)

        new_lengths = []
        sliced_embeddings = []
        for i in range(batch_size):
            if scaled_length - seq_lengths[i] >= 0:
                sliced_embeddings.append(embedding[:, i, :seq_lengths[i]])
                new_lengths.append(seq_lengths[i])
            else:
                sliced_embeddings.append(embedding[:, i, :])
                new_lengths.append(scaled_length)

        # print(sliced_embeddings[0].shape, sliced_embeddings[17].shape, utterance_list[0].shape, utterance_list[17].shape)

        # hidden = self.hidden0.repeat(1, batch_size, 1)
        # cell = self.cell0.repeat(1, batch_size, 1)
        packed_input = pack_sequence(sliced_embeddings)
        output_packed, hidden = self.rnn(packed_input, hidden)


        output_padded, lengths = pad_packed_sequence(output_packed)

        output_flatten = output_padded.view(-1, output_padded.size(2))
        logits = self.output_layer(output_flatten)
        logits = logits.view(-1, batch_size, logits.size(1))
        # print(torch.tensor(new_lengths)==lengths)
        return logits, torch.tensor(new_lengths)

class ctcCriterion(CTCLoss):
    def forward(self, prediction, targets):
        acts = prediction[0]
        act_lens = prediction[1].int()
        label_lens = prediction[2].int()
        labels = [target.int() for target in targets]
        labels = torch.cat(labels)
        return super(ctcCriterion, self).forward(acts=acts, labels=labels.cpu(), act_lens=act_lens.cpu(),\
        label_lens=label_lens.cpu())

class ER:
    def __init__(self):
        self.label_map = [' ']+PHONEME_MAP
        self.decoder = CTCBeamDecoder(labels=self.label_map, beam_width=50, blank_id=0)

    def __call__(self, prediction, target):
        return self.forward(prediction, target)

    def forward(self, prediction, target):
        logits = prediction[0]
        feature_lengths = prediction[1].int()
        logits = torch.transpose(logits, 0, 1)
        logits = logits.cpu()
        probs = F.softmax(logits, dim=2)
        output, scores, timesteps, out_seq_len = self.decoder.decode(probs=probs, seq_lens=feature_lengths)
        predictions = []
        if target == None:
            for i in range(output.size(0)):
                pred = "".join(self.label_map[o] for o in output[i, 0, :out_seq_len[i, 0]])
                predictions.append(pred)
            return predictions
        pos = 0
        ls = 0.
        for i in range(output.size(0)):
            pred = "".join(self.label_map[o] for o in output[i, 0, :out_seq_len[i, 0]])
            true = "".join(self.label_map[l] for l in target[i])
            # print("Pred: {}, True: {}".format(pred, true))
            # pos = pos + target_lengths[i]
            ls += L.distance(pred, true)
        # assert pos == labels.size(0)
        return ls/output.size(0)


def main():
    print('The code assumes that the data is inside a folder named data.')
    print('If this is not the case, please set the path to the data using --path-to-data in the command line')
    global phoneme_freq
    model = baselineLSTM()
    model.to(device)
    train_data = np.load(os.path.join(args.path_to_data, 'wsj0_train.npy'), encoding='bytes')
    train_labels = np.load(os.path.join(args.path_to_data, 'wsj0_train_merged_labels.npy'), encoding='bytes')
    # train_data = train_data[:1000]
    # train_labels = train_labels[:1000]
    phoneme_freq = phoneme_dist(train_labels)
    model.apply(bias_init)
    #n sys.exit(0)
    dev_data = np.load(os.path.join(args.path_to_data, 'wsj0_dev.npy'), encoding='bytes')
    dev_labels = np.load(os.path.join(args.path_to_data, 'wsj0_dev_merged_labels.npy'), encoding='bytes')
    test_data = np.load(os.path.join(args.path_to_data, 'wsj0_test.npy'), encoding='bytes')

    train_dataset = framesDataset(frames=train_data, labels=train_labels)
    dev_dataset = framesDataset(frames=dev_data, labels=dev_labels)
    test_dataset = framesDataset(frames=test_data, labels=None)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_frames)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_frames)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)



    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
    criterion = ctcCriterion()
    min_error = math.inf
    # model.init_hidden(args.batch_size)
    if not args.eval:
        if args.finetune:
            checkpoint = torch.load(os.path.join(args.load_folder,'model.pth.tar'))
            model.load_state_dict(checkpoint['state_dict'])
        for epoch in range(args.epochs):
            adjust_learning_rate(optimizer, epoch)
            train(model, criterion, optimizer, train_loader, epoch)
            error = validate(model, criterion, dev_loader, epoch)
            if min_error > error:
                min_error = error
                checkpoint = {'state_dict': model.state_dict()}
                torch.save(checkpoint, os.path.join(args.foldername, 'model.pth.tar'))

    checkpoint = torch.load(os.path.join(args.foldername,'model.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    test(model, test_loader)

def train(model, criterion, optimizer, train_loader, epoch):
    outfile.write("Training:\n")
    print("Training: ")
    error_rate_op = ER()
    start = time.time()
    model.train()
    predictions = []
    feature_lens = []
    labels = []
    avg_loss = 0
    avg_error = 0
    hidden = None
    for i, (data, target) in enumerate(train_loader):
        data_time = time.time() - start
        # print("Hidden: ", hidden)
        logits, input_len = model(data)
        target_lengths = torch.tensor(getLengths(target))
        loss = criterion.forward((logits, input_len, target_lengths), target)/args.batch_size
        # loss_time = time.time() - start
        avg_loss += loss.item()
        # print(logits.size(1))
        # predictions.append(logits.cpu())
        # feature_lens.append(input_len.cpu())
        # labels = labels + target
        # input_len = torch.cat(input_len)
        # concat_target = torch.cat(target)
        train_error = error_rate_op((logits, input_len), target)
        # error_time = time.time() - start
        avg_error += train_error
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        batch_time = time.time() - start
        if i%args.print_freq==0:
            # print("Loss time: ", loss_time)
            # print("Error time: ", error_time)
            print("Epoch: [{0}][{1}/{2}]\t"
                  "Loss: {3}\t"
                  "Error: {4}\t"
                  "Data time: {5}\t"
                  "Batch time: {6}".format(epoch, i, len(train_loader), avg_loss/(i+1), avg_error/(i+1), data_time, batch_time))
            outfile.write("Epoch: [{0}][{1}/{2}]\t"
                  "Loss: {3}\t"
                  "Error: {4}\t"
                  "Data time: {5}\t"
                  "Batch time: {6}".format(epoch, i, len(train_loader), avg_loss/(i+1), avg_error/(i+1), data_time, batch_time))
            outfile.write('\n')

    # last_batch = predictions[-1]
    # predictions = torch.cat(predictions[:-1], dim=0)
    # predictions = predictions.view(-1, predictions.size(2))
    # last_batch = last_batch.view(-1, last_batch.size(2))
    # predictions = torch.cat((predictions, last_batch), dim=0)
    # predictions = predictions.view(-1, , predictions.size(1))
    # predictions = torch.cat(predictions[:-1], dim=0)
    # feature_lens = torch.cat(feature_lens[:-1], dim=0)
    #
    # train_error = error_rate_op((predictions, feature_lens), labels[:-1])
    # print("Training Error per Epoch: ", train_error)
    # outfile.write("Training Error per Epoch: "+str(train_error)+"\n")

def validate(model, criterion, val_loader, epoch):
    model.eval()
    outfile.write("Validation:\n")
    print("Validation: ")
    start = time.time()
    avg_loss = 0
    avg_error = 0
    predictions = []
    feature_lens = []
    labels = []
    error_rate_op = ER()
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            data_time = time.time() - start
            logits, input_len = model(data)
            target_lengths = torch.tensor(getLengths(target))
            loss = criterion.forward((logits, input_len, target_lengths), target)/args.batch_size
            # loss_time = time.time() - start
            avg_loss += loss.item()
            # predictions.append(logits.cpu())
            # feature_lens.append(input_len.cpu())
            # labels = labels + target
            # input_len = torch.cat(input_len)
            # concat_target = torch.cat(target)
            val_error = error_rate_op((logits, input_len), target)
            # error_time = time.time() - start
            avg_error += val_error
            batch_time = time.time() - start
            if i%args.print_freq==0:
                # print("Loss time: ", loss_time)
                # print("Error time: ", error_time)
                print("Epoch:[{0}][{1}/{2}]\t"
                      "Loss: {3}\t"
                      "Error: {4}\t"
                      "Data time: {5}\t"
                      "Batch time: {6}".format(epoch, i, len(val_loader), avg_loss/(i+1), avg_error/(i+1), data_time, batch_time))
                outfile.write("Epoch:[{0}][{1}/{2}]\t"
                      "Loss: {3}\t"
                      "Error: {4}\t"
                      "Data time: {5}\t"
                      "Batch time: {6}".format(epoch, i, len(val_loader), avg_loss/(i+1), avg_error/(i+1), data_time, batch_time))
                outfile.write('\n')
                avg_loss = 0

    # predictions = torch.cat(predictions[:-1], dim=0)
    # # print(labels)
    # feature_lens = torch.cat(feature_lens[:-1], dim=0)
    #
    # val_error = error_rate_op((predictions, feature_lens), labels[:-1])
    # print("Validation Error per Epoch: ", val_error)
    # outfile.write("Validation Error per Epoch: "+str(val_error)+"\n")
    return avg_error/len(val_loader)


def test(model, test_loader):
    test_predictions = []
    error_rate_op = ER()
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            print("Testing: [{0}]/[{1}]".format(i+1, len(test_loader)))
            logits, input_len = model(data)
            test_prediction = error_rate_op((logits, input_len), None)
            test_predictions = test_predictions + test_prediction

    write_to_csv(test_predictions)

def adjust_learning_rate(optimizer, epoch):
    lr = args.lr
    if (epoch+1) >= args.epochs//2:
        lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def bias_init(layer):
    global phoneme_freq
    if type(layer)==nn.Linear:
        layer.bias.data[0] = 0
        for i in range(1, layer.bias.data.shape[0]):
            layer.bias.data[i].fill_(phoneme_freq[i])
            # print(layer.bias.data)

def phoneme_dist(labels):
    freq = Counter()
    total = sum([len(label) for label in labels])
    for label in labels:
        for l in label:
                freq[l+1] += 1

    for key, value in freq.items():
        freq[key] = np.log(value/total)

    return freq



if __name__=='__main__':
    main()
