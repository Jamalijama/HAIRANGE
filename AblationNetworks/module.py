import random

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_curve, auc, \
    precision_recall_curve, average_precision_score
from torch.nn import functional as F

from transformer import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    #    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


#    torch.use_deterministic_algorithms(True)


def PreTrainnig(fixed_value, freq_path_in, freq_path_out):
    all_inputs = np.load(freq_path_in, allow_pickle=True)
    seq_num = len(all_inputs)
    all_inputs = torch.FloatTensor(all_inputs)
    print('before trm:', all_inputs.shape)

    all_outputs = []
    model = Transformer(fixed_value)
    model = model.to(device)
    interval = 50
    trm_batch_size = 50

    for i in range(0, seq_num, interval):
        all_input = all_inputs[i:i + interval].clone()
        loader = Data.DataLoader(MyDataSet_freq(all_input), trm_batch_size, False)
        for input in loader:
            input = input.to(device)
            output = model(input).to(device).detach().cpu()
            # output_ = output.clone()
            all_outputs.extend(output.numpy())

    array_input = np.array(all_outputs)
    np.save(freq_path_out, array_input, allow_pickle=True)
    inputs = torch.FloatTensor(array_input)
    print('after trm:', inputs.shape)


def PrintTrain(x, train_loss_lst, train_acc_lst, file_name):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(x, train_loss_lst, 'r')
    ax[0].set_title('train loss')
    #    ax[0].set_xlabel('epochs')
    #    ax[0].set_ylabel('train loss')

    ax[1].plot(x, train_acc_lst, 'g')
    ax[1].set_title('train accuracy')
    #    ax[1].set_xlabel('epochs')
    #    ax[1].set_ylabel('train accuracy')
    ax[1].set_ylim((0, 1.2))

    plt.savefig(file_name, dpi=300)


def PrintValid(x, valid_acc_lst, valid_f1_lst, file_name):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(x, valid_acc_lst, 'b')
    ax[0].set_title('valid accuracy')
    #    ax[0].set_xlabel('epochs')
    #    ax[0].set_ylabel('valid accuracy')
    ax[0].set_ylim((0, 1.2))

    ax[1].plot(x, valid_f1_lst, 'g')
    ax[1].set_title('valid f1')
    #    ax[1].set_xlabel('epochs')
    #    ax[1].set_ylabel('valid f1')
    ax[1].set_ylim((0, 1.2))

    plt.savefig(file_name, dpi=300)


def PrintROCPRAndCM(true, pred, probs, pos_label, cm_file, roc_file, pr_file, roc_csv, pr_csv):
    #    sorted_indices = sorted(range(len(probs)), key=lambda k: probs[k], reverse=True)
    #    sorted_true = [true[i] for i in sorted_indices]
    #    sorted_probs = [probs[i] for i in sorted_indices]

    plt.figure()
    cm = confusion_matrix(true, pred)
    cm_prob = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #    sns.set_context("paper", font_scale=2)
    sns.set(font_scale=3)
    sns.heatmap(cm_prob, annot=True, fmt=".4f", cmap="Blues")
    #    plt.xlabel("Predicted")
    #    plt.ylabel("Actual")
    plt.savefig(cm_file, dpi=300)
    plt.close()

    plt.figure()
    sns.set(font_scale=2)
    fpr, tpr, thresholds = roc_curve(true, probs, pos_label=pos_label)
    # auc = roc_auc_score(true, probs)
    roc_auc = auc(fpr, tpr)
    df = pd.DataFrame()
    df['fpr'] = fpr
    df['tpr'] = tpr
    df.to_csv(roc_csv, index=False)
    plt.plot(fpr, tpr, label="ROC Curve (AUC = %0.4f)" % roc_auc)
    plt.plot([0, 1], [0, 1], "k--")
    #    plt.xlabel("False Positive Rate")
    #    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.savefig(roc_file, dpi=300)
    plt.close()

    plt.figure()
    precision, recall, thresholds = precision_recall_curve(true, probs, pos_label=pos_label)
    auc_score = average_precision_score(true, probs)
    #    precision = precision[:-1]
    #    recall = recall[:-1]
    df = pd.DataFrame()
    df['precision'] = precision
    df['recall'] = recall
    df.to_csv(pr_csv, index=False)
    plt.plot(recall, precision, color='blue', label='PR curve(AUC = %0.4f)' % auc_score)
    plt.plot([0, 1], [1, 0], "k--")
    #    plt.xlabel('Recall')
    #    plt.ylabel('Precision')
    plt.title('P-R Curve')
    #    plt.xlim([0.0, 1.0])
    #    plt.ylim([0.0, 1.0])
    plt.legend(loc="lower right")
    plt.savefig(pr_file, dpi=300)
    plt.close()


def Positive_probs(pos_label, pred, probs):
    positive_probs = []
    for i in range(len(pred)):
        if pred[i] == pos_label:
            positive_probs.append(probs[i][0])
        else:
            positive_probs.append(1 - probs[i][0])
    return positive_probs


# Calculate accuracy, loss, and gradient descent
def train(model, loader, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for input, label in loader:
        # input = torch.LongTensor(input)
        input = input.to(device)
        optimizer.zero_grad()
        predictions = model(input).to(device)
        #        print(predictions.cpu())
        #        print(label)
        loss = criterion(predictions.cpu(), label)
        acc = accuracy_score(predictions.argmax(1).cpu().detach().numpy(), label.cpu().detach().numpy())
        loss.backward(retain_graph=True)
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc
    return epoch_loss / len(loader), epoch_acc / len(loader)


# Calculate accuracy only
def evaluate(model, loader):
    model.eval()
    predictions = []
    true_labels = []
    probs = []
    with torch.no_grad():
        for input, label in loader:
            input = input.to(device)
            preds = model(input).to(device)
            prob = F.softmax(preds, dim=1)
            top_p, top_class = prob.topk(1, dim=1)
            probs.extend(top_p.cpu().detach().numpy().tolist())
            #            predictions.extend(top_class)
            predictions.extend(F.softmax(preds, dim=1).argmax(1).cpu().detach().numpy().tolist())
            true_labels.extend(label.cpu().detach().numpy().tolist())
    #            print('preds', preds)
    #            print('prob', prob)
    #            print('predictions', predictions[0])
    valid_acc = accuracy_score(true_labels, predictions)
    valid_f1 = f1_score(true_labels, predictions, average='macro')
    #    prec = precision_score(true_labels, predictions, average='macro')
    rec = recall_score(true_labels, predictions, average='macro')
    return predictions, true_labels, probs, valid_acc, valid_f1


def test(model, loader):
    model.eval()
    predictions = []
    true_labels = []
    ids = []
    #    probabilities = []
    probs = []
    with torch.no_grad():
        for input, label, id in loader:
            input = input.to(device)
            preds = model(input).to(device)
            prob = F.softmax(preds, dim=1)
            # probs = []
            # for i in prob:
            #     max_value, _ = torch.max(i, dim=0)
            #     probs.append(max_value.item())
            top_p, top_class = prob.topk(1, dim=1)
            probs.extend(top_p)
            predictions.extend(top_class)
            # predictions.extend(F.softmax(preds, dim=1).argmax(1).cpu().detach().numpy().tolist())
            #            probabilities.extend(probs)
            true_labels.extend(label.cpu().detach().numpy().tolist())
            ids.extend(list(id))

    return ids, predictions, true_labels, probs
