import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ECGDataset
from utils import split_data
from resnet import resnet34, resnet18
from lstm import LSTMModel
import warnings
from tqdm import tqdm
from utils import cal_f1s, cal_aucs 
import numpy as np

import torch
from tqdm import tqdm

def train(dataloader, net, args, criterion, epoch, scheduler, optimizer, device):
    print(f'Training epoch {epoch}:')
    net.train()
    running_loss = 0
    correct_preds_per_label = None
    total_preds_per_label = None
    accuracy_per_label = []
    precision_per_label = []
    recall_per_label = []
    f1_per_label = []

    for _, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Convert outputs to binary predictions
        predicted = torch.sigmoid(outputs).data > 0.5

        # Update correct and total predictions for each label
        if correct_preds_per_label is None:
            correct_preds_per_label = (predicted == labels).sum(axis=0)
            total_preds_per_label = labels.size(0)
        else:
            correct_preds_per_label += (predicted == labels).sum(axis=0)
            total_preds_per_label += labels.size(0)

    avg_loss = running_loss / len(dataloader)
    
    # Calculate accuracy, precision, recall, and F1 score for each label
    for i in range(labels.size(1)):
        accuracy = correct_preds_per_label[i].item() / total_preds_per_label
        precision = correct_preds_per_label[i].item() / (correct_preds_per_label[i].item() + (predicted[:, i] == 0).sum().item())
        recall = correct_preds_per_label[i].item() / (correct_preds_per_label[i].item() + (labels[:, i] == 0).sum().item())
        f1 = 2 * precision * recall / (precision + recall)
        accuracy_per_label.append(accuracy)
        precision_per_label.append(precision)
        recall_per_label.append(recall)
        f1_per_label.append(f1)

    # Calculate the averages
    avg_accuracy = sum(accuracy_per_label) / len(accuracy_per_label)
    avg_precision = sum(precision_per_label) / len(precision_per_label)
    avg_recall = sum(recall_per_label) / len(recall_per_label)
    avg_f1 = sum(f1_per_label) / len(f1_per_label)

    print(f'Loss: {avg_loss:.4f}')
    print('Accuracy per label:')
    for i in range(labels.size(1)):
        print(f'Label {i}: {accuracy_per_label[i]:.4f}')
    print('Precision per label:')
    for i in range(labels.size(1)):
        print(f'Label {i}: {precision_per_label[i]:.4f}')
    print('Recall per label:')
    for i in range(labels.size(1)):
        print(f'Label {i}: {recall_per_label[i]:.4f}')
    print('F1 score per label:')
    for i in range(labels.size(1)):
        print(f'Label {i}: {f1_per_label[i]:.4f}')

    # Print average metrics
    print(f'Average Accuracy: {avg_accuracy:.4f}')
    print(f'Average Precision: {avg_precision:.4f}')
    print(f'Average Recall: {avg_recall:.4f}')
    print(f'Average F1 Score: {avg_f1:.4f}')

    return accuracy_per_label, precision_per_label, recall_per_label, f1_per_label, avg_accuracy, avg_precision, avg_recall, avg_f1



def evaluate(dataloader, net, args, criterion, device):
    print('Validating...')
    net.eval()
    running_loss = 0
    output_list, labels_list = [], []
    correct_preds_per_label = None
    total_preds_per_label = None
    accuracy_per_label = []
    precision_per_label = []
    recall_per_label = []
    f1_per_label = []

    with torch.no_grad():
        for _, (data, labels) in enumerate(tqdm(dataloader)):
            data, labels = data.to(device), labels.to(device)
            output = net(data)
            loss = criterion(output, labels)
            running_loss += loss.item()
            output = torch.sigmoid(output)
            predicted = output.data > 0.5

            output_list.append(output.data.cpu().numpy())
            labels_list.append(labels.data.cpu().numpy())

            # Update correct and total predictions for each label
            if correct_preds_per_label is None:
                correct_preds_per_label = (predicted == labels).sum(axis=0)
                total_preds_per_label = labels.size(0)
            else:
                correct_preds_per_label += (predicted == labels).sum(axis=0)
                total_preds_per_label += labels.size(0)

        avg_loss = running_loss / len(dataloader)

        # Calculate accuracy, precision, recall, and F1 score for each label
        for i in range(labels.size(1)):
            accuracy = correct_preds_per_label[i].item() / total_preds_per_label
            precision = correct_preds_per_label[i].item() / (correct_preds_per_label[i].item() + (predicted[:, i] == 0).sum().item())
            recall = correct_preds_per_label[i].item() / (correct_preds_per_label[i].item() + (labels[:, i] == 0).sum().item())
            f1 = 2 * precision * recall / (precision + recall)
            accuracy_per_label.append(accuracy)
            precision_per_label.append(precision)
            recall_per_label.append(recall)
            f1_per_label.append(f1)

        avg_accuracy = sum(accuracy_per_label) / len(accuracy_per_label)
        avg_precision = sum(precision_per_label) / len(precision_per_label)
        avg_recall = sum(recall_per_label) / len(recall_per_label)
        avg_f1 = sum(f1_per_label) / len(f1_per_label)

        print(f'Loss: {avg_loss:.4f}')
        print('Accuracy per label:')
        for i in range(labels.size(1)):
            print(f'Label {i}: {accuracy_per_label[i]:.4f}')
        print('Precision per label:')
        for i in range(labels.size(1)):
            print(f'Label {i}: {precision_per_label[i]:.4f}')
        print('Recall per label:')
        for i in range(labels.size(1)):
            print(f'Label {i}: {recall_per_label[i]:.4f}')
        print('F1 score per label:')
        for i in range(labels.size(1)):
            print(f'Label {i}: {f1_per_label[i]:.4f}')

        print(f'Average Accuracy: {avg_accuracy:.4f}')
        print(f'Average Precision: {avg_precision:.4f}')
        print(f'Average Recall: {avg_recall:.4f}')
        print(f'Average F1 Score: {avg_f1:.4f}')

        y_trues = np.vstack(labels_list)
        y_scores = np.vstack(output_list)
        f1s = cal_f1s(y_trues, y_scores)
        avg_f1_cal = np.mean(f1s)
        print('F1s:', f1s)
        print('Avg F1 (calculated): %.4f' % avg_f1_cal)

        if args.phase == 'train' and avg_f1 > args.best_metric:
            args.best_metric = avg_f1
            torch.save(net.state_dict(), args.model_path)
        else:
            aucs = cal_aucs(y_trues, y_scores)
            avg_auc = np.mean(aucs)
            print('AUCs:', aucs)
            print('Avg AUC: %.4f' % avg_auc)

    return accuracy_per_label, precision_per_label, recall_per_label, f1_per_label, avg_accuracy, avg_precision, avg_recall, avg_f1



warnings.filterwarnings('ignore', category=FutureWarning)

class Args:
    data_dir = '/home/asus/Documents/cpsc_2018/datas'
    leads = 'all'
    seed = 42
    num_classes = 9  # Set this to your number of classes
    lr = 0.0001
    batch_size = 32
    num_workers = 4
    phase = 'train'
    epochs = 10
    resume = False
    use_gpu = True  # Set to True if you want to use GPU and it's available
    model_path = ''  # Set a default path or leave it as an empty string

args = Args()



if __name__ == "__main__":
    data_dir = os.path.normpath(args.data_dir)
    database = os.path.basename(data_dir)
    args.best_metric = 0

    # Set the model path if it's not already set
    if not args.model_path:
        args.model_path = f'models/lstm_{database}_{args.leads}_{args.seed}.pth'
    
    # Ensure the 'models' directory exists
    model_dir = os.path.dirname(args.model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    device = torch.device('cuda:0') if args.use_gpu and torch.cuda.is_available() else torch.device('cpu')
    
    leads = args.leads.split(',') if args.leads != 'all' else 'all'
    nleads = len(leads) if args.leads != 'all' else 12
    
    label_csv = os.path.join(data_dir, 'labels.csv')


    # Initialize LSTM model
    input_size = 15000  # Number of ECG leads
    hidden_size = 128  # Adjust as needed
    num_layers = 2  # Adjust as needed
    net = LSTMModel(input_size, hidden_size, num_layers, args.num_classes)
    net = net.to(device)
    
    # Assuming split_data, ECGDataset, and resnet34 are defined elsewhere
    train_folds, val_folds, test_folds = split_data(seed=args.seed)
    train_dataset = ECGDataset('train', data_dir, label_csv, train_folds, leads)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_dataset = ECGDataset('val', data_dir, label_csv, val_folds, leads)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_dataset = ECGDataset('test', data_dir, label_csv, test_folds, leads)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    # net = resnet18(input_channels=nleads).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)
    
    criterion = nn.BCEWithLogitsLoss()  # or another appropriate loss function
    
    if args.phase == 'train':
        if args.resume:
            net.load_state_dict(torch.load(args.model_path, map_location=device))
        for epoch in range(args.epochs):
            train(train_loader, net, args, criterion, epoch, scheduler, optimizer, device)
            evaluate(val_loader, net, args, criterion, device)
    else:
        net.load_state_dict(torch.load(args.model_path, map_location=device))
        evaluate(test_loader, net, args, criterion, device)