import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import train
from MyDataset import TextDataset, get_word2index


def calculate_metrics(pred_labels, true_labels, num_classes):
    confusion_matrix = torch.zeros(num_classes, num_classes)
    for pred, true in zip(pred_labels, true_labels):
        confusion_matrix[pred, true] += 1

    precision = torch.zeros(num_classes)
    recall = torch.zeros(num_classes)
    f1_score = torch.zeros(num_classes)

    for c in range(num_classes):
        true_positives = confusion_matrix[c, c]
        false_positives = confusion_matrix[:, c].sum() - true_positives
        false_negatives = confusion_matrix[c, :].sum() - true_positives

        precision[c] = true_positives / (true_positives + false_positives)
        recall[c] = true_positives / (true_positives + false_negatives)
        f1_score[c] = 2 * (precision[c] * recall[c]) / (precision[c] + recall[c])

    return precision.mean().item(), recall.mean().item(), f1_score.mean().item()


def test(labeled_loader, num_classes):
    # write_file = open('21307077_lingguoming_agnews_classification', mode='w', encoding='ascii')
    read_file = open(r'processed_data/dataset.csv', mode='r', encoding='ascii')
    right_ = 0
    total_ = 0
    true_labels = []
    pred_labels = []

    model = torch.load(r'model/best_agnews_1000_dist.pth')
    model = model.cuda()
    model.eval()

    with torch.no_grad():
        for i, (texts, real_labels, known_labels) in enumerate(labeled_loader):
            line = read_file.readline().split('\t')[2]
            outputs = model(texts.cuda())
            total_ += 1

            # 在实际标签上评测性能（实际标签是正确的）
            # write_file.write(str(torch.argmax(outputs.cpu(), dim=1).numpy()[0]) + '\t' + line)
            true_labels.extend(real_labels.tolist())
            pred_labels.extend(torch.argmax(outputs.cpu(), dim=1).tolist())

            if torch.argmax(outputs.cpu(), dim=1) == real_labels:
                right_ += 1

        print('正确数量: ', right_, ' 总样本数: ', total_)

        accuracy = right_ / total_
        precision, recall, f1_score = calculate_metrics(pred_labels, true_labels, num_classes)

        print('准确率: ', accuracy)
        print('精确率: ', precision)
        print('召回率: ', recall)
        print('F1分数: ', f1_score)

        return accuracy


if __name__ == '__main__':
    labeled_data_path = r'processed_data/labeled_dataset.csv'
    unlabeled_data_path = r'processed_data/unlabeled_dataset.csv'
    word2index = get_word2index(labeled_data_path, unlabeled_data_path)
    labeled_dataset = TextDataset(r'processed_data/dataset.csv', word2index)
    labeled_loader = DataLoader(labeled_dataset, batch_size=1, shuffle=False)

    num_classes = 4

    test(labeled_loader, num_classes)
