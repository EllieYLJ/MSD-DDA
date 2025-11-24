# standard
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import copy
import random
import time
import math
#from torch.utils.tensorboard import SummaryWriter

# Custom modules
import utils
import models

# Set random seed for reproducibility
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(20)

# writer = SummaryWriter()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class MSMDAER():
    def __init__(self, model=models.MSMDAERNet(), source_loaders=6, target_loader=1, batch_size=64, iteration=10000, lr=0.001, momentum=0.9, log_interval=10):
        self.model = model
        self.model.to(device)
        self.source_loaders = source_loaders
        self.target_loader = target_loader
        self.batch_size = batch_size
        self.iteration = iteration
        self.lr = lr
        self.momentum = momentum
        self.log_interval = log_interval

        if not hasattr(self, 'domain_weights'):
            self.domain_weights = [1.0 / len(self.source_loaders) for _ in range(len(self.source_loaders))]

    def __getModel__(self):
        return self.model

    def calculate_domain_weights1(self, avg_mmd_losses, avg_cls_losses):
        eta = 2
        alpha = 10
        beta = 0
        weights_unnormalized = [math.exp(eta / (alpha * (mmd + 1e-8) + beta * (cls + 1e-8))) 
                                for mmd, cls in zip(avg_mmd_losses, avg_cls_losses)]
        print("Unnormalized weights:", [round(w, 4) for w in weights_unnormalized])

        normalized_weights = [w / sum(weights_unnormalized) for w in weights_unnormalized]
        print("Normalized weights:", [round(w, 4) for w in normalized_weights])
        return normalized_weights

    def calculate_domain_weights2(self, avg_mmd_losses, avg_cls_losses):
        sorted_indices = sorted(range(len(avg_cls_losses)), key=lambda k: avg_cls_losses[k])
        top_k = min(3, len(avg_cls_losses))
        normalized_weights = [0.0] * len(avg_cls_losses)
        weight_per_top = 1.0 / top_k
        
        for idx in range(top_k):
            normalized_weights[sorted_indices[idx]] = weight_per_top
        
        print("Normalized weights (top-k):", [round(w, 4) for w in normalized_weights])
        return normalized_weights

    def train(self):
        source_iters = []
        for i in range(len(self.source_loaders)):
            source_iters.append(iter(self.source_loaders[i]))
        target_iter = iter(self.target_loader)
        best_correct = 0

        mmd_losses_per_domain = [[] for _ in range(len(self.source_loaders))]
        cls_losses_per_domain = [[] for _ in range(len(self.source_loaders))]

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for iter_num in range(1, self.iteration + 1):
            self.model.train()
            LEARNING_RATE = self.lr
            # LEARNING_RATE = self.lr / math.pow((1 + 10 * (iter_num - 1) / self.iteration), 0.75)
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE

            for domain_idx in range(len(source_iters)):
                try:
                    source_data, source_label = next(source_iters[domain_idx])
                except Exception as err:
                    source_iters[domain_idx] = iter(self.source_loaders[domain_idx])
                    source_data, source_label = next(source_iters[domain_idx])
                
                try:
                    target_data, _ = next(target_iter)
                except Exception as err:
                    target_iter = iter(self.target_loader)
                    target_data, _ = next(target_iter)
                
                source_data, source_label = source_data.to(device), source_label.to(device)
                target_data = target_data.to(device)

                cls_loss, mmd_loss, lsd_loss, transfer_loss = self.model(
                    source_data, 
                    number_of_source=len(source_iters), 
                    data_tgt=target_data, 
                    label_src=source_label, 
                    mark=domain_idx
                )

                LOSS_WEIGHT = 1.0 / (1.0 + torch.exp(torch.tensor(100.0 - iter_num, device=device)))
                gamma = LOSS_WEIGHT / 50

                total_loss = (self.domain_weights[domain_idx] * cls_loss +
                              0.3 * ((1 - LOSS_WEIGHT) * mmd_loss + LOSS_WEIGHT * lsd_loss) +
                              gamma * transfer_loss)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                mmd_losses_per_domain[domain_idx].append(mmd_loss.item())
                cls_losses_per_domain[domain_idx].append(cls_loss.item())

                if iter_num % self.log_interval == 0:
                    print(
                        f'Train Source{domain_idx}, Iter: {iter_num} [{100. * iter_num / self.iteration:.0f}%]\t'
                        f'Total Loss: {total_loss.item():.6f}\t'
                        f'Cls Loss: {cls_loss.item():.6f}\t'
                        f'MMD Loss: {mmd_loss.item():.6f}\t'
                        f'LSD Loss: {lsd_loss.item():.6f}\t'
                        f'Transfer Loss: {transfer_loss.item():.6f}'
                    )

            if iter_num % (self.log_interval * 20) == 0:
                avg_mmd_losses = [sum(losses) / len(losses) for losses in mmd_losses_per_domain]
                avg_cls_losses = [sum(losses) / len(losses) for losses in cls_losses_per_domain]
                
                self.domain_weights = self.calculate_domain_weights2(avg_mmd_losses, avg_cls_losses)
                
                current_correct = self.test(iter_num)
                if current_correct > best_correct:
                    best_correct = current_correct
                print(f'Current Best Correct Samples: {best_correct.item()}\n')

        avg_mmd_losses = [sum(losses) / len(losses) for losses in mmd_losses_per_domain]
        avg_cls_losses = [sum(losses) / len(losses) for losses in cls_losses_per_domain]
        self.domain_weights = self.calculate_domain_weights2(avg_mmd_losses, avg_cls_losses)

        print("\n=== Training Completed ===")
        print(f"Average MMD Losses per Domain: {[round(loss, 4) for loss in avg_mmd_losses]}")
        print(f"Average Cls Losses per Domain: {[round(loss, 4) for loss in avg_cls_losses]}")
        print(f"Final Domain Weights: {[round(w, 4) for w in self.domain_weights]}")

        target_dataset_size = len(self.target_loader.dataset)
        best_acc = 100. * best_correct / target_dataset_size
        return best_acc

    def test(self, iter_num):
        self.model.eval()
        test_loss = 0.0
        correct = 0
        domain_corrects = [0 for _ in range(len(self.source_loaders))]

        with torch.no_grad():
            for data, target in self.target_loader:
                data = data.to(device)
                target = target.to(device).squeeze()

                preds = self.model(data, len(self.source_loaders))

                assert len(self.domain_weights) == len(self.source_loaders), \
                    f"Domain weights length ({len(self.domain_weights)}) != Source domains count ({len(self.source_loaders)})"
                assert len(preds) == len(self.source_loaders), \
                    f"Preds length ({len(preds)}) != Source domains count ({len(self.source_loaders)})"

                preds_softmax = [F.softmax(pred, dim=1) for pred in preds]

                fused_pred = sum(w * pred for w, pred in zip(self.domain_weights, preds_softmax))
                
                test_loss += F.nll_loss(torch.log(fused_pred + 1e-8), target).item()

                pred_label = fused_pred.data.max(1)[1]
                correct += pred_label.eq(target.data).cpu().sum()

                for domain_idx in range(len(preds_softmax)):
                    domain_pred_label = preds_softmax[domain_idx].data.max(1)[1]
                    domain_corrects[domain_idx] += domain_pred_label.eq(target.data).cpu().sum()

        avg_test_loss = test_loss / len(self.target_loader.dataset)
        # writer.add_scalar("Test/Average Loss", avg_test_loss, iter_num)

        target_dataset_size = len(self.target_loader.dataset)
        print(f'\nTest (Iter {iter_num}):')
        print(f'Average Loss: {avg_test_loss:.4f}')
        print(f'Fused Accuracy: {correct}/{target_dataset_size} ({100. * correct / target_dataset_size:.2f}%)')
        for domain_idx in range(len(domain_corrects)):
            print(f'Source{domain_idx} Accuracy: {domain_corrects[domain_idx]}/{target_dataset_size} '
                  f'({100. * domain_corrects[domain_idx] / target_dataset_size:.2f}%)')
        print("-" * 50)

        return correct

def cross_subject(data, label, session_id, subject_id, category_number, batch_size, iteration, lr, momentum, log_interval):
    one_session_data, one_session_label = copy.deepcopy(data[session_id]), copy.deepcopy(label[session_id])
    
    train_idxs = list(range(9))
    del train_idxs[subject_id]
    test_idx = subject_id
    
    target_data, target_label = copy.deepcopy(one_session_data[test_idx]), copy.deepcopy(one_session_label[test_idx])
    source_data, source_label = copy.deepcopy([one_session_data[idx] for idx in train_idxs]), \
                                copy.deepcopy([one_session_label[idx] for idx in train_idxs])

    del one_session_data, one_session_label

    source_loaders = []
    for j in range(len(source_data)):
        source_dataset = utils.CustomDataset(source_data[j], source_label[j])
        source_loader = torch.utils.data.DataLoader(
            dataset=source_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )
        source_loaders.append(source_loader)

    target_dataset = utils.CustomDataset(target_data, target_label)
    target_loader = torch.utils.data.DataLoader(
        dataset=target_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    model = MSMDAER(
        model=models.MSMDAERNet(pretrained=False, number_of_source=len(source_loaders), number_of_category=category_number),
        source_loaders=source_loaders,
        target_loader=target_loader,
        batch_size=batch_size,
        iteration=iteration,
        lr=lr,
        momentum=momentum,
        log_interval=log_interval
    )
    acc = model.train()
    print(f'\n=== Cross-Subject Result ===')
    print(f'Target Subject ID: {test_idx}, Session ID: {session_id}, Best Accuracy: {acc:.2f}%\n')
    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MS-MDAER Model Parameters (Cross-Subject Only)')
    parser.add_argument('--dataset', type=str, default='seed3',
                        help='Dataset used for MS-MDAER, "seed3" or "seed4"')
    parser.add_argument('--norm_type', type=str, default="ele",
                        help='Normalization type for data, "ele", "sample", "global" or "none"')
    parser.add_argument('--batch_size', type=int, default=18,
                        help='Batch size (integer)')
    parser.add_argument('--epoch', type=int, default=200,
                        help='Training epoch (integer)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    args = parser.parse_args()
    dataset_name = args.dataset
    bn = args.norm_type

    from scipy.io import loadmat
    data0 = loadmat('D://DATA/MSD-DDA-main/f9_144data.mat')['data']
    data = np.zeros((1, data0.shape[0], data0.shape[1], data0.shape[2]))
    data[0] = data0
    label0 = loadmat('D://DATA/MSD-DDA-main/f9_144data.mat')['new_label']
    label = np.zeros((1, label0.shape[0], label0.shape[1], 1))
    label[0, :, :, 0] = label0
    print('Normalization type: ', bn)
    
    if bn == 'ele':
        data_tmp = copy.deepcopy(data)
        label_tmp = copy.deepcopy(label)
        for i in range(len(data_tmp)):
            for j in range(len(data_tmp[0])):
                data_tmp[i][j] = utils.norminy(data_tmp[i][j])
    elif bn == 'sample':
        data_tmp = copy.deepcopy(data)
        label_tmp = copy.deepcopy(label)
        for i in range(len(data_tmp)):
            for j in range(len(data_tmp[0])):
                data_tmp[i][j] = utils.norminx(data_tmp[i][j])
    elif bn == 'global':
        data_tmp = copy.deepcopy(data)
        label_tmp = copy.deepcopy(label)
        for i in range(len(data_tmp)):
            for j in range(len(data_tmp[0])):
                data_tmp[i][j] = utils.normalization(data_tmp[i][j])
    elif bn == 'none':
        data_tmp = copy.deepcopy(data)
        label_tmp = copy.deepcopy(label)
    else:
        pass

    trial_total, category_number, _ = utils.get_number_of_label_n_trial(
        dataset_name)

    batch_size = args.batch_size
    epoch = args.epoch
    lr = args.lr
    print('BS: {}, epoch: {}'.format(batch_size, epoch))
    momentum = 0.9
    log_interval = 10
    iteration = 0
    if dataset_name == 'seed3':
        iteration = 1000#math.ceil(epoch*200/batch_size)
    elif dataset_name == 'seed4':
        iteration = math.ceil(epoch*820/batch_size)
    else:
        iteration = 5000
    print('Iteration: {}'.format(iteration))

    csub = []
    for subject_id_main in range(9):
        csub.append(cross_subject(data_tmp, label_tmp, 0, subject_id_main, 2,
                                batch_size, iteration, lr, momentum, log_interval))
    print("Cross-subject: ", csub)
    print("Cross-subject mean: ", np.mean(csub), "std: ", np.std(csub))