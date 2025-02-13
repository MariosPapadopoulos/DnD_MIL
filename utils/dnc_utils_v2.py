import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Subset
import os
import json
import copy
from sklearn.cluster import KMeans
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc

from .utils import *
from .file_utils import *
from dataset.dataset_generic import save_splits
from .fairlets import *


def setup_writer(args, cur):
    """Creates tensorboard writer"""
    if args.log_data:
        from torch.utils.tensorboard import SummaryWriter
        writer_dir = os.path.join(args.results_dir, 'tb_split' + str(cur))
        if not os.path.isdir(writer_dir):
            os.mkdir(writer_dir)
        writer = SummaryWriter(writer_dir, flush_secs=15)
    else:
        writer = None
    return writer


def compute_class_weights(loader, n_classes):
    """Computes class weights for criterion/loss function"""
    # Initialize counts for all classes
    class_counts = {i: 0 for i in range(n_classes)}
    total_samples = 0
    for _, label in loader:
        if isinstance(label, torch.Tensor):
            label = label.item()
        if label not in class_counts:
            print(f"Warning: Label {label} not in range(n_classes)")
            continue
        class_counts[label] += 1
        total_samples += 1

    # Compute weights for all classes
    num_classes_present = sum(1 for count in class_counts.values() if count > 0)
    class_weights = {}
    for cls in range(n_classes):
        count = class_counts[cls]
        if count == 0:
            print(f"Warning: Class {cls} has zero samples. Assigning a weight of zero.")
            class_weights[cls] = 0.0  # Assign zero weight to missing classes
        else:
            class_weights[cls] = total_samples / (num_classes_present * count)

    total_weight = sum(class_weights.values())
    if total_weight > 0:
        for cls in range(n_classes):
            if class_weights[cls] > 0:
                class_weights[cls] /= total_weight
            else:
                class_weights[cls] = 0.0
            print(f"Weight for class {cls}: {class_weights[cls]:.4f}")
    else:
        print("Warning: Total weight is zero. Assigning equal weights.")
        for cls in range(n_classes):
            class_weights[cls] = 1.0 / n_classes

    weight_tensor = torch.FloatTensor([class_weights[i] for i in range(n_classes)])
    return weight_tensor


def get_optim(model, opt_name, args):
    """Returns optimizer"""
    if opt_name == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
    elif opt_name == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    elif opt_name == 'adamw':
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
    elif opt_name == 'radam':
        return torch.optim.RAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    return optimizer


def get_criterion(loader, args):
    """Returns criterion/loss function, either weighted or not"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.use_weighted_ce:
        weights = compute_class_weights(loader, args.n_classes).to(device)
        return nn.CrossEntropyLoss(weight=weights)
    else:
        return nn.CrossEntropyLoss()


def initialize_model(args):
    """Creates and returns MIL model"""
    if args.model_type == 'mean_mil':
        from models.Mean_Max_MIL import MeanMIL
        model = MeanMIL(args.in_dim, args.n_classes)
    elif args.model_type == 'max_mil':
        from models.Mean_Max_MIL import MaxMIL
        model = MaxMIL(args.in_dim, args.n_classes)
    elif args.model_type == 'att_mil':
        from models.ABMIL import DAttention
        model = DAttention(args.in_dim, args.n_classes, dropout=args.drop_out, act='relu')
    elif args.model_type == 'trans_mil':
        from models.TransMIL import TransMIL
        model = TransMIL(args.in_dim, args.n_classes, dropout=args.drop_out, act='relu')
    elif args.model_type == 's4model':
        from models.S4MIL import S4Model
        model = S4Model(in_dim=args.in_dim, n_classes=args.n_classes, act='gelu', dropout=args.drop_out)
    elif args.model_type == 'mamba_mil':
        from models.MambaMIL import MambaMIL
        model = MambaMIL(in_dim=args.in_dim, n_classes=args.n_classes, dropout=args.drop_out, act='gelu', layer=args.mambamil_layer, rate=args.mambamil_rate, type=args.mambamil_type)
    else:
        raise NotImplementedError(f'{args.model_type} is not implemented ...')
    model.relocate()
    return model


def main_loop(datasets, cur, args):

    print('\n' + '=' * 30)
    print(f'Training fold {cur}/{args.k - 1}'.center(30))
    print('=' * 30)

    # ============= writer .... =============
    writer = setup_writer(args, cur)

    # ============= get splits .... =============
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    # ============= loaders .... =============
    train_loader = get_split_loader(train_split, training=True, testing=args.testing, weighted=args.weighted_sample)
    val_loader = get_split_loader(val_split, testing=args.testing)
    test_loader = get_split_loader(test_split, testing=args.testing)

    # ============= Part 1. Train base model =============
    base_model, results_dict, test_auc, val_auc, test_acc, val_acc = train_base_model(cur, train_loader, val_loader, test_loader, writer, args)
    if not args.dnc:
        return results_dict, test_auc, val_auc, test_acc, val_acc
    save_results_to_file(test_auc, val_auc, test_acc, val_acc, 'base.json', cur, args)

    # ============= Part 2. Cluster logits =============
    logits = extract_logits(base_model, train_loader)
    cluster_labels = perform_clustering(logits, args.K, train_loader, args)
    _, counts = np.unique(cluster_labels, return_counts=True)
    print('Count for each label after running k-means:', counts)

    # ============= Part 3. Train expert models =============
    expert_models = train_expert_models(cur, train_split, val_loader, test_loader, cluster_labels, writer, args)

    # ============= Part 4. Distill =============
    _, results_dict, test_auc, val_auc, test_acc, val_acc = train_distillation_model(cur, base_model, expert_models, train_loader, val_loader, test_loader, writer, args)
    save_results_to_file(test_auc, val_auc, test_acc, val_acc, 'distill.json', cur, args)
    save_results_to_pkl(results_dict, args, cur)

    return results_dict, test_auc, val_auc, test_acc, val_acc


def train_base_model(cur, train_loader, val_loader, test_loader, writer, args):
    print('Training base model...')

    # =============== init loss, model, optim ... ===============
    model = initialize_model(args)
    criterion = get_criterion(train_loader, args)
    optimizer = get_optim(model, args.opt, args)

    # =============== variables to track best model ... ===============
    best_val_acc = 0.0
    best_epoch = 0
    ckpt_path = os.path.join(args.results_dir, "base_split_{}_best_checkpoint.pt".format(cur))

    # =============== routine ... ===============
    for epoch in range(args.max_epochs):
        train_one_epoch(epoch, model, train_loader, optimizer, args.n_classes, writer, criterion, 'base', args)
        val_loss, val_acc, val_auc = validate(epoch, model, val_loader, args.n_classes, writer, criterion, 'base', args)

        # save model if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)
            print(f"==> New best model saved at epoch {epoch} with validation accuracy: {val_acc:.4f}")

    # load best model
    model.load_state_dict(torch.load(ckpt_path))
    print(f"Best model from epoch {best_epoch} loaded with validation accuracy: {best_val_acc:.4f}")

    # ============== run inference and save logits ==============
    val_logits= 0

    # =============== val metrics ... ===============
    #patient_results, test_error, auc, acc_logger, all_logits, all_Y_hat, all_label
    _, val_error, val_auc, _, all_val_logits, all_Y_hat, all_labels = summary(model, val_loader, args.n_classes)
    print('[BASE] Validation error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))
    print('Saving [BASE] val tensors')

    if hasattr(args, "logits_dir") and args.logits_dir:
        complete_path = os.path.join(args.logits_dir, "base/val/")  # Fixed path join
        if not os.path.exists(complete_path):
            os.makedirs(complete_path)

        with open(os.path.join(complete_path, f"all_val_logits.pt"), 'wb') as f:
            torch.save(all_val_logits, f)
        with open(os.path.join(complete_path, f"all_val_Y_hat.pt"), 'wb') as f:
            torch.save(torch.from_numpy(all_Y_hat), f)
        with open(os.path.join(complete_path, f"all_val_labels.pt"), 'wb') as f:
            torch.save(torch.from_numpy(all_labels), f)

    # =============== test metrics ... ===============
    results_dict, test_error, test_auc, acc_logger, all_test_logits, all_Y_hat, all_labels = summary(model, test_loader, args.n_classes)
    print('[BASE] Testing error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))
    print('Saving [BASE] test tensors')
    if hasattr(args, "logits_dir") and args.logits_dir:
        complete_path = os.path.join(args.logits_dir, "base/test/")
   
        if not os.path.exists(complete_path):
            os.makedirs(complete_path)

        with open(os.path.join(complete_path, f"all_test_logits.pt"), 'wb') as f:
            torch.save(all_test_logits, f)
        with open(os.path.join(complete_path, f"all_test_Y_hat.pt"), 'wb') as f:
            torch.save(torch.from_numpy(all_Y_hat), f)
        with open(os.path.join(complete_path, f"all_test_labels.pt"), 'wb') as f:
            torch.save(torch.from_numpy(all_labels), f)

    # =============== test class metrics ... ===============
    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('[BASE] Class {}: Accuracy {}, Correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('base/test_class_{}_acc'.format(i), acc, 0)

    # =============== write final val/test metrics ... ===============
    if writer:
        writer.add_scalar('base/val_error', val_error, 0)
        writer.add_scalar('base/val_auc', val_auc, 0)
        writer.add_scalar('base/test_error', test_error, 0)
        writer.add_scalar('base/test_auc', test_auc, 0)
        writer.close()

    # =============== print final val/test metrics ... ===============
    print('\n' + '*' * 30)
    print('Base model results\n'.center(30))
    print(f'Fold {cur}/{args.k - 1} Results'.center(30))
    print(f'Testing AUC: {test_auc:.4f}'.center(30))
    print(f'Validation AUC: {val_auc:.4f}'.center(30))
    print(f'Testing Accuracy: {(1 - test_error):.4f}'.center(30))
    print(f'Validation Accuracy: {(1 - val_error):.4f}'.center(30))
    print('*' * 30 + '\n')

    return model, results_dict, test_auc, val_auc, 1 - test_error, 1 - val_error


def train_expert_models(cur, train_split, val_loader, test_loader, cluster_labels, writer, args):
    expert_models = []
    for k in range(args.K):
        print(f'Training expert model {k}/{args.K - 1}...')

        # =============== take subset of data ===============
        cluster_indices = np.where(cluster_labels == k)[0]
        cluster_split = Subset(train_split, cluster_indices)
        print('Taking cluster split with {} number of bags'.format(len(cluster_split)))
        cluster_loader = get_split_loader(cluster_split, training=True, testing=args.testing)

        # =============== init loss, model, optim ... ===============
        model = initialize_model(args)
        criterion = get_criterion(cluster_loader, args)
        optimizer = get_optim(model, args.opt, args)

        # =============== variables to track best model ... ===============
        best_val_acc = 0.0
        best_epoch = 0
        ckpt_path = os.path.join(args.results_dir, "expert_{}_split_{}_best_checkpoint.pt".format(k, cur))

        # =============== routine ... ===============
        for epoch in range(args.max_epochs):
            train_one_epoch(epoch, model, cluster_loader, optimizer, args.n_classes, writer, criterion, f'expert_{k}', args)
            val_loss, val_acc, val_auc = validate(epoch, model, val_loader, args.n_classes, writer, criterion, f'expert_{k}', args)

            # save model if validation accuracy improves
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(model.state_dict(), ckpt_path)
                print(f"==> Expert {k}: New best model saved at epoch {epoch} with validation accuracy: {val_acc:.4f}")

        # load best model
        model.load_state_dict(torch.load(ckpt_path))
        print(f"Expert {k}: Best model from epoch {best_epoch} loaded with validation accuracy: {best_val_acc:.4f}")
        expert_models.append(model)

        # =============== val metrics ... ===============
        #patient_results, test_error, auc, acc_logger, all_logits, all_Y_hat, all_label
        _, val_error, val_auc, _, _, _, _  = summary(model, val_loader, args.n_classes)
        

        # =============== test metrics ... ===============
        results_dict, test_error, test_auc, acc_logger, _, _, _ = summary(model, test_loader, args.n_classes)
        print('[EXPERT {}] Testing error: {:.4f}, ROC AUC: {:.4f}'.format(k, test_error, test_auc))

        # =============== test class metrics ... ===============
        for i in range(args.n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            print('[EXPERT {}] Class {}: Accuracy {}, Correct {}/{}'.format(k, i, acc, correct, count))
            if writer:
                writer.add_scalar('expert_{}/test_class_{}_acc'.format(k, i), acc, 0)

        # =============== write final val/test metrics ... ===============
        if writer:
            writer.add_scalar('expert_{}/val_error'.format(k), val_error, 0)
            writer.add_scalar('expert_{}/val_auc'.format(k), val_auc, 0)
            writer.add_scalar('expert_{}/test_error'.format(k), test_error, 0)
            writer.add_scalar('expert_{}/test_auc'.format(k), test_auc, 0)
            writer.close()

        # =============== print final val/test metrics ... ===============
        print('\n' + '*' * 30)
        print(f'Expert model {k}/{args.K - 1} results\n'.center(30))
        print(f'Fold {cur}/{args.k} Results'.center(30))
        print(f'Testing AUC: {test_auc:.4f}'.center(30))
        print(f'Validation AUC: {val_auc:.4f}'.center(30))
        print(f'Testing Accuracy: {(1 - test_error):.4f}'.center(30))
        print(f'Validation Accuracy: {(1 - val_error):.4f}'.center(30))
        print('*' * 30 + '\n')

    return expert_models


def train_distillation_model(cur, base_model, expert_models, train_loader, val_loader, test_loader, writer, args):
    print('Training distilled model...')

    # =============== init model, optim ... ===============
    if args.distill_random_init:
        distilled_model = initialize_model(args)
    else:
        distilled_model = copy.deepcopy(base_model)
    optimizer = get_optim(distilled_model, args.opt_distill, args)

    # =============== variables to track best model ... ===============
    best_val_acc = 0.0
    best_epoch = 0
    ckpt_path = os.path.join(args.results_dir, "distill_split_{}_best_checkpoint.pt".format(cur))

    # =============== routine ... ===============
    for epoch in range(args.max_epochs_distill):
        train_one_epoch_distill(epoch, distilled_model, base_model, expert_models, train_loader, optimizer, args.n_classes, writer, args)
        val_loss, val_acc, val_auc = validate_distill(epoch, distilled_model, base_model, expert_models, val_loader, args.n_classes, args, writer)

        # save model if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(distilled_model.state_dict(), ckpt_path)
            print(f"==> New best distilled model saved at epoch {epoch} with validation accuracy: {val_acc:.4f}")

    # load best model
    distilled_model.load_state_dict(torch.load(ckpt_path))
    print(f"Best distilled model from epoch {best_epoch} loaded with validation accuracy: {best_val_acc:.4f}")

    # =============== val metrics ... ===============
    _, val_error, val_auc, _, all_val_logits, all_Y_hat, all_labels = summary(distilled_model, val_loader, args.n_classes)
    print('[DISTILLED] Validation error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))
    print('Saving [DISTILLED] val tensors')

    if hasattr(args, "logits_dir") and args.logits_dir:
        complete_path= os.path.join(args.logits_dir, "distill/val/")   
        if not os.path.exists(complete_path):
            os.makedirs(complete_path)
        
        with open(os.path.join(complete_path, f"all_val_logits.pt"), 'wb') as f:
            torch.save(all_val_logits, f)
        with open(os.path.join(complete_path, f"all_val_Y_hat.pt"), 'wb') as f:
            torch.save(torch.from_numpy(all_Y_hat), f)
        with open(os.path.join(complete_path, f"all_val_labels.pt"), 'wb') as f:
            torch.save(torch.from_numpy(all_labels), f)

    # =============== test metrics ... ===============
    results_dict, test_error, test_auc, acc_logger, all_test_logits, all_Y_hat, all_labels = summary(distilled_model, test_loader, args.n_classes)
    print('[DISTILLED] Testing error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))
    print('Saving [DISTILLED] test tensors')

    if hasattr(args, "logits_dir") and args.logits_dir:

        complete_path= os.path.join(args.logits_dir, "distill/test/")
        if not os.path.exists(complete_path):
            os.makedirs(complete_path)
        
        with open(os.path.join(complete_path, f"all_test_logits.pt"), 'wb') as f:
            torch.save(all_test_logits, f)
        with open(os.path.join(complete_path, f"all_test_Y_hat.pt"), 'wb') as f:
            torch.save(torch.from_numpy(all_Y_hat), f)
        with open(os.path.join(complete_path, f"all_test_labels.pt"), 'wb') as f:
            torch.save(torch.from_numpy(all_labels), f)

    # =============== test class metrics ... ===============
    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('[DISTILLED] Class {}: Accuracy {}, Correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('distill/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('distill/val_error', val_error, 0)
        writer.add_scalar('distill/val_auc', val_auc, 0)
        writer.add_scalar('distill/test_error', test_error, 0)
        writer.add_scalar('distill/test_auc', test_auc, 0)
        writer.close()

    print('\n' + '*' * 30)
    print('Distillation model results\n'.center(30))
    print(f'Fold {cur}/{args.k - 1} Results'.center(30))
    print(f'Testing AUC: {test_auc:.4f}'.center(30))
    print(f'Validation AUC: {val_auc:.4f}'.center(30))
    print(f'Testing Accuracy: {(1 - test_error):.4f}'.center(30))
    print(f'Validation Accuracy: {(1 - val_error):.4f}'.center(30))
    print('*' * 30)

    return distilled_model, results_dict, test_auc, val_auc, 1 - test_error, 1 - val_error


def train_one_epoch(epoch, model, loader, optimizer, n_classes, writer=None, criterion=None, mode='base', args=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)

    # routine
    train_loss = 0.
    train_error = 0.
    for batch_idx, (data, label) in enumerate(loader):

        # get data
        data, label = data.to(device), label.to(device)

        # forward
        logits, Y_prob, Y_hat, _, _ = model(data)
        acc_logger.log(Y_hat, label)

        # calculate loss
        loss = criterion(logits, label)
        loss_value = loss.item()

        # misc
        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('[Training][Epoch {}] batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(epoch, batch_idx, loss_value, label.item(), data.size(0)))

        # calculate batch error
        error = calculate_error(Y_hat, label)
        train_error += error

        # backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    # print them
    print('[Training] Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('[Training] class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar(f'train_{mode}/class_{i}_acc', acc, epoch)

    # tensorboard log
    if writer:
        writer.add_scalar(f'train_{mode}/loss', train_loss, epoch)
        writer.add_scalar(f'train_{mode}/error', train_error, epoch)


def validate(epoch, model, loader, n_classes, writer=None, criterion=None, mode='base', args=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    epoch_val_logits = []


    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            # get data
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            # forward
            logits, Y_prob, Y_hat, _, _ = model(data)
            acc_logger.log(Y_hat, label)

            # save logits
            epoch_val_logits.append(logits)
            # file_path= os.path.join(args.logits_dir, f"{epoch}_{batch_idx}.pt")
            # torch.save(logits, f="args.logits_dir/")

            # calculate loss
            loss = criterion(logits, label)
            val_loss += loss.item()

            # get error
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            error = calculate_error(Y_hat, label)
            val_error += error

    # calculate loss, error, and auc for epoch
    val_error /= len(loader)
    val_loss /= len(loader)
    val_acc = 1 - val_error
    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    #save val logits for epoch
    # inner_folder = args.logits_dir + "/" + mode
    # #create inner folder if it does not exist
    # if not os.path.exists(inner_folder):
    #     os.makedirs(inner_folder)

    # pred_file_path= os.path.join(inner_folder, f"logits_epoch_{epoch}.pt")
    # torch.save(epoch_val_logits, pred_file_path) #save logits
    # label_file_path= os.path.join(inner_folder, f"labels_epoch_{epoch}.pt")
    # torch.save(torch.from_numpy(labels), label_file_path) #save labels



    # print them
    print('[Validation] val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('[Validation] class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

    # tensorboard log
    if writer:
        writer.add_scalar(f'val_{mode}/loss', val_loss, epoch)
        writer.add_scalar(f'val_{mode}/auc', auc, epoch)
        writer.add_scalar(f'val_{mode}/error', val_error, epoch)

    return val_loss, val_acc, auc


def train_one_epoch_distill(epoch, student_model, base_model, expert_models, loader, optimizer, n_classes, writer, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)

    # prepare models
    student_model.train()
    base_model.eval()
    for expert_model in expert_models:
        expert_model.eval()

    # routine
    train_loss = 0.
    train_error = 0.
    for batch_idx, (data, label) in enumerate(loader):

        # get data
        data, label = data.to(device), label.to(device)

        # student forward pass
        student_logits, Y_prob, Y_hat, _, _ = student_model(data)
        acc_logger.log(Y_hat, label)

        # teacher (base model) forward pass
        with torch.no_grad():
            base_logits, _, _, _, _ = base_model(data)

        # expert models forward passes
        expert_logits_list = []
        for expert_model in expert_models:
            with torch.no_grad():
                expert_logits, _, _, _, _ = expert_model(data)
                expert_logits_list.append(expert_logits)

        # compute distillation loss
        loss = compute_distillation_loss(student_logits, base_logits, expert_logits_list, label, args)
        loss_value = loss.item()

        # misc
        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('[Distillation][Epoch {}] batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(epoch, batch_idx, loss_value, label.item(), data.size(0)))

        # calculate batch error
        error = calculate_error(Y_hat, label)
        train_error += error

        # backwards
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    # print them
    print('[Distillation] Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('[Distillation] class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar(f'train_distill/class_{i}_acc', acc, epoch)

    # tensorboard log
    if writer and acc is not None:
        writer.add_scalar('train_distill/loss', train_loss, epoch)
        writer.add_scalar('train_distill/error', train_error, epoch)


def validate_distill(epoch, student_model, base_model, expert_models, loader, n_classes, args, writer=None, mode='distill'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student_model.eval()
    base_model.eval()
    for expert_model in expert_models:
        expert_model.eval()

    acc_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    base_logits_list= []
    student_logits_list = []

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            # forward pass through the student model
            student_logits, Y_prob, Y_hat, _, _ = student_model(data)
            acc_logger.log(Y_hat, label)
            student_logits_list.append(student_logits)

            # forward pass through the base (teacher) model
            base_logits, _, _, _, _ = base_model(data)
            base_logits_list.append(base_logits)

            # forward pass through the expert models
            expert_logits_list = []
            for expert_model in expert_models:
                expert_logits, _, _, _, _ = expert_model(data)
                expert_logits_list.append(expert_logits)

            # compute distillation loss
            loss = compute_distillation_loss(student_logits, base_logits, expert_logits_list, label, args)
            val_loss += loss.item()

            # get error
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            error = calculate_error(Y_hat, label)
            val_error += error

    # calculate loss, error, and auc for epoch
    val_error /= len(loader)
    val_loss /= len(loader)
    val_acc = 1 - val_error
    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    #epoch-level saving of logits
    # inner_folder = args.logits_dir + "/" + mode
    # #create inner folder if it does not exist
    # if not os.path.exists(inner_folder):
    #     os.makedirs(inner_folder)

    # base_file_path= os.path.join(inner_folder, f"base_epoch_{epoch}.pt")
    # student_file_path= os.path.join(inner_folder, f"student_epoch_{epoch}.pt")
    # expert_file_path= os.path.join(inner_folder, f"expert_epoch_{epoch}.pt")
    # label_file_path= os.path.join(inner_folder, f"labels_epoch_{epoch}.pt")

    # torch.save(base_logits_list, base_file_path)
    # torch.save(student_logits_list, student_file_path)
    # torch.save(expert_logits_list, expert_file_path)
    # torch.save(torch.from_numpy(labels), label_file_path)



    # print them
    print('[Validation] val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('[Validation] class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

    # tensorboard log
    if writer:
        writer.add_scalar(f'val_{mode}/loss', val_loss, epoch)
        writer.add_scalar(f'val_{mode}/auc', auc, epoch)
        writer.add_scalar(f'val_{mode}/error', val_error, epoch)

    return val_loss, val_acc, auc


def compute_distillation_loss(student_logits, base_logits, expert_logits_list, labels, args):
    """Computes KL divergence of an ensemble of expert models and CE on student model and logits"""
    # ce between student logits and labels
    ce_loss = nn.CrossEntropyLoss()(student_logits, labels)

    # kl-div between student and base logits
    kl_loss = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(student_logits / args.temperature, dim=1),
        F.softmax(base_logits / args.temperature, dim=1)
    ) * (args.temperature ** 2)

    # kl-div between student and expert logits
    expert_kl_losses = [
        nn.KLDivLoss(reduction='batchmean')(
            F.log_softmax(student_logits / args.temperature, dim=1),
            F.softmax(expert_logits / args.temperature, dim=1)
        ) * (args.temperature ** 2)
        for expert_logits in expert_logits_list
    ]

    expert_kl_loss = sum(expert_kl_losses) / len(expert_kl_losses)

    total_loss = args.ce_weight * ce_loss + \
                 args.kl_weight_base * kl_loss + \
                 args.kl_weight_expert * expert_kl_loss
    return total_loss


def summary(model, loader, n_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = []

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    all_Y_hat = []
    all_label = []
    all_logits = []
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad(): 
            logits, Y_prob, Y_hat, _, _ = model(data)
            all_logits.append(logits.cpu().numpy())

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds.extend(Y_hat.cpu().numpy())

        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'logits': logits.cpu().numpy(), 'Y_hat': Y_hat.cpu().numpy(),
                                            'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

        all_Y_hat.append(Y_hat.cpu().numpy())
        all_label.append(label.cpu().numpy())

    test_error /= len(loader)
    all_Y_hat = np.concatenate(all_Y_hat)
    all_label = np.concatenate(all_label)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))
        auc = np.nanmean(np.array(aucs))

    return patient_results, test_error, auc, acc_logger, all_logits, all_Y_hat, all_label


def perform_clustering(logits, n_clusters, loader, args):
    """Cluster logits into a specific number of clusters"""
    samples = logits.shape[0]
    labels = []
    for _, l in loader:
        labels.append(l)
    labels = torch.cat(labels).numpy()

    n_classes = len(np.unique(labels))

    if args.use_fairlets:
        class_counts = np.bincount(labels)
        total_samples = np.sum(class_counts)
        class_distribution = {
            cls: int(count * args.fairlet_size / total_samples)
            for cls, count in enumerate(class_counts)
        }
        for cls in class_distribution:
            class_distribution[cls] = max(1, class_distribution[cls])
        args.fairlet_size = sum(class_distribution.values())
        fairlets = construct_fairlets_multiclass(labels, class_distribution)
        cluster_labels = cluster_fairlets(logits, fairlets, n_clusters)
    elif args.use_constrained_kmeans:
        from k_means_constrained import KMeansConstrained
        kmeans = KMeansConstrained(
            n_clusters=n_clusters,
            size_min=int(samples / n_clusters - samples / 20),
            size_max=int(samples / n_clusters + samples / 20),
            random_state=args.seed,
        )
        cluster_labels = kmeans.fit_predict(logits)
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=args.seed)
        cluster_labels = kmeans.fit_predict(logits)

    # print cluster details
    for cluster_id in range(n_clusters):
        indices = np.where(cluster_labels == cluster_id)[0]
        cluster_size = len(indices)
        cluster_labels_in_cluster = labels[indices]
        class_counts_in_cluster = np.bincount(cluster_labels_in_cluster, minlength=n_classes)
        class_counts_str = ', '.join([f'class {i} -> {count} points' for i, count in enumerate(class_counts_in_cluster)])
        print(f'Cluster {cluster_id}: {cluster_size} points, {class_counts_str}')

    return cluster_labels


def extract_logits(model, loader):
    """Extracts logits from MIL model"""
    slide_ids = loader.dataset.slide_data['slide_id']
    model.eval()
    all_logits = []
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            slide_id = slide_ids.iloc[batch_idx]
            logits, _, _, _, _ = model(data.to(torch.device('cuda')))
            all_logits.append(logits.cpu().numpy())
    return np.concatenate(all_logits)


def save_results_to_file(test_auc, val_auc, test_acc, val_acc, filename, cur, args):
    """Saves `summary` results to a json file"""
    filepath = os.path.join(args.results_dir, "split_{}_{}".format(cur, filename))
    results = {
        'test_auc': test_auc,
        'val_auc': val_auc,
        'test_acc': test_acc,
        'val_acc': val_acc
    }
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filepath}")


def save_results_to_pkl(results_dict, args, cur):
    """Saves `summary` results to a pkl file"""
    os.makedirs(args.results_dir, exist_ok=True)
    filename = os.path.join(args.results_dir, f'split_{cur}_results.pkl')
    save_pkl(filename, results_dict)
    print(f"Results saved to {filename}")
