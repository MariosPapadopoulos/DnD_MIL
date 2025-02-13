from __future__ import print_function
import argparse
import os
import datetime
import torch
import pandas as pd
import numpy as np
from datetime import datetime

from utils.file_utils import save_pkl
from utils.utils import *
from utils.dnc_utils_v2 import main_loop # best val acc
#from utils.dnc_utils import main_loop # early stopping
from dataset.dataset_generic import Generic_MIL_Dataset


parser = argparse.ArgumentParser(description='Configurations for WSI Training')

#################
##### data ######
#################
parser.add_argument('--data_root_dir', type=str, default=None, help='data directory')
parser.add_argument('--label_frac', type=float, default=1.0, help='fraction of training labels (default: 1.0)')
parser.add_argument('--split_dir', type=str, default=None, help='manually specify the set of splits to use')
parser.add_argument('--task', type=str, help='specify the task for the training')
parser.add_argument('--backbone', type=str, default='resnet50', help='model backbone (default: resnet50)')
parser.add_argument('--patch_size', type=str, default='', help='size of patches to extract (default: unspecified)')
parser.add_argument('--preloading', type=str, default='no', help='whether to preload data')

#################
###### optim ####
#################
parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam', help='optimizer type (default: adam)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 0.0001)')
parser.add_argument('--reg', type=float, default=1e-5, help='weight decay (default: 1e-5)')
parser.add_argument('--max_epochs', type=int, default=200, help='maximum number of epochs to train')
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout probability (default: 0.25)')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')

#################
## experiment ###
#################
parser.add_argument('--model_type', type=str,  choices=['mean_mil', 'max_mil', 'att_mil','trans_mil', 's4model','mamba_mil'], default='mean_mil', help='type of model')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--log_data', action='store_true', default=True, help='log data using tensorboard')

#################
##### other #####
#################
parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiments (default: 1)')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--k', type=int, default=5, help='number of folds (default: 5)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, use last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, use first fold)')
parser.add_argument('--in_dim', type=int, default=1024, help='input dimension of the model (default: 1024)')
parser.add_argument('--testing', action='store_true', default=False, help='enable testing mode for debugging')
parser.add_argument('--save_logits', action='store_true', default=False, help='save logits for visualization')

# mambamil
parser.add_argument('--mambamil_rate',type=int, default=10, help='mambamil_rate')
parser.add_argument('--mambamil_layer',type=int, default=2, help='mambamil_layer')
parser.add_argument('--mambamil_type',type=str, default='SRMamba', choices= ['Mamba', 'BiMamba', 'SRMamba'], help='mambamil_type')

# Distill
parser.add_argument('--distill_random_init', action='store_true', default=False, help='If True, the final distillation model is randomly initialized. If False, it\'s initialized from the base model')
parser.add_argument('--temperature', type=float, default=0.2, help='Temperature parameter for KL-divergence in knowledge distillation (default: 0.2)')
parser.add_argument('--ce_weight', type=float, default=1.0, help='Weight for the cross-entropy loss in knowledge distillation (default: 1.0)')
parser.add_argument('--kl_weight_base', type=float, default=1.0, help='Weight for the KL-divergence loss in knowledge distillation (default: 1.0)')
parser.add_argument('--kl_weight_expert', type=float, default=1.0, help='Weight for the KL-divergence loss in knowledge distillation (default: 1.0)')
parser.add_argument('--max_epochs_distill', type=int, default=400, help='Maximum number of epochs for the distillation phase (default: 200)')
parser.add_argument('--opt_distill', type=str, choices=['adam', 'sgd', 'radam', 'adamw'], default='sgd', help='optimizer type (default: sgd)')

# Divide and Conquer
parser.add_argument('--patience_base', type=int, default=15, help='Patience for early stopping during base model training (default: 15)')
parser.add_argument('--patience_expert', type=int, default=15, help='Patience for early stopping during expert model training (default: 5)')
parser.add_argument('--patience_distill', type=int, default=30, help='Patience for early stopping during distillation phase (default: 200)')
parser.add_argument('--use_weighted_ce', action='store_true', default=False, help='Use weighted cross-entropy loss')
parser.add_argument('--dnc', action='store_true', default=True, help='If True, use the Divide-and-Conquer approach for training')

# clustering
parser.add_argument('--K', type=int, default=5, help='Number of clusters to divide the dataset into for the DNC approach (default: 5)')
parser.add_argument('--use_constrained_kmeans', action='store_true', default=False, help='Use constrained k-means clustering')
parser.add_argument('--use_fairlets', action='store_true', default=False, help='Use fairlets clustering')
parser.add_argument('--min_cluster_size', type=int, default=5, help='min cluster size') ##int(samples/n_clusters - samples/20)
parser.add_argument('--max_cluster_size', type=int, default=100, help='max cluster size') ##int(samples/n_clusters + samples/20)
parser.add_argument('--fairlet_size', type=int, default=10, help='fairlet size') 

args = parser.parse_args()

#args.dnc = True
#args.split_dir= './/data//splits//CAMELYON_16'
#args.task= 'CAMELYON_16'
#args.backbone= 'resnet50'
#args.log_data= True
#args.weighted_sample= True
#args.k=1
#args.results_dir = './/results//'
#args.use_weighted_ce = True

#args.model_type= 'mean_mil'
#args.exp_code = 'test' 
#args.K = 4
#args.use_fairlets = True


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    
    # =============== k-fold ... ===============
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        
        # =============== get data ... ===============
        train_dataset, val_dataset, test_dataset = dataset.return_splits(args.backbone, args.patch_size, from_id=False, csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        
        datasets = (train_dataset, val_dataset, test_dataset)
        
        if args.preloading == 'yes':
            for d in datasets:
                d.pre_loading()
            
        # =============== train and val ... ===============
        results, test_auc, val_auc, test_acc, val_acc  = main_loop(datasets, i, args)

        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        
        # write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    # save k-fold results
    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
        
    final_df.to_csv(os.path.join(args.results_dir, save_name))
    mean_auc_test = final_df['test_auc'].mean()
    std_auc_test = final_df['test_auc'].std()
    mean_auc_val = final_df['val_auc'].mean()
    std_auc_val = final_df['val_auc'].std()
    mean_acc_test = final_df['test_acc'].mean()
    std_acc_test = final_df['test_acc'].std()
    mean_acc_val = final_df['val_acc'].mean()
    std_acc_val = final_df['val_acc'].std()

    #wandb.log({"mean_auc_test": mean_auc_test, "std_auc_test": std_auc_test, "mean_auc_val": mean_auc_val, "std_auc_val": std_auc_val})
    df_append = pd.DataFrame({
        'folds': ['mean', 'std'],
        'test_auc': [mean_auc_test, std_auc_test],
        'val_auc': [mean_auc_val, std_auc_val],
        'test_acc': [mean_acc_test, std_acc_test],
        'val_acc': [mean_acc_val, std_acc_val],
    })
    final_df = pd.concat([final_df, df_append])
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))
    final_df['folds'] = final_df['folds'].astype(str)

    


def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    
    seed_torch(args.seed)

    encoding_size = 1024
    settings = {'num_splits': args.k, 
                'k_start': args.k_start,
                'k_end': args.k_end,
                'task': args.task,
                'max_epochs': args.max_epochs, 
                'results_dir': args.results_dir, 
                'lr': args.lr,
                'experiment': args.exp_code,
                'reg': args.reg,
                'label_frac': args.label_frac,
                'seed': args.seed,
                'model_type': args.model_type,
                "use_drop_out": args.drop_out,
                'weighted_sample': args.weighted_sample,
                'opt': args.opt,
                }

    # =============== get dataset ... ===============
    print('Load Dataset')
    if args.task == 'LUAD_LUSC':
        args.n_classes=2
        dataset = Generic_MIL_Dataset(csv_path = './/data//dataset_csv/LUAD_LUSC_mod.csv',
                                data_dir= None,
                                shuffle = False, 
                                seed = args.seed, 
                                print_info = True,
                                label_dict = {'LUAD':0, 'LUSC':1},
                                patient_strat=False,
                                ignore=[])

    elif args.task == 'BRACS':
        args.n_classes=7
        dataset = Generic_MIL_Dataset(csv_path = './/data//dataset_csv/BRACS.csv',
                                data_dir= None,
                                shuffle = False, 
                                seed = args.seed, 
                                print_info = True,
                                label_dict = {'PB':0, 'IC':1, 'DCIS':2, 'N':3, 'ADH': 4, 'FEA':5, 'UDH': 6 },
                                patient_strat=False,
                                ignore=[])
    elif args.task == 'CAMELYON_16':
        args.n_classes=2  
        dataset = Generic_MIL_Dataset(csv_path = './/data//dataset_csv//CAMELYON_16.csv',
                                      data_dir= None,
                                      shuffle = False, 
                                      seed= args.seed,
                                      print_info = True,
                                      label_dict={'normal': 0, 'tumour': 1},
                                      patient_strat=False,
                                      ignore=[])
    else:
        raise NotImplementedError

    # =============== write args ... ===============
    current_time = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
    
    # results_dir
    args.results_dir = os.path.join(args.results_dir, '{}'.format(args.task) + '_{}_'.format(args.model_type) + str(args.exp_code))
    if args.dnc:
        args.results_dir += '_DNC'
    args.results_dir += '_' + current_time
    print('results_dir: ', args.results_dir)
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)
        
    # split_dir
    if args.split_dir is None:
        args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
    print('split_dir: ', args.split_dir)
    assert os.path.isdir(args.split_dir)
    settings.update({'split_dir': args.split_dir})

    # write txt
    with open(args.results_dir + '/experiment.txt', 'w') as f:
        print(settings, file=f)      

    # set auto resume 
    if args.k_start == -1:
        folds = args.k if args.k_end == -1 else args.k_end
        for i in range(folds):
            filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
            if not os.path.exists(filename):
                args.k_start = i
                break
        print('Training from fold: {}'.format(args.k_start))
    
    #Create directory to save logits
    if args.save_logits:
        logits_dir = os.path.join('logits', args.task, args.backbone, args.model_type, args.exp_code)
        if not os.path.exists(logits_dir):
            os.makedirs(logits_dir)
        args.logits_dir = logits_dir
        
    # =============== run main ... ===============
    main(args)
