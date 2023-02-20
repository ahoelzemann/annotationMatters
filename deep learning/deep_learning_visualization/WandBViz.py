import wandb
import pickle
import os
import pandas as pd


def log_metrics(logs, epoch, fold):

    wandb.log({'train_loss_' + fold: logs['loss'], 'epoch': epoch})
    wandb.log({'train_acc_' + fold: logs['accuracy'], 'epoch': epoch})
    wandb.log({'train_f1_' + fold: logs['f1_m'], 'epoch': epoch})
    wandb.log({'train_precision_' + fold: logs['precision_m'], 'epoch': epoch})
    wandb.log({'train_recall_' + fold: logs['recall_m'], 'epoch': epoch})
    wandb.log({'val_acc_' + fold: logs['val_accuracy'], 'epoch': epoch})
    wandb.log({'val_f1_' + fold: logs['val_f1_m'], 'epoch': epoch})
    wandb.log({'val_precision_' + fold: logs['val_precision_m'], 'epoch': epoch})
    wandb.log({'val_recall_' + fold: logs['val_recall_m'], 'epoch': epoch})


def save_best_model(logs, epoch, fold, model):
    if epoch == 0:
        lastbestF1 = 0
        with open(os.path.join(wandb.run.dir, fold + "_bestf1.pkl"), 'wb') as fp:
            pickle.dump({"f1": lastbestF1, "epoch": epoch}, fp)
            print(f'Best F1 saved.')
    else:
        lastbestF1 = pd.read_pickle(os.path.join(wandb.run.dir, fold + "_bestf1.pkl"))['f1']
    if logs['val_f1_m'] > lastbestF1:
        with open(os.path.join(wandb.run.dir, fold + "_bestf1.pkl"), 'wb') as fp:
            pickle.dump({"f1": logs['val_f1_m'], "epoch": epoch}, fp)
            print(f'Best F1 saved.')
        model.save(os.path.join(wandb.run.dir, fold + "_bestModel.h5"))
        print(f'Best Model updated.')
