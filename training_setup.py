"""Train the model"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

from utils.logger_utils import set_logger
from utils.param_utils import Params

import model.net as net
import model.data_loader as data_loader
from evaluate import evaluate
from trainer import Trainer
from final_loss import final_loss_fn
from loss import loss_fn
from metrics import metrics
from evaluate import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/64x64_SIGNS',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore', default=False,
                    help="Restore last Parameters")  # 'best' or 'train'

if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(
        ['train', 'val', 'test'], args.data_dir, params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']
    test_dl = dataloaders['test']

    logging.info("- done.")

    # Define the model and optimizer
    model = net.Net(params).cuda() if params.cuda else net.Net(params)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    trainer = Trainer(model = model, optimizer = optimizer, loss_fn = loss_fn, final_test_loss_fn = final_loss_fn, train_dataloader = train_dl, val_dataloader = val_dl, test_dataloader = test_dl)
    trainer.train_and_evaluate(metrics = metrics, params = params, model_dir = args.model_dir, restore = args.restore)