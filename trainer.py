"""Train the model"""

import logging
import utils.file_utils
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

import model.net as net
import model.data_loader as data_loader
from utils.stats_utils import RunningAverage
from utils.file_utils import load_checkpoint, save_checkpoint, save_dict_to_json, BEST_DICT_JSON

class Trainer():

    def __init__(self, model, optimizer, loss_fn, final_test_loss_fn, train_dataloader, val_dataloader, test_dataloader):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.final_test_loss_fn = final_test_loss_fn
    
    def run_model(self, loss_fn, loader, metrics, params, training = False):
        if training:
            self.model.train()
        else:
            self.model.eval()

        summ = []
        loss_avg = RunningAverage()

        # Use tqdm for progress bar
        with tqdm(total=len(self.train_dataloader)) as t:
            for i, (input_batch, labels_batch) in enumerate(loader):
                # move to GPU if available
                if params.cuda:
                    input_batch, labels_batch = input_batch.cuda(
                        non_blocking=True), labels_batch.cuda(non_blocking=True)
                # convert to torch Variables
                input_batch, labels_batch = Variable(input_batch), Variable(labels_batch)

                # compute model output and loss
                output_batch = self.model(input_batch)
                loss = loss_fn(output_batch, labels_batch)
        
                if training:
                    # clear previous gradients, compute gradients of all variables wrt loss
                    self.optimizer.zero_grad()
                    loss.backward()

                    # performs updates using calculated gradients
                    self.optimizer.step()

                # Evaluate summaries only once in a while
                if (i % params.save_summary_steps == 0) or not training:
                    # extract data from torch Variable, move to cpu, convert to numpy arrays
                    output_batch = output_batch.data.cpu().numpy()
                    labels_batch = labels_batch.data.cpu().numpy()

                    # compute all metrics on this batch
                    summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                                    for metric in metrics}
                    summary_batch['loss'] = loss.item()
                    summ.append(summary_batch)

                    logging.info(f"Loss: {loss}")
                    logging.info(f"output_batch: {output_batch}")
                    logging.info(f"labels_batch: {labels_batch}")

                if training:
                    # update the average loss
                    loss_avg.update(loss.item())

                    t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
                    t.update()

        # compute mean of all metrics in summary
        metrics_mean = {metric: np.mean([x[metric]
                                        for x in summ]) for metric in summ[0]}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                    for k, v in metrics_mean.items())
        logging.info("- Train metrics: " + metrics_string)
        return metrics_mean
    
    
    def train_and_evaluate(self, metrics, params, model_dir, restore):
        """Train the model and evaluate every epoch.

        Args:
            model: (torch.nn.Module) the neural network
            train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
            val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
            optimizer: (torch.optim) optimizer for parameters of model
            loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
            metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
            params: (Params) hyperparameters
            model_dir: (string) directory containing config, weights and log
            restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
        """
        # reload weights from restore_file if specified
        if restore:
            logging.info("Restoring parameters from ")
            load_checkpoint(model = self.model, optimizer = self.optimizer)
        best_val_acc = 0.0

        for epoch in range(params.num_epochs):
            # Run one epoch
            logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

            # compute number of batches in one epoch (one full pass over the training set)
            _ = self.run_model(loss_fn = self.loss_fn, loader = self.train_dataloader, metrics = metrics, params = params, training = True)

            # Evaluate for one epoch on validation set
            val_metrics = self.run_model(loss_fn = self.loss_fn, loader = self.train_dataloader, metrics = metrics, params = params, training = False)

            val_acc = val_metrics['accuracy']
            is_best = val_acc >= best_val_acc

            # Save weights
            save_checkpoint(state = {'epoch': epoch + 1,
                                'state_dict': self.model.state_dict(),
                                'optim_dict': self.optimizer.state_dict()},
                                is_best=is_best,
                                model_dir=model_dir)

            # If best_eval, best_save_path
            if is_best:
                logging.info("- Found new best accuracy")
                best_val_acc = val_acc

                # Save best val metrics in a json file in the model directory
                save_dict_to_json(d = val_metrics, json_file_name = BEST_DICT_JSON)
                
            # Save latest val metrics in a json file in the model directory
            save_dict_to_json(d = val_metrics)

        #
        # Final Evaluation Test
        #
        test_metrics = self.run_model(loss_fn = self.final_test_loss_fn, loader = self.train_dataloader, metrics = metrics, params = params, training = False)
        logging.info(f"Test Metrics: {test_metrics}")



