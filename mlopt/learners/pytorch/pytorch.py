from mlopt.learners.learner import Learner
from mlopt.settings import N_BEST, FRAC_TRAIN, PYTORCH, \
        NET_TRAINING_PARAMS, DIVISION_TOL
import mlopt.learners.pytorch.utils as u
from mlopt.learners.pytorch.model import Net
from tqdm import tqdm
import os
import torch                                            # Basic utilities
import torch.nn as nn                                   # Neural network tools
import torch.optim as optim                             # Optimizer tools
import numpy as np
import logging


class PyTorchNeuralNet(Learner):
    """
    PyTorch Neural Network learner.
    """

    def __init__(self, **options):
        """
        Initialize PyTorch neural network class.

        Parameters
        ----------
        options : dict
            Learner options as a dictionary.
        """
        # Define learner name
        self.name = PYTORCH
        self.n_input = options.pop('n_input')
        self.n_classes = options.pop('n_classes')

        # Default params grid
        default_params = NET_TRAINING_PARAMS

        # Unpack settings
        self.options = {}
        self.options['params'] = options.pop('params', default_params)
        not_specified_params = [x for x in default_params.keys()
                                if x not in self.options['params'].keys()]
        # Assign remaining keys
        for p in not_specified_params:
            self.options['params'][p] = default_params[p]
        if 'n_hidden' not in self.options['params'].keys():
            self.options['params']['n_hidden'] = \
                [int((self.n_classes + self.n_input)/2)]

        # Get fraction between training and validation
        self.options['frac_train'] = options.pop('frac_train', FRAC_TRAIN)

        # Define device
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            logging.info("Using CUDA GPU %s with Pytorch" %
                         torch.cuda.get_device_name(self.device))
        else:
            self.device = torch.device("cpu")
            logging.info("Using CPU with Pytorch")

        # Pick minimum between n_best and n_classes
        self.options['n_best'] = min(options.pop('n_best', N_BEST),
                                     self.n_classes)

        # Define loss
        self.loss = nn.CrossEntropyLoss()

    def train_epoch(self, dataloader):

        # Initialize loss average and summary
        metrics = []
        loss_avg = u.RunningAverage()

        with tqdm(total=len(dataloader)) as t_epoch:
            for i, (inputs, labels) in enumerate(dataloader):

                # Move to CUDA if selected
                inputs, labels = \
                    inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()                   # zero grad
                outputs = self.net(inputs)                   # forward
                loss = self.loss(outputs, labels)            # loss
                loss.backward()                              # backward
                self.optimizer.step()                        # optimizer

                # Update average loss
                loss_avg.update(loss.item())
                t_epoch.set_postfix(loss='{:05.3f}'.format(loss_avg()))
                t_epoch.update()

                # Evaluate metrics only once in a while
                if i % 100 == 0:
                    metrics.append(u.eval_metrics(outputs,
                                                  labels,
                                                  loss))

        return u.log_metrics(metrics, string="Train")

    def evaluate(self, dataloader):

        metrics = []

        # set model to evaluation mode
        self.net.eval()

        with torch.no_grad():  # Disable gradients

            # compute metrics over the dataset
            for inputs, labels in dataloader:

                # Move to CUDA if selected
                inputs, labels = \
                    inputs.to(self.device), labels.to(self.device)

                # compute model output
                outputs = self.net(inputs)
                loss = self.loss(outputs, labels)

                metrics.append(u.eval_metrics(outputs, labels, loss))

        return u.log_metrics(metrics, string="Eval")

    def train_instance(self,
                       train_dl,
                       valid_dl,
                       params):
        """
        Train single instance of the network for parameters in params
        """

        # Create PyTorch Neural Network and port to to device
        self.net = Net(self.n_input,
                       self.n_classes,
                       params['n_hidden']).to(self.device)

        info_str = "Learning Neural Network with parameters: "
        info_str += str(params)
        logging.info(info_str)

        # Define optimizer
        self.optimizer = optim.Adam(self.net.parameters(),
                                    lr=params['learning_rate'])

        # Reset seed
        torch.manual_seed(1)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(1)

        # Set network in training mode (not evaluation)
        self.net.train()

        # TODO: Add best validaiton accuracy check and checkpoint
        # store/load
        # best_valid_accuracy = 0.0

        for epoch in range(params['n_epochs']):  # loop over dataset multiple times

            logging.info("Epoch {}/{}".format(epoch + 1, params['n_epochs']))

            train_metrics = self.train_epoch(train_dl)
            valid_metrics = self.evaluate(valid_dl)

            valid_accuracy = valid_metrics['accuracy']


            # is_best = valid_accuracy >= best_valid_accuracy
            # if is_best:
            #     logging.info("- Found new best accuracy")
            #     best_valid_accuracy = valid_accuracy

        return valid_accuracy

    def train(self, X, y):
        """
        Train model.

        Parameters
        ----------
        X : pandas DataFrame
            Features.
        y : numpy int array
            Labels.
        """

        self.n_train = len(X)

        # Convert X dataframe to numpy array
        # TODO: Move outside
        # X = pandas2array(X)

        # # Normalize data

        # Shuffle data, split in train and validation and create dataloader here
        np.random.seed(0)
        idx_pick = np.arange(self.n_train)
        np.random.shuffle(idx_pick)

        split_valid = int(self.options['frac_train'] * len(idx_pick))
        train_idx = idx_pick[:split_valid]
        valid_idx = idx_pick[split_valid:]
        X_train, y_train = X[train_idx], y[train_idx]
        X_valid, y_valid = X[valid_idx], y[valid_idx]

        # Create validation data loader
        # Training data loader will be created when evaluating the model
        # which depends on the batch_size variable
        valid_dl = u.get_dataloader(X_valid, y_valid)

        logging.info("Split dataset in %d training and %d validation" %
                     (len(train_idx), len(valid_idx)))

        # Create parameter vector
        params = [{
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'n_hidden': n_hidden}
            for learning_rate in self.options['params']['learning_rate']
            for batch_size in self.options['params']['batch_size']
            for n_epochs in self.options['params']['n_epochs']
            for n_hidden in self.options['params']['n_hidden']
        ]
        n_models = len(params)

        logging.info("Train Neural Network with %d " % n_models +
                     "sets of parameters, " +
                     "%d inputs, %d outputs" % (self.n_input, self.n_classes))

        # Create vector of results
        accuracy_vec = np.zeros(n_models)

        if n_models > 1:
            for i in range(n_models):

                # Create dataloader 
                train_dl = u.get_dataloader(X_train, y_train,
                                            batch_size=params[i]['batch_size'])

                accuracy_vec[i] = self.train_instance(train_dl, valid_dl,
                                                      params[i])

            # Pick best parameters
            self.best_params = params[np.argmax(accuracy_vec)]
            logging.info("Best parameters")
            logging.info(str(self.best_params))
            logging.info("Train neural network with best parameters")

        else:

            logging.info("Train neural network with "
                         "just one set of parameters")
            self.best_params = params[0]
            train_dl = \
                u.get_dataloader(X_train, y_train,
                                 batch_size=self.best_params['batch_size'])

        logging.info(self.best_params)
        # Retrain network with best parameters over whole dataset
        self.train_instance(train_dl, valid_dl, self.best_params)

    def predict(self, X, n_best=None):

        # Disable gradients computation
        n_best = n_best if (n_best is not None) else self.options['n_best']

        self.net.eval()  # Put layers in evaluation mode
        with torch.no_grad():

            # Convert pandas df to array (unroll tuples)
            X = torch.tensor(X, dtype=torch.float).to(self.device)

            # Evaluate classes
            # NB. Removed softmax (unscaled probabilities)
            y = self.net(X).detach().cpu().numpy()

        return self.pick_best_class(y, n_best=n_best)

    def save(self, file_name):
        # Save state dictionary to file
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        torch.save(self.net.state_dict(), file_name + ".pkl")

    def load(self, file_name):
        # Check if file name exists
        if not os.path.isfile(file_name + ".pkl"):
            err = "PyTorch pkl file does not exist."
            logging.error(err)
            raise ValueError(err)

        # Load state dictionary from file
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        self.net.load_state_dict(torch.load(file_name + ".pkl"))
        self.net.eval()  # Necessary to set the model to evaluation mode