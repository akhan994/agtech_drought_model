import argparse
from pathlib import Path

def create_parser():

    parser = argparse.ArgumentParser(description="ML Model Configuration", fromfile_prefix_chars='@')

    parser.add_argument('--verbose', '-v',                      action='count',     default=0,                              help="Verbosity level")
    parser.add_argument('--model_type',                         type=str,           default='mlp',                          help="Model Architecture type")
    
    # paths
    parser.add_argument('--results_path',                     type=Path,           default='results',                      help="Folder path to which to save the results to.")
    parser.add_argument('--keras_path',                     type=Path,           default='keras_files',                      help="Folder path to which to save the keras files to.")
    parser.add_argument('--dataset',                            type=Path,           default='whole',                        help="Which dataset should get loaded into the model?")
    parser.add_argument('--inference_csv',                      type=Path,        default=None,                              help="the csv to predict on")
    parser.add_argument('--viz_path',                     type=Path,           default='viz',                      help="Folder path to which to save visualizations to.")

    parser.add_argument('--rotation_list',           nargs='+', type=int,           default=[0],                            help="Number of rotations for k-fold cross validation")
    parser.add_argument('--leadtime',                           type=int,           default=4,                              help="Leadtime to forecast measured in weeks.")
    parser.add_argument('--input_window',                       type=int,           default=4,                              help="Size of input vector. Try 4, 12, 24, 52.")

    # training parameters
    parser.add_argument('--lrate',                              type=float,         default=0.0001,                         help="Learning rate")
    parser.add_argument('--epochs',                             type=int,           default=100,                            help="Number of training epochs")
    parser.add_argument('--metrics',                 nargs='+', type=str,           default=None,                            help="Metrics to evaluate the model with.")
    parser.add_argument('-l', '--loss_function',                type=str,           default='categorical_crossentropy',     help="Loss function")
    parser.add_argument('--hidden_activation',                type=str,           default='relu',                         help="Activation function")
    parser.add_argument('--batch_size',                         type=int,           default=32,                             help="Batch size")
    parser.add_argument('--min_delta',                          type=float,         default=0.01,                           help="Patience for Early Stopping")
    parser.add_argument('--lr_reducer_patience',                type=int,           default=15,                             help="Sets the patience parameter for the learning rate reducer callback")
    parser.add_argument('--early_stop_patience',                type=int,           default=45,                             help="Sets the patience parameter for the early stopping callback")
    parser.add_argument('--num_output_neurons',                 type=int,           default=6,                              help="Number of Neurons in Output Layer.")

    parser.add_argument('--call_back_monitor',                  type=str,           default="val_loss",                     help="this is for picking which metric is being checked during training. for callback options such as early stopping and learning rate reducer")
    parser.add_argument('--input_structure',                    type=str,           default="descending",                   help="placeholder text")
    parser.add_argument('--optimizer',                          type=str,           default="adam",                         help="placeholder text")
    parser.add_argument('--kernel_regularizer',                 type=str,           default="l2",                           help="placeholder text")
    parser.add_argument('--num_layers',                         type=int,           default=1,                              help="number of hidden layers")
    parser.add_argument('--neurons',                            type=int,           default=32,                             help="neurons per layer")
    
    parser.add_argument('--start_iteration',                    type=int,           default=1,                              help="starting training iteration")
    parser.add_argument('--end_iteration',                      type=int,           default=30,                             help="ending training iteration")

    parser.add_argument('--output_activation',                  type=str,           default="softmax",                      help="placeholder text")
    parser.add_argument('--factor',                             type=float,         default=0.1,                            help="placeholder text")
    parser.add_argument('--min_lr',                             type=float,         default=0.00001,                        help="placeholder text")
    parser.add_argument('--scale',                              action='store_true',                                        help="flag whether we want to scale the data using StandardScaler.")
    parser.add_argument('--class_names', nargs='+',             type=str,        default=None,                           help="list the class names for a classifier model")
    parser.add_argument('--feature_cols', nargs='+',             type=str,        default=None,                           help="list the feature cols you want in the model from the master dataframe that is made in preparing_data")
    return parser
