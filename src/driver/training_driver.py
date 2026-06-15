# this driver will train the models 

# analogous to operational_mse_crps_driver.py from uq4ml_watertemp'

"""

The purpose of this script is to train a drought classification model. 

"""

from pathlib import Path
from datetime import datetime

import keras
import joblib 
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from keras.callbacks  import TensorBoard, EarlyStopping
from keras.layers import Input, Dense
from keras.models import Sequential 

from src.helper.training_logger import TrainingLogger
from src.data_prep.preparing_data import preparingData
from src.helper.parser import create_parser

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

def train_models(args):
    
    if args.start_iteration > args.end_iteration:
        up_down = -1
    else:
        up_down = 1
        args.end_iteration += 1

    prediction_column_names = []
    for k in range(args.num_output_neurons):
        prediction_column_names.append(f'pred_{k+1}')   

    print("\n\n----------------------------- TRAINING ! -----------------------------\n\n")

    for iteration in range(args.start_iteration, args.end_iteration, up_down):  
            combo_name = f"{args.model_type.lower()}-{args.num_layers}_layers-{args.activation_function}-{args.neurons}_neurons"

            for rotation in args.rotation_list:
                print(f"RUNNING {args.leadtime}h, {combo_name}-cycle_{rotation}-iteration_{iteration} ...\n")
                
                cycle_time_start = datetime.now()

                data_prep_time_start = datetime.now()

                # clearing stale nodes that might be persisting in the 
                # computation graph, causing corruption; throwing an error
                K.clear_session()

                """ Manipulating data for AI Model """
                
                preparingData_result = preparingData(
                    args.dataset,
                    args.input_structure,
                    args.independent_year,
                    args.leadtime,
                    args.atp_hours_back,
                    args.wtp_hours_back,
                    args.pred_atp_interval,
                    IPPOffset=args.temperature_list,
                    cycle=rotation,
                    model=args.model_type,
                    scale=args.scale
                )
                
                if args.scale:
                    x_train, y_train, x_val, y_val, x_test, y_test, training_dates, validation_dates, testingDates, testingAir, scaler = preparingData_result
                else:
                    x_train, y_train, x_val, y_val, x_test, y_test, training_dates, validation_dates, testingDates, testingAir = preparingData_result
                    scaler = None
                
                """PREPARINGDATA FUNCTION COMPUTE TIME"""
                data_prep_time_end = datetime.now()

                # Path to folder for visualization results
                save_path = Path(args.results_folder) / f"{args.leadtime}h" / f"{combo_name}-rotation_{rotation}-iteration_{iteration}"
                save_path.mkdir(parents=True, exist_ok=True)

                with open(save_path / "data_prep_compute_time.txt", 'w') as compute_time_file:
                    compute_time_file.write(f"preparingData() compute time: {data_prep_time_end - data_prep_time_start}")

                train_time_start = datetime.now()

                inputShape = x_train[0].shape
                
                if args.batch_size == -1:
                    batch_size = x_train.shape[0]

                else:
                    batch_size = args.batch_size

                """TRAINING THE MODEL"""

                model = Sequential()                

                # first layer = input layer
                model.add(Input(shape=(inputShape)))

                # hidden layer(s)
                for _ in range(args.num_layers):
                    model.add(Dense(units=args.neurons, 
                                    activation=args.activation_function, 
                                    kernel_regularizer=args.kernel_regularizer))
                
                # last layer = output layer
                model.add(Dense(args.num_output_neurons, activation=args.output_activation))

                model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=args.lrate), 
                            loss=args.loss_function, metrics=args.metrics)

                logger = TrainingLogger(save_path / "std_output.txt")

                # Learning rate reducer
                reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor=args.call_back_monitor, min_delta=args.min_delta,
                                                                factor=args.factor, patience=args.lr_reducer_patience, min_lr=args.min_lr)
                
                # Defining the early stopping
                early_stopping = EarlyStopping(monitor=args.call_back_monitor,
                                                    min_delta=args.min_delta,
                                                    patience=args.early_stop_patience,
                                                    verbose=args.verbose,
                                                    mode='auto',
                                                    restore_best_weights=True)
                
                
                log_dir = save_path / "tensorboard_logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

                model_callbacks = [early_stopping, reduce_lr, tensorboard_callback, logger]

                # Training the model
                history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=args.epochs, 
                                    batch_size=batch_size, callbacks=model_callbacks, verbose=args.verbose) 
                
                """TRAINING COMPUTE TIME"""
                train_time_end = datetime.now()

                # look into re-loading models instead of re-training them again 
                # use pickle
                # need to do this for re-producing the bug 

                with open(save_path / "train_compute_time.txt", 'w') as compute_time_file:
                    compute_time_file.write(f"Model Train Time: {train_time_end-train_time_start}")
                
                """LOSS INFORMATION"""
                loss = history.history['loss']
                val_loss = history.history['val_loss']
            
                losses = pd.DataFrame(columns=['Loss', 'Val_Loss']) 
                losses['Loss'] = loss
                losses['Val_Loss'] = val_loss
                losses.to_csv(save_path / "losses.csv")

                # saving the model to keras file
                model.save(save_path / f"model_{datetime.now().strftime('%Y%m%d-%H%M%S')}_.keras") 
                
                if scaler is not None:
                    scaler_filename = f"scaler_{args.model_type}_{args.leadtime}h_cycle{rotation}_iter{iteration}.joblib"
                    joblib.dump(scaler, save_path / scaler_filename)

                train_predictions = model.predict(x_train)
                val_predictions = model.predict(x_val)
                # put print here; focus on those 3 days during the cs event of 2021 to ease analysis
                test_predictions = model.predict(x_test)

                """SAVING PREDICTIONS AND OBSERVATIONS"""
                train_vs_preds = pd.DataFrame(columns=prediction_column_names, data=train_predictions)
                val_vs_preds = pd.DataFrame(columns=prediction_column_names, data=val_predictions)
                test_vs_preds = pd.DataFrame(columns=prediction_column_names, data=test_predictions)
                
                train_vs_preds.insert(loc=0, column='date_time', value=training_dates)
                val_vs_preds.insert(loc=0, column='date_time', value=validation_dates)
                test_vs_preds.insert(loc=0, column='date_time', value=testingDates)

                train_vs_preds.insert(loc=1, column='target', value=y_train)
                val_vs_preds.insert(loc=1, column='target', value=y_val)
                test_vs_preds.insert(loc=1, column='target', value=y_test)
                
                """TOTAL CYCLE MODEL COMPUTE TIME"""
                cycle_time_end = datetime.now()
                train_path = save_path / "train_datetime_obsv_predictions.csv"
                val_path = save_path / "val_datetime_obsv_predictions.csv"

                if args.independent_year != "cycle":
                    test_path = save_path / f"{args.independent_year}_datetime_obsv_predictions.csv"
                else:
                    test_path = save_path / "test_datetime_obsv_predictions.csv"

                train_vs_preds.to_csv(train_path)
                val_vs_preds.to_csv(val_path)
                test_vs_preds.to_csv(test_path)
    
                with open(save_path / "cycle_compute_time.txt", 'w') as compute_time_file:
                    compute_time_file.write(f"Total Cycle Time: {cycle_time_end-cycle_time_start}")

def main():
    print("yippee!")
    pass

def show_keys(args):
    args_dict = vars(args)
    for key, value in args_dict.items():
        print(f"{key}: {value}")
