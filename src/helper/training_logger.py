import tensorflow as tf

class TrainingLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_file):
        super().__init__()
        self.log_file = log_file
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        log_message = f"Epoch {epoch + 1}: " + ", ".join([f"{key}={value:.4f}" for key, value in logs.items()]) + "\n"
        
        # Print to console (optional)
        print(log_message, end="")
        
        # Append to log file
        with open(self.log_file, "a") as f:
            f.write(log_message)
