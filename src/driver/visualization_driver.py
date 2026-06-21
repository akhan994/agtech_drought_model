import plotly.graph_objects as go
import plotly.express as px

def loss_curve():
    loss_fig = go.Figure()
    loss_fig.add_scatter(y=history.history["loss"], name="train loss")
    loss_fig.add_scatter(y=history.history["val_loss"], name="val loss")
    