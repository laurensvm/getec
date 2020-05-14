import os
import numpy as np
import matplotlib.pyplot as plt


class Visualizer(object):

    def __init__(self, figure_directory):
        self.figure_directory = figure_directory

    def _get_filename(self, name):
        import uuid
        uid = "_" + str(uuid.uuid4())
        return os.path.join(self.figure_directory, name + uid)

    def plot_history(self, history, model):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        if model.uid:
            fig.suptitle(f"Performance of the {model.__class__.__name__} ({model.uid})")
        else:
            fig.suptitle(f"Performance of the {model.__class__.__name__}")

        ax1.set_title("Losses")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.plot(history["loss"], label="Train Loss", linewidth=0.5)
        ax1.plot(history["val_loss"], label="Validation Loss", linewidth=0.5)
        ax1.legend(loc="upper left")

        ax2.set_title("Accuracy")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Accuracy")
        ax2.plot(history["accuracy"], label="Training Accuracy", linewidth=0.5)
        ax2.plot(history["val_accuracy"], label="Validation Accuracy", linewidth=0.5)
        ax2.legend(loc="upper left")

        plt.savefig(self._get_filename(model.__class__.__name__), format='eps')
        plt.show()

    def plot_models_performances(self, data):
        accs = np.zeros((5, 4))
        models = []

        for i, d in enumerate(data):
            acc, model = zip(*d)

            models.append(model[0].__class__.__name__)

            for j, accuracy in enumerate(acc):
                accs[j, i] = accuracy[1]

        # models = np.array(models)
        # models = np.transpose(models)

        x = np.arange(len(accs[0]))

        fig, ax = plt.subplots()

        ax.set_title("Test Set Performance Per Model")
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylabel("Accuracy")
        ax.set_yticks(np.linspace(0, 1, 11))

        ax.bar(x - 0.30, accs[0], color='#80c904', width=0.15, label="1 Sec")
        ax.bar(x - 0.15, accs[1], color='#73b504', width=0.15, label="1.5 Sec")
        ax.bar(x, accs[2], color='#66a103', width=0.15, label="2 Sec")
        ax.bar(x + 0.15, accs[3], color='#5a8d03', width=0.15, label="2.5 Sec")
        ax.bar(x + 0.30, accs[4], color='#4d7902', width=0.15, label="3 Sec")

        ax.legend()

        fig.tight_layout()

        plt.savefig(self._get_filename("model_performances"), format='eps')

        plt.show()
