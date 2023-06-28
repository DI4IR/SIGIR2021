import torch
import os


class ModelCheckpoint:
    def __init__(self, model, optimizer, path, filename, every_n_train_steps):
        if not os.path.exists(path):
            os.mkdir(path)
        self.path = path
        self.filename = filename
        self.model = model
        self.optimizer = optimizer
        self.every_n_train_steps = every_n_train_steps
        self.step = 0

    def __call__(self):
        self.step += 1
        if self.step % self.every_n_train_steps == 0:
            model = self.model
            optimizer = self.optimizer
            filename = self.filename
            if hasattr(model, 'module'):
                model = model.module
            checkpoint = {}
            checkpoint['model_state_dict'] = model.state_dict()
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            filename = self.filename + "-step={" + str(self.step) + "}.ckpt"
            torch.save(checkpoint, os.path.join(self.path, filename))