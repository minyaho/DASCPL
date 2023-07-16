from model.vision_multi import VGG_DASCPL_m, resnet18_DASCPL_m
from utils import CPUThread

def _parser_trigger_stages(trigger_epochs, epoch_step, num_layers, default_epochs=[50, 100, 150]):
    if trigger_epochs != None:
        print("[Model INFO] Trigger epochs: {}".format(trigger_epochs))
        epochs = trigger_epochs.replace(' ', '').split(",")
        if len(epochs) == (num_layers-1):
            trigger_stages = [int(e)*epoch_step for e in epochs]
        else:
            raise ValueError("epochs != num_layers-1 ({} != {})".format(len(epochs), num_layers))
        return trigger_stages
    else:
        print("[Model INFO] Trigger epochs: {}".format(default_epochs))
        return [int(e)*epoch_step for e in default_epochs]

class VGG_DASCPL_da_m(VGG_DASCPL_m):
    def __init__(self, configs):
        super(VGG_DASCPL_da_m, self).__init__(configs)

    def _init_data(self, configs):
        super()._init_data(configs)
        self.epoch_step = len(self.train_loader)
        self.trigger_step = _parser_trigger_stages(configs['trigger_epochs'], self.epoch_step, self.num_layers)

    def train_step(self, X, Y, multi_t=True):
        self.global_steps += 1
        tasks = list()
        gpu_losses = list()
        loss_all = 0

        for layer in self.model:
            layer.train()
    
        true_y0 = Y.to(self.devices[0], non_blocking=True)
        true_y1 = Y.to(self.devices[1])
        true_y2 = Y.to(self.devices[2])
        true_y3 = Y.to(self.devices[3])

        for opt in self.opts:
            opt.zero_grad()

        # Forward: Block 0 ~ 3
        ## Forward: Block 0
        x0 = X.to(self.devices[0], non_blocking=True)
        hat_y0 = self.model[0](x0)       # Block 0
        x1 = hat_y0.detach().to(self.devices[1])
    
        ## Loss,Backward,Update: Block 0
        args = ([self.model[0]], self.opts[0].optimizer, [hat_y0], [true_y0])
        if not multi_t:
            gpu_losses.append(self._loss_backward_update(*args)) # Block 0

        else:
            tasks.append(CPUThread(target=self._loss_backward_update, args=args))
            tasks[-1].start()
        
        if self.global_steps > self.trigger_step[0]:
            ## Forward: Block 1
            hat_y1 = self.model[1](x1)       # Block 1 
            x2 = hat_y1.detach().to(self.devices[2])
        
            ## Loss,Backward,Update: Block 1
            args = ([self.model[1]], self.opts[1].optimizer, [hat_y1], [true_y1])
            if not multi_t:
                gpu_losses.append(self._loss_backward_update(*args)) # Block 1

            else:
                tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                tasks[-1].start()

        if self.global_steps > self.trigger_step[1]:
            ## Forward: Block 2
            hat_y2 = self.model[2](x2)       # Block 2
            x3 = hat_y2.detach().to(self.devices[3])
        
            ## Loss,Backward,Update: Block 2
            args = ([self.model[2]], self.opts[2].optimizer, [hat_y2], [true_y2])
            if not multi_t:
                gpu_losses.append(self._loss_backward_update(*args)) # Block 2

            else:
                tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                tasks[-1].start()

        if self.global_steps > self.trigger_step[2]:

            ## Forward: Block 3
            hat_y3 = self.model[3](x3)       # Block 3
            
            x4 = hat_y3.detach()
            hat_y4 = self.model[4](x4)       # Block 3
        
            ## Loss,Backward,Update: Block 3
            args = ([self.model[3], self.model[4]], self.opts[3].optimizer, [hat_y3, hat_y4], [true_y3]*2)
            if not multi_t:
                gpu_losses.append(self._loss_backward_update(*args)) # Block 3

            else:
                tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                tasks[-1].start()

        if multi_t:
            for t in range(len(tasks)):
                loss_all += tasks[t].get_result().item()
        else:
            for loss in gpu_losses:
                loss_all += loss.item()
            
        return None, loss_all

class VGG_DASCPL_dr_m(VGG_DASCPL_m):
    def __init__(self, configs):
        super(VGG_DASCPL_dr_m, self).__init__(configs)

    def _init_data(self, configs):
        super()._init_data(configs)
        self.epoch_step = len(self.train_loader)
        self.trigger_step = _parser_trigger_stages(configs['trigger_epochs'], self.epoch_step, self.num_layers)

    def train_step(self, X, Y, multi_t=True):
        self.global_steps += 1
        tasks = list()
        gpu_losses = list()
        loss_all = 0

        for layer in self.model:
            layer.train()
    
        true_y0 = Y.to(self.devices[0], non_blocking=True)
        true_y1 = Y.to(self.devices[1])
        true_y2 = Y.to(self.devices[2])
        true_y3 = Y.to(self.devices[3])

        for opt in self.opts:
            opt.zero_grad()

        # Forward: Block 0 ~ 3
        ## Forward: Block 0
        x0 = X.to(self.devices[0], non_blocking=True)
        hat_y0 = self.model[0](x0)       # Block 0
        x1 = hat_y0.detach().to(self.devices[1])
    
        ## Loss,Backward,Update: Block 0
        args = ([self.model[0]], self.opts[0].optimizer, [hat_y0], [true_y0])
        if not multi_t:
            gpu_losses.append(self._loss_backward_update(*args)) # Block 0

        else:
            tasks.append(CPUThread(target=self._loss_backward_update, args=args))
            tasks[-1].start()
        
        if self.global_steps <= self.trigger_step[2]:
            ## Forward: Block 1
            hat_y1 = self.model[1](x1)       # Block 1 
            x2 = hat_y1.detach().to(self.devices[2])
        
            ## Loss,Backward,Update: Block 1
            args = ([self.model[1]], self.opts[1].optimizer, [hat_y1], [true_y1])
            if not multi_t:
                gpu_losses.append(self._loss_backward_update(*args)) # Block 1

            else:
                tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                tasks[-1].start()

        if self.global_steps <= self.trigger_step[1]:
            ## Forward: Block 2
            hat_y2 = self.model[2](x2)       # Block 2
            x3 = hat_y2.detach().to(self.devices[3])
        
            ## Loss,Backward,Update: Block 2
            args = ([self.model[2]], self.opts[2].optimizer, [hat_y2], [true_y2])
            if not multi_t:
                gpu_losses.append(self._loss_backward_update(*args)) # Block 2

            else:
                tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                tasks[-1].start()

        if self.global_steps <= self.trigger_step[0]:

            ## Forward: Block 3
            hat_y3 = self.model[3](x3)       # Block 3
            
            x4 = hat_y3.detach()
            hat_y4 = self.model[4](x4)       # Block 3
        
            ## Loss,Backward,Update: Block 3
            args = ([self.model[3], self.model[4]], self.opts[3].optimizer, [hat_y3, hat_y4], [true_y3]*2)
            if not multi_t:
                gpu_losses.append(self._loss_backward_update(*args)) # Block 3

            else:
                tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                tasks[-1].start()

        if multi_t:
            for t in range(len(tasks)):
                loss_all += tasks[t].get_result().item()
        else:
            for loss in gpu_losses:
                loss_all += loss.item()
            
        return None, loss_all

class resnet18_DASCPL_da_m(resnet18_DASCPL_m):
    def __init__(self, configs):
        super(resnet18_DASCPL_da_m, self).__init__(configs)

    def _init_data(self, configs):
        super()._init_data(configs)
        self.epoch_step = len(self.train_loader)
        self.trigger_step = _parser_trigger_stages(configs['trigger_epochs'], self.epoch_step, self.num_layers)

    def train_step(self, X, Y, multi_t=True):
        self.global_steps += 1
        tasks = list()
        gpu_losses = list()
        loss_all = 0

        true_y0 = Y.to(self.devices[0], non_blocking=True)
        true_y1 = Y.to(self.devices[1])
        true_y2 = Y.to(self.devices[2])
        true_y3 = Y.to(self.devices[3])

        for layer in self.model:
            layer.train()

        for opt in self.opts:
            opt.zero_grad()

        # Forward: Block 0 ~ 3
        ## Forward: Block 0
        x0 = X.to(self.devices[0], non_blocking=True)
        x1 = self.model[0](x0)       # Block 0

        hat_y1 = self.model[1](x1)       # Block 0
        x2 = hat_y1.detach().to(self.devices[1])

        ## Loss,Backward,Update: Block 0
        args = ([self.model[1]], self.opts[0].optimizer, [hat_y1], [true_y0])
        if not multi_t:
            gpu_losses.append(self._loss_backward_update(*args)) # Block 0

        else:
            tasks.append(CPUThread(target=self._loss_backward_update, args=args))
            tasks[-1].start()
    
        if self.global_steps > self.trigger_step[0]:
            ## Forward: GPU 1
            hat_y2 = self.model[2](x2)       # Block 1 
            x3 = hat_y2.detach().to(self.devices[2])
            
            ## Loss,Backward,Update: Block 1
            args = ([self.model[2]], self.opts[1].optimizer, [hat_y2], [true_y1])
            if not multi_t:
                gpu_losses.append(self._loss_backward_update(*args)) # Block 1

            else:
                tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                tasks[-1].start()

        if self.global_steps > self.trigger_step[1]:
            ## Forward: Block 2
            hat_y3 = self.model[3](x3)       # Block 2
            x4 = hat_y3.detach().to(self.devices[3])
            
            ## Loss,Backward,Update: Block 2
            args = ([self.model[3]], self.opts[2].optimizer, [hat_y3], [true_y2])
            if not multi_t:
                gpu_losses.append(self._loss_backward_update(*args)) # Block 2

            else:
                tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                tasks[-1].start()

        if self.global_steps > self.trigger_step[2]:
            ## Forward: Block 3
            hat_y4 = self.model[4](x4)       # Block 3
            
            x5 = hat_y4.detach()
            hat_y5 = self.model[5](x5)       # Block 3

            ## Loss,Backward,Update: Block 3
            args = ([self.model[4], self.model[5]], self.opts[3].optimizer, [hat_y4, hat_y5], [true_y3]*2)
            if not multi_t:
                gpu_losses.append(self._loss_backward_update(*args)) # Block 3

            else:
                tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                tasks[-1].start()

        if multi_t:
            for t in range(len(tasks)):
                loss_all += tasks[t].get_result().item()
        else:
            for loss in gpu_losses:
                loss_all += loss.item()
    
        return None, loss_all
    
class resnet18_DASCPL_dr_m(resnet18_DASCPL_m):
    def __init__(self, configs):
        super(resnet18_DASCPL_dr_m, self).__init__(configs)

    def _init_data(self, configs):
        super()._init_data(configs)
        self.epoch_step = len(self.train_loader)
        self.trigger_step = _parser_trigger_stages(configs['trigger_epochs'], self.epoch_step, self.num_layers)
        
    def train_step(self, X, Y, multi_t=True):
        self.global_steps += 1
        tasks = list()
        gpu_losses = list()
        loss_all = 0

        true_y0 = Y.to(self.devices[0], non_blocking=True)
        true_y1 = Y.to(self.devices[1])
        true_y2 = Y.to(self.devices[2])
        true_y3 = Y.to(self.devices[3])

        for layer in self.model:
            layer.train()

        for opt in self.opts:
            opt.zero_grad()

        # Forward: Block 0 ~ 3
        ## Forward: Block 0
        x0 = X.to(self.devices[0], non_blocking=True)
        x1 = self.model[0](x0)       # Block 0

        hat_y1 = self.model[1](x1)       # Block 0
        x2 = hat_y1.detach().to(self.devices[1])

        ## Loss,Backward,Update: Block 0
        args = ([self.model[1]], self.opts[0].optimizer, [hat_y1], [true_y0])
        if not multi_t:
            gpu_losses.append(self._loss_backward_update(*args)) # Block 0

        else:
            tasks.append(CPUThread(target=self._loss_backward_update, args=args))
            tasks[-1].start()
    
        if self.global_steps <= self.trigger_step[2]:
            ## Forward: GPU 1
            hat_y2 = self.model[2](x2)       # Block 1 
            x3 = hat_y2.detach().to(self.devices[2])
            
            ## Loss,Backward,Update: Block 1
            args = ([self.model[2]], self.opts[1].optimizer, [hat_y2], [true_y1])
            if not multi_t:
                gpu_losses.append(self._loss_backward_update(*args)) # Block 1

            else:
                tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                tasks[-1].start()

        if self.global_steps <= self.trigger_step[1]:
            ## Forward: Block 2
            hat_y3 = self.model[3](x3)       # Block 2
            x4 = hat_y3.detach().to(self.devices[3])
            
            ## Loss,Backward,Update: Block 2
            args = ([self.model[3]], self.opts[2].optimizer, [hat_y3], [true_y2])
            if not multi_t:
                gpu_losses.append(self._loss_backward_update(*args)) # Block 2

            else:
                tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                tasks[-1].start()

        if self.global_steps <= self.trigger_step[0]:
            ## Forward: Block 3
            hat_y4 = self.model[4](x4)       # Block 3
            
            x5 = hat_y4.detach()
            hat_y5 = self.model[5](x5)       # Block 3

            ## Loss,Backward,Update: Block 3
            args = ([self.model[4], self.model[5]], self.opts[3].optimizer, [hat_y4, hat_y5], [true_y3]*2)
            if not multi_t:
                gpu_losses.append(self._loss_backward_update(*args)) # Block 3

            else:
                tasks.append(CPUThread(target=self._loss_backward_update, args=args))
                tasks[-1].start()

        if multi_t:
            for t in range(len(tasks)):
                loss_all += tasks[t].get_result().item()
        else:
            for loss in gpu_losses:
                loss_all += loss.item()
    
        return None, loss_all
