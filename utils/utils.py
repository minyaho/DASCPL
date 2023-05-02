import torch
import torch.nn as nn
import random
import numpy as np
import math
import os
import json, datetime
import threading
import time
import sys

class SingleGPUModel(nn.Module):
    device_type = "single"

class MultiGPUModel(nn.Module):
    device_type = "multi"

    def opt_step(self, global_steps):
        for opt in self.opts:
            lr = opt.step(global_steps)
        return lr

    def _loss_backward_update(self, layer_model:list, optimizer, hat_y:list, true_y:list, diff_device=False, name=None):
        assert type(layer_model)==list, 'Arguments error! Your input (layer_model) must be a list.'
        assert type(true_y)==list, 'Arguments error! Your input (true_y) must be a list.'
        assert type(hat_y)==list, 'Arguments error! Your input (hat_y) must be a list.'

        assert len(layer_model) == len(hat_y), 'Arguments error! Your input (hat_y) must be equal length of layer_model.\
            But \"layer_model\" is {} and \"hat_y\" is {}'.format(len(layer_model), len(hat_y))
        assert len(layer_model) == len(true_y), 'Arguments error! Your input (true_y) must be equal length of layer_model.\
            But \"layer_model\" is {} and \"true_y\" is {}'.format(len(layer_model), len(true_y))

        if diff_device:
            loss = None
            for i, layer in enumerate(layer_model):
                _ = layer.loss(hat_y[i], true_y[i])
                if loss == None:
                    loss = _
                else:
                    loss += _.to(loss.device) if (_.device != loss.device) else _
        else:
            loss = 0
            for i, layer in enumerate(layer_model):
                loss += layer.loss(hat_y[i], true_y[i])
        
        loss.backward()
        optimizer.step()
                
        if name:
            print(name)

        return loss

    
class ProfilerMultiGPUModel(MultiGPUModel):
    def _loss_backward_update(self, layer_model:list, optimizer, hat_y:list, true_y:list, diff_device=False, name=None):
        assert type(layer_model)==list, 'Arguments error! Your input (layer_model) must be a list.'
        assert type(true_y)==list, 'Arguments error! Your input (true_y) must be a list.'
        assert type(hat_y)==list, 'Arguments error! Your input (hat_y) must be a list.'

        assert len(layer_model) == len(hat_y), 'Arguments error! Your input (hat_y) must be equal length of layer_model.\
            But \"layer_model\" is {} and \"hat_y\" is {}'.format(len(layer_model), len(hat_y))
        assert len(layer_model) == len(true_y), 'Arguments error! Your input (true_y) must be equal length of layer_model.\
            But \"layer_model\" is {} and \"true_y\" is {}'.format(len(layer_model), len(true_y))

        with torch.profiler.record_function("loss"):
            if diff_device:
                loss = None
                for i, layer in enumerate(layer_model):
                    _ = layer.loss(hat_y[i], true_y[i])
                    if loss == None:
                        loss = _
                    else:
                        loss += _.to(loss.device) if (_.device != loss.device) else _
            else:
                loss = 0
                for i, layer in enumerate(layer_model):
                    loss += layer.loss(hat_y[i], true_y[i])

        with torch.profiler.record_function("backward"):
            loss.backward()
        with torch.profiler.record_function("update"):
            optimizer.step()

        if name:
            print(name)

        return loss

class ALComponent(nn.Module):
    """
        Base class of a single associated learning block
        
        f: forward function
        g: autoencoder function
        b: bridge function
        inv: inverse function of autoencoder
        cb: creterion of bridge function
        ca: creterion of autoencoder
        cf: creterion of forward function
    """
    def __init__(
        self,
        f: nn.Module,
        g: nn.Module,
        b: nn.Module,
        inv: nn.Module,
        cf: nn.Module,
        cb: nn.Module,
        ca: nn.Module
    )->None:
        super(ALComponent, self).__init__()
        self.f = f
        self.g = g
        self.b = b
        self.inv = inv
        self.cb = cb
        self.ca = ca
        self.cf = cf
    
    def forward(self, x=None, y=None, label=None):
        if self.training:
            s = self.f(x)
            #loss_f = 100 * self.cf(s, label)
            loss_f = 0
            s0 = self.b(s)
            t = self.g(y)
            t0 = self.inv(t)
            
            loss_b = self.cb(s0, t.detach()) # contrastive loss
            loss_ae = self.ca(t0, y)
            return s.detach(), t.detach(), loss_f, loss_b, loss_ae
        else:
            if y == None:
                s = self.f(x)
                return s
            else:
                t0 = self.inv(y)
                return t0
        
    # for bridge block inference
    def bridge_forward(self, x):
        s = self.f(x)
        s0 = self.b(s)
        t0 = self.inv(s0)
        return t0

class ResultMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.container = list()

    def update_by_list(self, _list):

        for item in _list:
            self.container.append(item)
        
    def update(self, val, n=1):

        for i in range(n):
            self.container.append(val)

    @property
    def count(self):
        return len(self.container)
    
    @property
    def avg(self):
        return np.mean(self.container)

    @property
    def std(self):
        return np.std(self.container)
    
    @property
    def sum(self):
        return np.sum(self.container)
    
    @property
    def len(self):
        return len(self.container)
    
    @property
    def is_empty(self):
        if self.len == 0:
            return True
        else:
            return False

    def __repr__(self):
        return self.container.__repr__()

    def __str__(self):
        return self.container.__str__()
    
def accuracy(output, target):
    with torch.no_grad():
        bsz = target.shape[0]
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        acc = correct[0].view(-1).float().sum(0, keepdim=True).mul_(100 / bsz)
        return acc
    
def flat_accuracy(preds, labels, ignore_index=-1):
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    
    answer = list()
    for idx, label in enumerate(labels_flat):
        if(label != ignore_index):
            answer.append(pred_flat[idx]==label)

    if len(answer)==0: 
        return 0
    else:
        return (np.sum(answer) / len(answer)) * 100

def gpu_setting(gpu_list, layers_num):

    assert layers_num >= 2, "Model layer setting error! The number of layers must be greater than 2."

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

    gpus_old = gpu_list.split(',')
    gpus_num = len(set(gpus_old))
    gpus_new = [str(i) for i in range(gpus_num)]

    print("[GPU Setting] Set \"CUDA_VISIBLE_DEVICES\" in", [i for i in gpus_old])
    print("[GPU Setting] Renumbered GPU number as {} (old -> new)".format(
        ["{} -> {}".format(gpus_old[i], gpus_new[i]) for i in range(gpus_num)]))

    setting_list = ['' for i in range(layers_num)]

    if layers_num % gpus_num != 0:
        raise(ValueError("Layer number or GPU setting error"))
    else:
        if gpus_num == 1:
            setting_list = ['cuda:'+gpus_new[0] for i in range(layers_num)]
        elif layers_num % gpus_num == 0:
            for i in range(0, layers_num):
                setting_list[i] = 'cuda:'+gpus_new[i%gpus_num]
            
    print("[GPU Setting] Use list of GPUs:", setting_list)

    return setting_list

def adjust_learning_rate(optimizer, base_lr, end_lr, step, max_steps):
    q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
    lr = base_lr * q + end_lr * (1 - q)
    set_lr(optimizer, lr)
    return lr

def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr

class Optimizer():
    def __init__(self, model_param, base_lr, end_lr, max_step, optimizer_fn=torch.optim.Adam):
        self.base_lr = base_lr
        self.end_lr = end_lr
        self.max_step = max_step
        self.lr = base_lr
        self.optimizer = optimizer_fn(model_param, lr=self.base_lr)
    
    def step(self, step):
        q = 0.5 * (1 + math.cos(math.pi * step / self.max_step))
        self.lr = self.base_lr * q + self.end_lr * (1 - q)
        self._set_lr()
        return self.lr
    
    def zero_grad(self):
        self.optimizer.zero_grad()

    def _set_lr(self):
        for g in self.optimizer.param_groups:
            g['lr'] = self.lr

    def get_lr(self):
        return self.lr

class CPUThread(threading.Thread):
    def __init__(self, target=None, args=(), **kwargs):
        super(CPUThread, self).__init__()
        self._target = target
        self._args = args
        self._kwargs = kwargs

    def run(self):
        if self._target == None:
            return
        self.__result__ = self._target(*self._args, **self._kwargs)

    def get_result(self):
        self.join()
        return self.__result__

def setup_seed(seed):
    # https://pytorch.org/docs/stable/notes/randomness.html
    if int(seed) == -1:
        # torch.backends.cudnn.benchmark = True # Restore benchmark to improve performance
        return "Random"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False # Disable benchmark to ensure reproducibility
    return seed

class ModelResultRecorder(object):
    def __init__(self, model_name=None):
        assert model_name!=None, "name can not empty"

        self.model_name = model_name

        self.model_hpyerparameter = dict()
        self.model_result = dict()
        self.global_stats = dict()  
        self.model_train_history = dict()
        self._train_acc = np.array([])
        self._test_acc = np.array([])

        self._field_string_init()

    def _field_string_init(self):
        self.name = 'model name'
        self.hpyerparameter = "model hpyerparameter"
        self.times_history = "times history"
        self.times = 'times'
        self.best_test_acc = 'best test acc.'
        self.best_test_epoch = 'best epoch'
        self.epoch ='epoch'
        self.epoch_train_time = 'epoch train time'
        self.epoch_train_eval_time = 'epoch_train eval time'
        self.epoch_test_time = 'epoch test time'
        self.epoch_time_mean = 'epoch time mean'
        self.epoch_time_std = 'epoch time std'
        self.runtime = 'runtime'
        self.train_history = 'train_history'
        self.mean = 'mean'
        self.std = 'std'
        self.train_acc = 'train acc'
        self.train_loss = 'train loss'
        self.train_time = 'train time'
        self.train_eval_time = 'train eval. time'
        self.test_acc = 'test acc'
        self.test_time = 'test time'
        self.gpus_info = 'gpus info'
        self.gpus_ram = 'gpus ram'
        self.stats = "stats"
        self.epoch_train_acc = "epoch train acc"
        self.epoch_test_acc = "epoch test acc"
        self.times_train_acc = "times train acc"
        self.times_test_acc = "times test acc"
        self.max = "max"
        self.pos = "pos" # position
        self.raw_data = "raw_data"

    def add(self, times, best_test_acc, best_test_epoch, 
            epoch_train_time, epoch_train_eval_time,
            epoch_test_time, runtime, gpus_info=None):
        assert type(times) == int, "Record data type error"

        self.model_result[times] = {
            self.times: times,
            self.best_test_acc: best_test_acc,
            self.best_test_epoch: best_test_epoch,
            self.epoch_train_time: epoch_train_time,
            self.epoch_train_eval_time: epoch_train_eval_time,
            self.epoch_test_time: epoch_test_time,
            self.runtime: runtime,
            self.gpus_info: gpus_info
        }

    def save_epoch_info(self, t, e, tr_acc, tr_loss, tr_t, te_acc, te_t, tr_ev_t=None):
        if t not in self.model_train_history.keys():
            self.model_train_history[t] = dict()

        self.model_train_history[t][e] = {
            self.times: t,
            self.epoch: e,
            self.train_acc: tr_acc,
            self.train_loss: tr_loss,
            self.train_time: tr_t,
            self.train_eval_time: tr_ev_t,
            self.test_acc: te_acc,
            self.test_time: te_t,
        }

    def get_ResultMeter(self, resultMeter:ResultMeter, mode="avg"):
        if resultMeter != None:
            if mode == "avg":
                return resultMeter.avg
            else:
                return resultMeter.std
        else:
            return None

    def save_mean_std_config(self, best_test_accs:ResultMeter, best_test_epochs:ResultMeter, 
                             epoch_train_times:ResultMeter, epoch_train_eval_times:ResultMeter, 
                             run_times:ResultMeter, gpu_ram:ResultMeter=None,
                             config:dict=dict()):

        self._make_times_stats()

        self.global_stats = {
            self.best_test_acc + ' ' + self.mean: self.get_ResultMeter(best_test_accs, 'avg'),
            self.best_test_acc + ' ' + self.std: self.get_ResultMeter(best_test_accs, 'std'),
            self.best_test_epoch + ' ' + self.mean: self.get_ResultMeter(best_test_epochs, 'avg'),
            self.best_test_epoch + ' ' + self.std: self.get_ResultMeter(best_test_epochs, 'std'),
            self.epoch_train_time + ' ' + self.mean: self.get_ResultMeter(epoch_train_times, 'avg'),
            self.epoch_train_time + ' ' + self.std: self.get_ResultMeter(epoch_train_times, 'std'),
            self.epoch_train_eval_time + ' ' + self.mean: self.get_ResultMeter(epoch_train_eval_times, 'avg'),
            self.epoch_train_eval_time + ' ' + self.std: self.get_ResultMeter(epoch_train_eval_times, 'std'),
            self.runtime + ' ' + self.mean: self.get_ResultMeter(run_times, 'avg'),
            self.runtime + ' ' + self.std: self.get_ResultMeter(run_times, 'std'),
            self.gpus_ram + ' ' + self.mean: self.get_ResultMeter(gpu_ram, 'avg'),
            self.gpus_ram + ' ' + self.std: self.get_ResultMeter(gpu_ram, 'std'),
        }
        self.model_hpyerparameter = config

    def _make_times_stats(self):
        _times_key = [key for key in self.model_train_history.keys()]
        _epoch_key = [key for key in self.model_train_history[_times_key[0]].keys()]
        _acc_key = [key for key in self.model_train_history[_times_key[0]][_epoch_key[0]][self.train_acc]]

        # print("_times_key: {}".format(_times_key))
        # print("_epoch_key: {}".format(_epoch_key))
        # print("_acc_key: {}".format(_acc_key))

        num_times = len(_times_key)
        num_epoch = len(_epoch_key)
        num_acc = len(_acc_key)

        # print("num_times: {}, num_epoch: {}, num_acc: {}".format(num_times, num_epoch, num_acc))

        _train_acc = np.zeros((num_times, num_epoch, num_acc))
        _test_acc = np.zeros((num_times, num_epoch, num_acc))

        for idx_times in range(num_times):
            _time_history = self.model_train_history[idx_times]
            for idx_epoch in range(1, num_epoch+1):
                _epoch_history = _time_history[idx_epoch]
                for idx_acc in range(num_acc):
                    _train_acc[idx_times][idx_epoch-1][idx_acc] = _epoch_history[self.train_acc][idx_acc]
                    _test_acc[idx_times][idx_epoch-1][idx_acc] = _epoch_history[self.test_acc][idx_acc]
        
        self.time_stats = {
            self.times_train_acc + ' ' + self.mean: np.mean(_train_acc, axis=0).tolist(),
            self.times_train_acc + ' ' + self.std: np.std(_train_acc, axis=0).tolist(),
            self.times_test_acc + ' ' + self.mean: np.mean(_test_acc, axis=0).tolist(),
            self.times_test_acc + ' ' + self.std: np.std(_test_acc, axis=0).tolist(),
            self.times_train_acc + ' ' + self.max: _train_acc.max(1).tolist(),
            self.times_train_acc + ' ' + self.pos: _train_acc.argmax(1).tolist(),
            self.times_test_acc + ' ' + self.max: _test_acc.max(1).tolist(),
            self.times_test_acc + ' ' + self.pos: _test_acc.argmax(1).tolist(),
            self.times_test_acc + ' ' + self.max + ' ' + self.mean: _test_acc.max(1).mean(0).tolist(),
            self.times_test_acc + ' ' + self.max + ' ' + self.std: _test_acc.max(1).std(0).tolist(),
        }

        self._train_acc = _train_acc
        self._test_acc = _test_acc

    def _make_result(self):
        result = {
            self.name: self.model_name,
            self.hpyerparameter: self.model_hpyerparameter,
            'statistics': self.global_stats,
            self.times_history: self.model_result,
            self.train_history: self.model_train_history,
            self.raw_data: {
                self.times + ' ' + self.stats: self.time_stats,
                "train": None if self._train_acc.shape[0] == 0 else self._train_acc.tolist(),
                "test": None if self._test_acc.shape[0] == 0 else self._test_acc.tolist(),
            }
        }
        return result

    def save(self, filename):

        now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        path = filename + '_d(' + now + ').json'

        with open(path, 'w', encoding='utf-8') as json_f:
            json_f.write(json.dumps(self._make_result(), indent = 4)) 
        
        print("[INFO] Save results, file name: {}".format(path))

    def __repr__(self):
        return self._make_result()

    def __str__(self):
        return str(self._make_result())

def tb_record_gradient(model, writer, epoch):
    if writer == None: return
    for name, module in model.named_children():
        if module.__class__.__name__ in [nn.CrossEntropyLoss.__name__, nn.AdaptiveAvgPool2d.__name__]: break
        grad_norms = list()
        for p in module.parameters():
            if p.grad != None:
                grad_norms.append(torch.norm(p.grad.detach(), 2))
        norm = torch.norm(torch.stack(grad_norms), 2) if len(grad_norms) != 0 else 0
        writer.add_scalar(f'gradient_info/layer-{name}', norm, epoch)

class SynchronizeTimer(object):
    def __init__(self):
        self.enter_time = None
        self.exit_time = None
        self.runtime = None
        
    def start(self):
        torch.cuda.synchronize()
        self.enter_time = time.time()
        
    def end(self):
        torch.cuda.synchronize()
        self.exit_time = time.time()
        self.runtime = self.exit_time - self.enter_time
        
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.end()

def get_gpu_info(gpu_id):
    meter_size = 10e5
    gpu_info = torch.cuda.get_device_properties(gpu_id)
    gpu_name = gpu_info.name
    gpu_cuda_ver = "{}.{}".format(gpu_info.major,gpu_info.minor)# CUDA Capability Major/Minor version number
    total_m = gpu_info.total_memory
    reserved_m = torch.cuda.memory_reserved(gpu_id)
    allocated_m = torch.cuda.memory_allocated(gpu_id)
    free_m = reserved_m-allocated_m  # free inside reserved
    # print('[GPU Info] ID {} \t name {} \t cuda ver. {} \t total {} \t reserved {} \t allocated {} \t free {}\t'.format(
    #     gpu_id, gpu_name, gpu_cuda_ver, total_m/meter_size, reserved_m/meter_size, allocated_m/meter_size, free_m/meter_size))
    
    return {
        "id": gpu_id,
        "name": gpu_name,
        "cuda": gpu_cuda_ver,
        "total": total_m,
        "reserved": reserved_m,
        "allocated": allocated_m,
        "free": free_m
    }

def calculate_GPUs_usage(gpu_ids:list()):
    total_all_m = 0
    reserved_all_m = 0
    gpus_info = list()

    for idx in range(len(set(gpu_ids))):
        gpu_info = get_gpu_info(gpu_ids[idx])
        total_all_m = gpu_info['total'] + total_all_m
        reserved_all_m = gpu_info['reserved'] + reserved_all_m
        gpus_info.append(gpu_info)

    return {"total_all_m":total_all_m, "reserved_all_m":reserved_all_m, "gpus_info": gpus_info}

class StdoutWithLogger(object):
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)
 
    def write(self, message):
        self.log.write(message)
        self.terminal.write(message)
        self.log.flush()    # flush to the log (realtime and update to context)

    def flush(self):
        self.terminal.flush()

    def getOldStdout(self):
        return self.terminal

def check_config(configs):
    assert configs['model'] != None, "You have not selected a model!"
    assert configs['dataset']!= None, "You have not selected a dataset!"

def model_save(times, configs, model, save_path, optimizer=None):
    if optimizer == None:
        optimizer = list()
        for opt in model.opts:
            optimizer.append(opt.state_dict())

    state = {
        "configs": configs,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer != None else optimizer,
    }
    save_files = os.path.join(save_path, "ckpt_last_{0}.pth".format(times))
    torch.save(state, save_files)