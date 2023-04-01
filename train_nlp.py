import torch
import configparser
import argparse

import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = 1
import sys, time
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
from copy import deepcopy
from utils import ResultMeter, ModelResultRecorder, SynchronizeTimer, StdoutWithLogger     
from utils import adjust_learning_rate, accuracy, gpu_setting, tb_record_gradient, setup_seed, calculate_GPUs_usage

import argparse
from utils.nlp import get_data, get_word_vector
from model.nlp_single import LSTM_BP_3, LSTM_BP_4, LSTM_BP_d, LSTM_SCPL_3, LSTM_SCPL_4, LSTM_SCPL_5
from model.nlp_single import Trans_BP_3, Trans_BP_4, Trans_BP_d, Trans_SCPL_3, Trans_SCPL_4, Trans_SCPL_5
from model.nlp_multi import LSTM_SCPL_m_4, LSTM_SCPL_m_d, LSTM_BP_m_d, LSTM_BP_p_m_d, LSTM_DASCPL_m_d
from model.nlp_multi import Trans_SCPL_m_4, Trans_SCPL_m_d, Trans_BP_m_d, Trans_BP_p_m_d, Trans_DASCPL_m_d

def get_args():
    parser = argparse.ArgumentParser('NLP SCPL training')
    parser.add_argument('--times', type=int, help='word embedding dimension', default="1")
    parser.add_argument('--train_bsz', type=int, help='word embedding dimension', default=64)
    parser.add_argument('--test_bsz', type=int, help='word embedding dimension', default=64)
    parser.add_argument('--epochs', type=int, help='word embedding dimension', default=50)
    parser.add_argument('--base_lr', type=float, help='word embedding dimension', default=0.001)
    parser.add_argument('--end_lr', type=float, help='word embedding dimension', default=0.001)
    parser.add_argument('--dataset', type=str, help='word embedding dimension', default="imdb")
    parser.add_argument('--max_len', type=int, help='word embedding dimension', default="1000")
    parser.add_argument('--h_dim', type=int, help='word embedding dimension', default="300")
    parser.add_argument('--model', type=str, help='word embedding dimension', default="LSTM_SCPL_3")
    parser.add_argument('--gpus', type=str, help='word embedding dimension', default="0")
    parser.add_argument('--seed', type=int, help='word embedding dimension', default="0")
    parser.add_argument('--layers', type=int, help='word embedding dimension', default="4")
    parser.add_argument('--heads', type=int, help='word embedding dimension', default="6")
    parser.add_argument('--vocab_size', type=int, help='word embedding dimension', default="30000")
    parser.add_argument('--word_vec', type=str, help='word embedding dimension', default="glove")
    parser.add_argument('--emb_dim', type=int, help='word embedding dimension', default="300")
    parser.add_argument('--proj_type', type=str, help='the projective head type in contrastive loss. \"i\" is identity. \"l\" is linear. \"m\" is mlp.', default="i")
    parser.add_argument('--pred_type', type=str, help='the predictor type in predictor loss.', default=None)
    parser.add_argument('--save_path', type=str, help='save the result', default=None)
    parser.add_argument('--multi_t', type=str, help='save the result', default="t")
    parser.add_argument('--profile', type=str, help='save the result', default="f")

    args = parser.parse_args()

    return args

def read_config(args=None, path = "config.ini"):
    configs = dict()

    # model = 'LSTM_BP_m_d'
    # # LSTM_SCPL, LSTM, LSTM_SCPL_Parallel_3, LSTM_SCPL_3, Trans_SCPL, Trans, Trans_SCPL_3, Trans_SCPL_Parallel_3
    # # LSTM_BP_m_d, LSTM_SCPL_m_d, Trans_BP_m_d, Trans_SCPL_m_d
    # dataset = 'sst2'
    # # ag_news, sst2, dbpedia_14, imdb
    # train_bsz = 128
    # test_bsz = 128
    # epochs = 50
    # base_lr = 0.001
    # end_lr = 0.001
    # h_dim = 300
    # max_len = 40
    # seed = 0
    # layers = 4
    # gpus_id = "0"
    # proj_type = "i"
    # save_path = None
    # multi_thread = True

    # # Only Transformer
    # head = 6 # Only use in Transformer

    # # Please Don't change
    vocab_size = 30000 # Do not change
    word_vec = 'glove'
    emb_dim = 300
        
    # gpus_list = gpus.split(',')
    # gpus_num = len(gpus_list)

    # configs["gpu_ids"] = gpus_id
    # configs['gpus'] = gpu_setting(gpus_id, layers)
    
    # configs['model'] = model
    # configs['dataset'] = dataset
    # configs['train_bsz'] = train_bsz
    # configs['test_bsz'] = test_bsz
    # configs['epochs'] = epochs
    # configs['base_lr'] = base_lr
    # configs['end_lr'] = end_lr
    # configs['layers'] = layers
    # configs['proj_type'] = proj_type
    # configs["multi_thread"] = multi_thread 
    
    # configs['emb_dim'] = emb_dim
    # configs['h_dim'] = h_dim
    # configs['head'] = head
    # configs['vocab_size'] = vocab_size
    # configs['max_len'] = max_len
    # configs['word_vec'] = word_vec
    # configs['seed'] = seed

    if args != None:
        configs['train_bsz'] = args.train_bsz
        configs['test_bsz'] = args.test_bsz
        configs['dataset'] = args.dataset
        configs['model'] = args.model
        configs['max_len'] = args.max_len
        configs['seed'] = args.seed
        configs['layers'] = args.layers
        configs['proj_type'] = None if (args.proj_type == None) or (args.proj_type == '') else args.proj_type.replace(' ', '').lower()
        configs['pred_type'] = None if (args.pred_type == None) or (args.pred_type == '') else args.pred_type.replace(' ', '').lower()
        configs['times'] = args.times
        configs['epochs'] = args.epochs
        configs['head'] = args.heads
        configs['vocab_size'] = args.vocab_size
        configs['word_vec'] = args.word_vec
        configs['emb_dim'] = args.emb_dim
        configs['h_dim'] = args.h_dim
        configs['base_lr'] = args.base_lr
        configs['end_lr'] = args.end_lr
        configs["save_path"] = args.save_path
        configs["gpu_ids"] = args.gpus
        configs["multi_t"] = True if args.multi_t.lower() in ['t', 'true'] else False
        configs["profile"] = True if args.profile.lower() in ['t', 'true'] else False

        configs['gpus'] = gpu_setting(args.gpus, args.layers)
            
    return configs

def set_model(name):
    # Single GPU - LSTM BP
    if name == "LSTM_BP_3":
        model = LSTM_BP_3
    elif name == "LSTM_BP_4":
        model = LSTM_BP_4
    elif name == "LSTM_BP_d":
        model = LSTM_BP_d
    
    # Single GPU - LSTM SCPL
    elif name == "LSTM_SCPL_3":
        model = LSTM_SCPL_3
    elif name == "LSTM_SCPL_4":
        model = LSTM_SCPL_4
    elif name == "LSTM_SCPL_5":
        model = LSTM_SCPL_5

    # Multi-GPU - LSTM BP
    elif name == "LSTM_BP_m_d":
        model = LSTM_BP_m_d
    elif name == "LSTM_BP_p_m_d":
        model = LSTM_BP_p_m_d

    # Multi-GPU - LSTM SCPL
    elif name == "LSTM_SCPL_m_d":
        model = LSTM_SCPL_m_d
    elif name == "LSTM_DASCPL_m_d":
        model = LSTM_DASCPL_m_d
    elif name == "LSTM_SCPL_m_4":
        model = LSTM_SCPL_m_4

    # Single GPU - Transfomer BP
    elif name == "Trans_BP_3":
        model = Trans_BP_3
    elif name == "Trans_BP_4":
        model = Trans_BP_4
    elif name == "Trans_BP_d":
        model = Trans_BP_d

    # Single GPU - Transfomer SCPL
    elif name == "Trans_SCPL_3":
        model = Trans_SCPL_3
    elif name == "Trans_SCPL_4":
        model = Trans_SCPL_4
    elif name == "Trans_SCPL_5":
        model = Trans_SCPL_5

    # Multi-GPU- Transfomer BP
    elif name == "Trans_BP_m_d":
        model = Trans_BP_m_d
    elif name == "Trans_BP_p_m_d":
        model = Trans_BP_p_m_d

    # Multi-GPU- Transfomer SCPL
    elif name == "Trans_SCPL_m_d":
        model = Trans_SCPL_m_d
    elif name == "Trans_DASCPL_m_d":
        model = Trans_DASCPL_m_d
    elif name == "Trans_SCPL_m_4":
        model = Trans_SCPL_m_4
    else:
        raise ValueError("Model not supported: {}".format(name))
    
    return model

def train(train_loader, model, optimizer, global_steps, epoch, config):
    train_time = ResultMeter()
    eval_time = ResultMeter()
    data_time = ResultMeter()
    losses = ResultMeter()
    accs = ResultMeter()

    with SynchronizeTimer() as data_timer:
        data_timer.start()
        for step, (X, Y) in enumerate(train_loader): 
            bsz = Y.shape[0]
            global_steps += 1
            data_timer.end()
            data_time.update(data_timer.runtime)

            model.train()
            with SynchronizeTimer() as train_timer:
                if torch.cuda.is_available():
                    X = X.cuda(non_blocking=True)
                    Y = Y.cuda(non_blocking=True)
                output, loss = model(X, Y)
                                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_time.update(train_timer.runtime)
            losses.update(loss.item(), bsz)
            
            model.eval()
            with SynchronizeTimer() as eval_timer:
                with torch.no_grad():
                    output, loss = model(X, Y)
                acc = accuracy(output, Y)
            eval_time.update(eval_timer.runtime)
            accs.update(acc.item(), bsz)
            
            data_timer.start()

    # print info
    print("Train: {0}\t"
        "T_Time {1:.3f}\t"
        "E_Time {2:.3f}\t"
        "DT {3:.3f}\t"
        "loss {4:.3f}\t"
        "Acc {5:.3f}\t".format(epoch, train_time.sum, eval_time.sum, data_time.sum, losses.avg, accs.avg))
    sys.stdout.flush()

    return losses.avg, accs.avg, global_steps, train_time.sum, eval_time.sum

def test(test_loader, model, epoch):
    model.eval()

    data_time = ResultMeter()
    eval_time = ResultMeter()
    accs = ResultMeter()

    with torch.no_grad():
        with SynchronizeTimer() as data_timer:
            for step, (X, Y) in enumerate(test_loader):
                bsz = Y.shape[0]

                data_timer.end()
                data_time.update(data_timer.runtime)

                with SynchronizeTimer() as eval_timer:
                    if torch.cuda.is_available():
                        X = X.cuda(non_blocking=True)
                        Y = Y.cuda(non_blocking=True)
                    output, loss = model(X, Y)
                    acc = accuracy(output, Y)
                accs.update(acc.item(), bsz)

                eval_time.update(eval_timer.runtime)
                data_timer.start()

    # print info
    print("Test:  {0}\t"
        "E_Time {1:.3f}\t"
        "DT {2:.3f}\t"
        "Acc {3:.3f}\t".format(epoch, data_time.sum, eval_time.sum, accs.avg))
    
    sys.stdout.flush()

    return accs.avg, eval_time.sum

def train_multiGPU(train_loader, model, global_steps, epoch, multi_t=True):
    train_time = ResultMeter()
    eval_time = ResultMeter()
    data_time = ResultMeter()
    losses = ResultMeter()
    # accs = ResultMeter()

    accs = [ResultMeter() for i in range(model.num_layers+1)]

    with SynchronizeTimer() as data_timer:
        data_timer.start()
        for step, (X, Y) in enumerate(train_loader): 
            bsz = Y.shape[0]
            global_steps += 1
            data_timer.end()
            data_time.update(data_timer.runtime)
            
            model.train()
            with SynchronizeTimer() as train_timer:
                output, loss = model(X, Y, multi_t=multi_t)

            train_time.update(train_timer.runtime)
            losses.update(loss, bsz)
            
            model.eval()
            with SynchronizeTimer() as eval_timer:
                with torch.no_grad():
                    layer_outputs, true_Ys = model(X, Y)
                    for idx in range(len(layer_outputs)):
                        acc = accuracy(layer_outputs[idx], true_Ys[idx])
                        accs[idx].update(acc.item(), bsz)
                # for idx, layer_y in enumerate(layer_outputs):
                #     if idx == len(layer_outputs)-1: # last layer
                #         acc = accuracy(layer_outputs[-1], true_Ys[-1])
                #     elif layer_y != None:
                #         # print(idx, layer_outputs[idx].shape, true_Ys[idx].shape)
                #         acc = accuracy(layer_outputs[idx], true_Ys[idx])
                #     else:
                #         continue
                #     accs[idx].update(acc.item(), bsz)

            eval_time.update(eval_timer.runtime)       
            # accs.update(acc.item(), bsz)
            data_timer.start()
    
    new_accs = list()
    acc_str = ""
    for acc in accs:
        if acc.avg != 0: 
            new_accs.append(acc.avg)
            acc_str = acc_str + "{:6.3f} ".format(acc.avg)

    # print info
    # (batch_time.avg)*len(train_loader)
    # (data_time.avg)*len(train_loader)
    print("Train: {0}\t"
        "T_Time {1:.3f}\t"
        "E_Time {2:.3f}\t"
        "DT {3:.3f}\t"
        "loss {4:.3f}\t"
        "Acc {5}\t".format(epoch, train_time.sum, eval_time.sum, data_time.sum, losses.avg, acc_str))
        # "Acc {5:.3f}\t".format(epoch, train_time.sum, eval_time.sum, data_time.sum, losses.avg, accs.avg))
    sys.stdout.flush()

    return losses.avg, new_accs, global_steps, train_time.sum, eval_time.sum

def eval_multiGPU(test_loader, model, epoch):
    model.eval()

    data_time = ResultMeter()
    eval_time = ResultMeter()
    # accs = ResultMeter()
    accs = [ResultMeter() for i in range(model.num_layers+1)]

    with torch.no_grad():
        with SynchronizeTimer() as data_timer:
            for step, (X, Y) in enumerate(test_loader):
                bsz = Y.shape[0]

                data_timer.end()
                data_time.update(data_timer.runtime)

                with SynchronizeTimer() as eval_timer:
                    with torch.no_grad():
                        layer_outputs, true_Ys = model(X, Y)
                    for idx in range(len(layer_outputs)):
                        acc = accuracy(layer_outputs[idx], true_Ys[idx])
                        accs[idx].update(acc.item(), bsz)
                    # for idx, layer_y in enumerate(layer_outputs):
                    #     if idx == len(layer_outputs)-1: # last layer
                    #         acc = accuracy(layer_outputs[-1], true_Ys[-1])
                    #     elif layer_y != None:
                    #         acc = accuracy(layer_outputs[idx], true_Ys[idx])
                    #     else:
                    #         continue
                    #     accs[idx].update(acc.item(), bsz)

                eval_time.update(eval_timer.runtime)       
                # accs.update(acc.item(), bsz)
                data_timer.start()

    new_accs = list()
    acc_str = ""
    for acc in accs:
        if acc.avg != 0: 
            new_accs.append(acc.avg)
            acc_str = acc_str + "{:6.3f} ".format(acc.avg)

    # print info
    print("Test:  {0}\t"
        "E_Time {1:.3f}\t"
        "DT {2:.3f}\t"
        "Acc {3}\t".format(epoch, eval_time.sum, data_time.sum, acc_str))
        # "Acc {3:.3f}\t".format(epoch, eval_time.sum, data_time.sum, accs.avg))
    
    sys.stdout.flush()

    return new_accs, eval_time.sum

def setup_seed(seed):
    if int(seed) == -1:
        return "Random"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True
    return seed

def main(times, conf, recorder: ModelResultRecorder==None):
    configs = deepcopy(conf)
    configs['seed'] = setup_seed(configs['seed'])
    
    train_loader, test_loader, n_classes, vocab = get_data(configs)
    word_vec = get_word_vector(vocab, configs['word_vec'])
    configs['max_steps'] = configs['epochs'] * len(train_loader)
    
    configs['n_classes'] = n_classes
    configs['train_loader'] = train_loader
    configs['test_loader'] = test_loader
    configs["vocab_size"] = len(vocab)
    configs["word_vec"] = word_vec

    select_model = set_model(configs['model'])
    
    if select_model.device_type == "multi":
        model = select_model(configs)
    else:
        model = select_model(configs).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=configs['base_lr'])

    print(model)

    print("[Model Info] Model name: {}, Dataset: {}, ".format(model.__class__.__name__, configs['dataset']), end="")
    if select_model.device_type == "multi":
        print("Multi-thread: {}, ".format(configs["multi_t"]), end="")
    print("Train_B: {}, Test_B: {}, ".format(configs['train_bsz'], configs['test_bsz']), end="")
    print("Layer: {}, MaxLen: {}, ".format(configs['layers'], configs['max_len']), end="")
    print("Epoch: {}, Seed: {}".format(configs['epochs'], configs['seed']))

    if recorder != None:
        writer = SummaryWriter(configs["save_path"]+'/tb_log_t({:03d})'.format(times))
    else:
        writer = None
                                                                    
    epoch_train_time = ResultMeter()  
    epoch_train_eval_time = ResultMeter()  
    epoch_test_time = ResultMeter() 
    global_steps = 0
    best_acc = 0
    best_epoch = 0
    
    for epoch in range(1, configs['epochs'] + 1):

        if select_model.device_type == "multi":
            lr = model.opt_step(global_steps)
        else:
            lr = adjust_learning_rate(optimizer, configs['base_lr'], configs['end_lr'], global_steps, configs['max_steps'])
        
        print("[Epoch {}] lr: {:.6f}".format(epoch, lr))

        if select_model.device_type == "multi":
            train_loss, train_acc, global_steps, train_time, train_eval_time = train_multiGPU(
                train_loader, model, global_steps, epoch, configs["multi_t"])
            tb_record_gradient(model=model.model, writer=writer, epoch=epoch)
        else:
            train_loss, train_acc, global_steps, train_time, train_eval_time = train(train_loader, model, optimizer, global_steps, epoch, configs)
            tb_record_gradient(model=model, writer=writer, epoch=epoch)
                                               
        epoch_train_time.update(train_time)
        epoch_train_eval_time.update(train_eval_time)

        if select_model.device_type == "multi":
            test_acc, test_time = eval_multiGPU(test_loader, model, epoch)
        else:
            test_acc, test_time = test(test_loader, model, epoch)

        epoch_test_time.update(test_time)

        if test_acc[-1] > best_acc:
            best_acc = test_acc[-1]
            best_epoch = epoch

        print("Now Epoch time\tAvg {:4.2f}\tStd {:4.2f}".format(epoch_train_time.avg, epoch_train_time.std))
        print("================================================")

        if recorder != None:
            recorder.save_epoch_info(
                t=times, e=epoch, 
                tr_acc=train_acc, tr_loss=train_loss, tr_t=train_time,  tr_ev_t=train_eval_time,
                te_acc=test_acc, te_t=test_time)

            writer.add_scalar(f'model history/train_loss', train_loss, epoch)
            for idx, tr_acc in enumerate(train_acc):
                writer.add_scalar(f'model history/train_acc-{idx}', tr_acc, epoch)
            writer.add_scalar(f'model history/train_time', train_time, epoch)
            writer.add_scalar(f'model history/train_time', train_eval_time, epoch)
            writer.add_scalar(f'model history/learning_rate', lr, epoch)
            for idx, te_acc in enumerate(test_acc):
                writer.add_scalar(f'model history/test_acc-{idx}', te_acc, epoch)
            writer.add_scalar(f'model history/test_time', test_time, epoch)
        
    # state = {
    #     "configs": configs,
    #     "model": model.state_dict(),
    #     "optimizer": optimizer.state_dict(),
    #     "epoch": epoch,
    # }
    # save_files = os.path.join("./save_models/", "ckpt_last_{0}.pth".format(i))
    # torch.save(state, save_files)

    # del state
    print("Best accuracy: {:.2f} / epoch: {}".format(best_acc, best_epoch))
    print("Epoch time\tAvg {:4.2f}\tStd {:4.2f}".format(epoch_train_time.avg, epoch_train_time.std))
                                                                    
    # Memory recycle
    gpus_info = calculate_GPUs_usage(configs['gpus'])
    del model

    return {"best_acc": best_acc, "best_epoch": best_epoch, "gpu_infos": gpus_info,
            "epoch_train_time_avg": epoch_train_time.avg, 
            "epoch_train_ev_time_avg": epoch_train_eval_time.avg, 
            "epoch_test_time_avg": epoch_test_time.avg}

if __name__ == '__main__':

    args = get_args()
    configs = read_config(args = args)

    run_times = configs['times']

    best_acc_meter = ResultMeter()
    best_epoch_meter = ResultMeter()
    epoch_train_time_meter = ResultMeter()
    epoch_train_eval_time_meter = ResultMeter()
    training_time_meter = ResultMeter()
    gpu_ram_meter = ResultMeter()

    if configs["save_path"] != None:
        from torch.utils.tensorboard import SummaryWriter
        recorder = ModelResultRecorder(model_name=configs['model'])
        if configs["profile"] == True:
            from model.nlp_multi_profile import LSTM_SCPL_m_d, LSTM_BP_m_d, Trans_SCPL_m_d, Trans_BP_m_d
            print("[INFO] Results and model profile will be saved in \"{}*\" later".format(configs["save_path"]))
        else:
            print("[INFO] Results will be saved in \"{}*\" later".format(configs["save_path"]))
        sys.stdout = StdoutWithLogger(configs["save_path"]+'.log') # Write log
    else:
        recorder = None

    print("[-] Setting {0} times running".format(run_times))

    for i in range(run_times):
        print("\n[Times {:2d}] Start".format(i+1))
        
        with SynchronizeTimer() as timer:
            result = main(i, configs, recorder)
        run_time = timer.runtime
        
        best_acc_meter.update(result["best_acc"])
        best_epoch_meter.update(result["best_epoch"])
        epoch_train_time_meter.update(result["epoch_train_time_avg"])
        epoch_train_eval_time_meter.update(result["epoch_train_ev_time_avg"])
        training_time_meter.update(run_time)
        gpu_ram_meter.update(result["gpu_infos"]["reserved_all_m"])

        print("Runtime (sec): {:.3f}".format(run_time))
        print("Runtime (min): {:.3f}".format(run_time/60))
        print("[Times {:2d}] End".format(i+1))
        print("================================================")

        if configs["save_path"] != None:
            recorder.add(times=i, best_test_acc=result["best_acc"], best_test_epoch=result["best_epoch"], 
            epoch_train_time=result["epoch_train_time_avg"], epoch_train_eval_time=result["epoch_train_ev_time_avg"],
            epoch_test_time=result["epoch_test_time_avg"], runtime=run_time, gpus_info=result["gpu_infos"])

    if configs["save_path"] != None:
        recorder.save_mean_std_config(
            best_test_accs=best_acc_meter,
            best_test_epochs=best_epoch_meter, 
            epoch_train_times=epoch_train_time_meter, 
            epoch_train_eval_times=epoch_train_eval_time_meter,
            run_times=training_time_meter,
            gpu_ram=gpu_ram_meter,
            config = configs
        )
        recorder.save(configs["save_path"])
    
    print("================================================")
    print("[-] Finish {0} times running".format(run_times))
    print("[-] Best acc list:", best_acc_meter)
    print(" |---- Avg {:5.3f}\tStd {:5.3f}".format(best_acc_meter.avg, best_acc_meter.std))
    print("[-] Best epoch list:", best_epoch_meter)
    print(" |---- Avg {:5.3f}\tStd {:5.3f}".format(best_epoch_meter.avg, best_epoch_meter.std))
    print("[-] Epoch time list:", epoch_train_time_meter)
    print(" |---- Avg {:5.3f}\tStd {:5.3f}".format(epoch_train_time_meter.avg, epoch_train_time_meter.std))
    print("[-] Runtime list:", training_time_meter)
    print(" |---- Avg {:5.3f}\tStd {:5.3f}".format(training_time_meter.avg, training_time_meter.std))

    # os._exit(0)