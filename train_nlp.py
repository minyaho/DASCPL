import torch
import os, sys, argparse
#os.environ['CUDA_LAUNCH_BLOCKING'] = 1
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
from copy import deepcopy
from utils import ResultMeter, ModelResultRecorder, SynchronizeTimer, StdoutWithLogger     
from utils import adjust_learning_rate, accuracy, gpu_setting, tb_record_gradient, setup_seed, calculate_GPUs_usage, check_config, model_save

def get_args():
    parser = argparse.ArgumentParser('NLP SCPL training')
    # General Options
    parser.add_argument('--model', type=str, help='Model name', default="LSTM_BP_m_d")
    parser.add_argument('--dataset', type=str, help='Dataset name', default="ag_news")
    parser.add_argument('--times', type=int, help='Number of experiments to run', default="1")
    parser.add_argument('--epochs', type=int, help='Number of training epochs', default=50)
    parser.add_argument('--train_bsz', type=int, help='Batch size of training data', default=1024)
    parser.add_argument('--test_bsz', type=int, help='Batch size of test data', default=1024)
    parser.add_argument('--base_lr', type=float, help='Initial learning rate', default=0.001)
    parser.add_argument('--end_lr', type=float, help='Learning rate at the end of training', default=0.001)
    parser.add_argument('--gpus', type=str, help=' ID of the GPU device. If you want to use multiple GPUs, you can separate their IDs with commas, \
         e.g., \"0,1\". For single GPU models, only the first GPU ID will be used.', default="0")
    parser.add_argument('--seed', type=int, help='Random seed used in the experiment. \
                        Use \"-1\" to generate a random seed for each run.', default="-1")
    parser.add_argument('--multi_t', type=str, help='Multi-threading flag. Set it to \"true\" to enable multi-threading, or \"false\" to disable it.', default="true")
    parser.add_argument('--proj_type', type=str, help='Projective head type in contrastive loss. \
        Use \"i\" for identity, \"l\" for linear, and \"m\" is mlp (only for multi-GPU models).', default=None)
    parser.add_argument('--pred_type', type=str, help='Predictor type in predict loss. \
        Use \"i\" for identity, \"l\" for linear, and \"m\" is mlp (only for multi-GPU models).', default=None)
    parser.add_argument('--save_path', type=str, help='Save path of the model log. \
                        Different types of logs, such as training logs, model results (JSON), and tensorboard files, can be saved. \
                        Use \"None\" to disable saving.', default=None)
    parser.add_argument('--profiler', type=str, help='Model profiler. \
                        Set it to \"true\" to enable the profiler and specify the \"save_path\". \
                        Set it to \"false\" to disable the profiler.', default="false")
    parser.add_argument('--train_eval', type=str, help='Flag to enable evaluation during training (only for multi-GPU models).', default="true")
    parser.add_argument('--train_eval_times', type=int, help='The number of epochs between evaluations during training.', default=1)
    parser.add_argument('--temperature', type=float, help='Temperature parameter of contrastive loss.', default=0.1)
    parser.add_argument('--noise_rate', type=float, help='Noise rate of labels in training dataset (default is 0 for no noise).', default=0.0)
    parser.add_argument('--speedup', type=str, help='This option will use "\"torch.backends.cudnn.benchmark\" to accelerate training. If you want to use it, please input \"t\".', default="f")
    parser.add_argument('--determine', type=str, help='This option will use \"torch.backends.cudnn.deterministic\" to enable the deterministic convolutional algorithm. \
                        If you want to use it, please input \"t\".', default="f")

    # NLP Options
    parser.add_argument('--max_len', type=int, help='Maximum length for the sequence of input samples', default="60")
    parser.add_argument('--h_dim', type=int, help='Dimensions of the hidden layer', default="300")
    parser.add_argument('--layers', type=int, help='Number of layers of the model. The minimum is \"2\". \
                        The first layer is the pre-training embedding layer, and the latter layer is lstm or transformer.', default="4")
    parser.add_argument('--heads', type=int, help='Number of heads in the transformer encoder. \
                        This option is only available for the transformer model.', default="6")
    parser.add_argument('--vocab_size', type=int, help='Size of vocabulary dictionary.', default="30000")
    parser.add_argument('--word_vec', type=str, help='Type of word embedding', default="glove")
    parser.add_argument('--emb_dim', type=int, help='Dimension of word embedding', default="300")

    args = parser.parse_args()

    return args

def read_config(args=None):
    configs = dict()

    if args != None:
        configs['train_bsz'] = args.train_bsz
        configs['test_bsz'] = args.test_bsz
        configs['dataset'] = args.dataset
        configs['model'] = args.model
        configs['max_len'] = args.max_len
        configs['seed'] = args.seed
        configs['layers'] = args.layers
        configs['proj_type'] = None if (args.proj_type == None) or (args.proj_type.replace(' ', '').lower() in ['none', '']) else args.proj_type.replace(' ', '').lower()
        configs['pred_type'] = None if (args.pred_type == None) or (args.pred_type.replace(' ', '').lower() in ['none', '']) else args.pred_type.replace(' ', '').lower()
        configs['times'] = args.times
        configs['epochs'] = args.epochs
        configs['head'] = args.heads
        configs['vocab_size'] = args.vocab_size
        configs['word_vec'] = args.word_vec
        configs['emb_dim'] = args.emb_dim
        configs['h_dim'] = args.h_dim
        configs['base_lr'] = args.base_lr
        configs['end_lr'] = args.end_lr
        configs["save_path"] = None if (args.save_path == None) or (args.save_path.lower() == 'none') else args.save_path
        configs["gpu_ids"] = args.gpus
        configs["multi_t"] = True if args.multi_t.lower() in ['t', 'true'] else False
        configs["profiler"] = True if args.profiler.lower() in ['t', 'true'] else False
        configs["train_eval"] = True if args.train_eval.lower() in ['t', 'true'] else False
        configs["train_eval_times"] = args.train_eval_times
        configs['temperature'] = args.temperature
        configs['noise_rate'] = args.noise_rate
        configs['speedup'] = True if args.speedup.lower() in ['t', 'true'] else False
        configs['determine'] = True if args.determine.lower() in ['t', 'true'] else False

        configs['gpus'] = gpu_setting(args.gpus, args.layers)

        check_config(configs)
            
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
    elif name == "LSTM_EE_m_d":
        model = LSTM_EE_m_d
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
    elif name == "Trans_EE_m_d":
        model = Trans_EE_m_d
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

    return losses.avg, [accs.avg], global_steps, train_time.sum, eval_time.sum

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

    return [accs.avg], eval_time.sum

def train_multiGPU(train_loader, model, global_steps, epoch, multi_t=True, eval_flag=True):
    train_time = ResultMeter()
    eval_time = ResultMeter()
    data_time = ResultMeter()
    losses = ResultMeter()
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
            
            if eval_flag == True:
                model.eval()
                acc_temp = list()
                with SynchronizeTimer() as eval_timer:
                    with torch.no_grad():
                        layer_outputs, true_Ys = model(X, Y)
                        for idx in range(len(layer_outputs)):
                            acc = accuracy(layer_outputs[idx], true_Ys[idx])
                            acc_temp.append(acc)
                for idx, acc in enumerate(acc_temp):
                    accs[idx].update(acc.item(), bsz)

                eval_time.update(eval_timer.runtime)  

            data_timer.start()
    
    new_accs = list()
    acc_str = ""
    if not eval_flag:
        acc_str = "no eval."
    else:
        for acc in accs:
            if not acc.is_empty:  
                new_accs.append(acc.avg)
                acc_str = acc_str + "{:6.3f} ".format(acc.avg)

    # print info
    print("Train: {0}\t"
        "T_Time {1:.3f}\t"
        "E_Time {2:.3f}\t"
        "DT {3:.3f}\t"
        "loss {4:.3f}\t"
        "Acc {5}\t".format(epoch, train_time.sum, eval_time.sum, data_time.sum, losses.avg, acc_str))
    sys.stdout.flush()

    return losses.avg, new_accs, global_steps, train_time.sum, eval_time.sum

def eval_multiGPU(test_loader, model, epoch):
    model.eval()

    data_time = ResultMeter()
    eval_time = ResultMeter()
    accs = [ResultMeter() for i in range(model.num_layers+1)]

    with torch.no_grad():
        with SynchronizeTimer() as data_timer:
            for step, (X, Y) in enumerate(test_loader):
                bsz = Y.shape[0]

                data_timer.end()
                data_time.update(data_timer.runtime)

                with SynchronizeTimer() as eval_timer:
                    layer_outputs, true_Ys = model(X, Y)
                    for idx in range(len(layer_outputs)):
                        acc = accuracy(layer_outputs[idx], true_Ys[idx])
                        accs[idx].update(acc.item(), bsz)

                eval_time.update(eval_timer.runtime)       
                data_timer.start()

    new_accs = list()
    acc_str = ""
    for acc in accs:
        if not acc.is_empty:  
            new_accs.append(acc.avg)
            acc_str = acc_str + "{:6.3f} ".format(acc.avg)

    # print info
    print("Test:  {0}\t"
        "E_Time {1:.3f}\t"
        "DT {2:.3f}\t"
        "Acc {3}\t".format(epoch, eval_time.sum, data_time.sum, acc_str))
    
    sys.stdout.flush()

    return new_accs, eval_time.sum

def main(times, conf, recorder: ModelResultRecorder==None):
    configs = deepcopy(conf)
    configs['seed'] = setup_seed(configs)
    
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
        optimizer = None # Include to the module
    else:
        model = select_model(configs).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=configs['base_lr'])

    print("[Model Info] Model name: {}, Dataset: {}, ".format(model.__class__.__name__, configs['dataset']), end="")
    if select_model.device_type == "multi":
        print("Multi-thread: {}, ".format(configs["multi_t"]), end="")
        print("Device: {}, ".format(configs["gpus"]), end="")
    else:
        print("Device: {}, ".format(configs["gpus"][0]), end="")
    print("Train_B: {}, Test_B: {}, ".format(configs['train_bsz'], configs['test_bsz']), end="")
    print("Layer: {}, MaxLen: {}, ".format(configs['layers'], configs['max_len']), end="")
    print("Epoch: {}, Train eval: {}, Seed: {}".format(configs['epochs'], configs['train_eval'], configs['seed']))

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
    train_eval_flag = False
    
    for epoch in range(1, configs['epochs'] + 1):

        if configs["train_eval"] == True:
            train_eval_flag = ((epoch%configs["train_eval_times"])==0)

        if select_model.device_type == "multi":
            lr = model.opt_step(global_steps)
        else:
            lr = adjust_learning_rate(optimizer, configs['base_lr'], configs['end_lr'], global_steps, configs['max_steps'])
        
        print("[Epoch {}] lr: {:.6f}".format(epoch, lr))

        if select_model.device_type == "multi":
            train_loss, train_acc, global_steps, train_time, train_eval_time = train_multiGPU(
                train_loader, model, global_steps, epoch, configs["multi_t"], train_eval_flag)
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
                t=times, e=epoch, lr=lr,
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
    
    # # Save model
    # if configs["save_path"] != None:
    #     model_save(times, configs, model, configs["save_path"], optimizer=optimizer)

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

    # Load packages
    from utils.nlp import get_data, get_word_vector
    from model.nlp_single import LSTM_BP_3, LSTM_BP_4, LSTM_BP_d, LSTM_SCPL_3, LSTM_SCPL_4, LSTM_SCPL_5
    from model.nlp_single import Trans_BP_3, Trans_BP_4, Trans_BP_d, Trans_SCPL_3, Trans_SCPL_4, Trans_SCPL_5
    from model.nlp_multi import LSTM_SCPL_m_4, LSTM_SCPL_m_d, LSTM_BP_m_d, LSTM_BP_p_m_d, LSTM_DASCPL_m_d, LSTM_EE_m_d
    from model.nlp_multi import Trans_SCPL_m_4, Trans_SCPL_m_d, Trans_BP_m_d, Trans_BP_p_m_d, Trans_DASCPL_m_d, Trans_EE_m_d

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
        if configs["profiler"] == True:
            from model.nlp_multi_profiler import LSTM_BP_m_d, LSTM_BP_p_m_d, LSTM_SCPL_m_d, LSTM_DASCPL_m_d, LSTM_EE_m_d
            from model.nlp_multi_profiler import Trans_BP_m_d, Trans_BP_p_m_d, Trans_SCPL_m_d, Trans_DASCPL_m_d, Trans_EE_m_d
            print("[INFO] Results and model profiler will be saved in \"{}*\" later".format(configs["save_path"]))
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