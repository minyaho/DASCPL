import torch
import os, sys, argparse
#os.environ['CUDA_LAUNCH_BLOCKING'] = 1
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
from copy import deepcopy
from utils import ResultMeter, ModelResultRecorder, SynchronizeTimer, StdoutWithLogger
from utils import adjust_learning_rate, accuracy, gpu_setting, tb_record_gradient, setup_seed, calculate_GPUs_usage, check_config, model_save

def get_args():
    parser = argparse.ArgumentParser('Vision SCPL training')
    parser.add_argument('--model', type=str, help='Model name', default="VGG_BP_m")
    parser.add_argument('--dataset', type=str, help='Dataset name', default="cifar10")
    parser.add_argument('--times', type=int, help='Times of experiment', default="1")
    parser.add_argument('--epochs', type=int, help='Number of epochs for training', default=200)
    parser.add_argument('--train_bsz', type=int, help='Batch size of training data', default=1024)
    parser.add_argument('--test_bsz', type=int, help='Batch size of test data', default=1024)
    parser.add_argument('--base_lr', type=float, help='Initial learning rate', default=0.001)
    parser.add_argument('--end_lr', type=float, help='Learning rate at the end of training', default=0.00001)
    parser.add_argument('--gpus', type=str, help='ID of the GPU device. If you want to use multiple GPUs, \
         you can separate them with commas, e.g., \"0,1\". The model type is Single GPU will only use first id.', default="0")
    parser.add_argument('--seed', type=int, help='Random seed in the experiment. \
        If you don\'t want to fix the random seed, you need to type "-1"', default="-1")
    parser.add_argument('--multi_t', type=str, help='Multi-threaded on-off flag. On is \"true\". Off is \"false\"', default="true")
    parser.add_argument('--proj_type', type=str, help='Projective head type in contrastive loss. \
        \"i\" is identity. \"l\" is linear. \"m\" is mlp. (mulitGPU types only)', default=None)
    parser.add_argument('--pred_type', type=str, help='Predictor type in predict loss. \
        \"i\" is identity. \"l\" is linear. \"m\" is mlp. (mulitGPU types only)', default=None)
    parser.add_argument('--save_path', type=str, help='Save path of the model log. \
        There are many types of logs, such as training logs, model results (JSON) and tensorboard files. \"None\" means do not save.', default=None)
    parser.add_argument('--profiler', type=str, help='Profiler of model. \
        If you want to use the profiler, please type "true" and set the "save_path". "false" means do not use and save. (mulitGPU types only)', default="false")
    parser.add_argument('--train_eval', type=str, help='On-off flag for evaluation behavior during training. (mulitGPU types only)', default="true")
    parser.add_argument('--temperature', type=float, help='Temperature parameter of contrastive loss', default=0.1)

    # Vision Options
    parser.add_argument('--aug_type', type=str, help='Type of Data augmentation. \
        Use \"basic\" augmentation like BP commonly used, or \"strong\" augmentation like contrastive learning used. Options: \"basic\", \"strong\"', default="strong")

    args = parser.parse_args()

    return args

def read_config(args=None):
    configs = dict()

    if args != None:
        configs['train_bsz'] = args.train_bsz
        configs['test_bsz'] = args.test_bsz
        configs['dataset'] = args.dataset
        configs['model'] = args.model
        configs['epochs'] = args.epochs
        configs['base_lr'] = args.base_lr
        configs['end_lr'] = args.end_lr
        configs['seed'] = args.seed
        configs['aug_type'] = args.aug_type
        configs['proj_type'] = None if (args.proj_type == None) or (args.proj_type.replace(' ', '').lower() in ['none', '']) else args.proj_type.replace(' ', '').lower()
        configs['pred_type'] = None if (args.pred_type == None) or (args.pred_type.replace(' ', '').lower() in ['none', '']) else args.pred_type.replace(' ', '').lower()
        configs['times'] = args.times
        configs["save_path"] = None if (args.save_path == None) or (args.save_path.lower() == 'none') else args.save_path
        configs["gpu_ids"] = args.gpus
        configs["multi_t"] = True if args.multi_t.lower() in ['t', 'true'] else False
        configs["profiler"] = True if args.profiler.lower() in ['t', 'true'] else False
        configs["train_eval"] = True if args.train_eval.lower() in ['t', 'true'] else False
        configs['temperature'] = args.temperature
        
        layers = 4
        assert layers==4, "layers are only 4"
        configs['layers'] = layers
        configs['gpus'] = gpu_setting(gpu_list=args.gpus, layers_num=layers)

        check_config(configs)

    return configs

def set_model(name):
    # VGG - Single GPU
    ## BP
    if name == "VGG_BP":
        model = VGG
    ## SCPL
    elif name == "VGG_SCPL":
        model = VGG_SCPL
    elif name == "VGG_SCPL_REWRITE":
        model = VGG_SCPL_REWRITE

    # VGG - Multi-GPU
    ## BP
    elif name == "VGG_BP_m":
        model = VGG_BP_m
    elif name == "VGG_BP_p_m":
        model = VGG_BP_p_m
    ## SCPL
    elif name == "VGG_SCPL_m":
        model = VGG_SCPL_m
    ## DASCPL
    elif name == "VGG_DASCPL_m":
        model = VGG_DASCPL_m

    # ResNet - Single GPU 
    ## BP
    elif name == "resnet_BP":
        model = resnet18
    ## SCPL
    elif name == "resnet_SCPL":
        model = resnet18_SCPL

    # ResNet - Multi-GPU
    ## BP
    elif name == "resnet_BP_m":
        model = resnet18_BP_m
    elif name == "resnet_BP_p_m":
        model = resnet18_BP_p_m
    ## SCPL
    elif name == "resnet_SCPL_m":
        model = resnet18_SCPL_m
    ## DASCPL
    elif name == "resnet_DASCPL_m":
        model = resnet18_DASCPL_m

    # Other - Single GPU 
    elif name == "VGG_AL":
        model = VGG_AL
    elif name == "VGG_PredSim":
        model = VGG_PredSim
    elif name == "resnet_AL":
        model = resnet18_AL
    elif name == "resnet_PredSim":
        model = resnet18_PredSim
    elif name == "CNN_BP":
        model = CNN
    elif name == "CNN_AL":
        model = CNN_AL
    elif name == "CNN_SCPL":
        model = CNN_SCPL
    elif name == "CNN_PredSim":
        model = CNN_PredSim

    # Unknow model
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

            if config['aug_type'] == "strong":
                if config['dataset'] == "cifar10" or config['dataset'] == "cifar100":
                    X = torch.cat(X)
                    Y = torch.cat(Y)
                else:
                    X = torch.cat(X)
                    Y = torch.cat([Y, Y])

            bsz = Y.shape[0]
            global_steps += 1
            data_timer.end()
            data_time.update(data_timer.runtime)

            model.train()
            with SynchronizeTimer() as train_timer:
                if torch.cuda.is_available():
                    X = X.cuda(non_blocking=True)
                    Y = Y.cuda(non_blocking=True)
                loss = model(X, Y)
                                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_time.update(train_timer.runtime)
            losses.update(loss.item(), bsz)
            
            model.eval()
            with SynchronizeTimer() as eval_timer:
                with torch.no_grad():
                    output = model(X, Y)
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
                    output = model(X, Y)
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
            if model.aug_type == "strong":
                if model.dataset == "cifar10" or model.dataset == "cifar100":
                    X = torch.cat(X)
                    Y = torch.cat(Y)
                else:
                    X = torch.cat(X)
                    Y = torch.cat([Y, Y])

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
    configs['seed'] = setup_seed(configs['seed'])
    
    train_loader, test_loader, n_classes = set_loader(configs['dataset'], configs['train_bsz'], configs['test_bsz'], configs['aug_type'])
    configs['max_steps'] = configs['epochs'] * len(train_loader)
    
    configs['n_classes'] = n_classes
    configs['train_loader'] = train_loader
    configs['test_loader'] = test_loader

    select_model = set_model(configs['model'])
    
    if select_model.device_type == "multi":
        model = select_model(configs)
        optimizer = None # Include to the module
    else:
        model = select_model(n_classes).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=configs['base_lr'])

    print("[Model Info] Model name: {}, Dataset: {}, ".format(model.__class__.__name__, configs['dataset']), end="")
    if select_model.device_type == "multi":
        print("Multi-thread: {}, ".format(configs["multi_t"]), end="")
        print("Device: {}, ".format(configs["gpus"]), end="")
    else:
        print("Device: {}, ".format(configs["gpus"][0]), end="")
    print("Train_B: {}, Test_B: {}, D.A.: {}, ".format(configs['train_bsz'], configs['test_bsz'], configs['aug_type']), end="")
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
    
    for epoch in range(1, configs['epochs'] + 1):

        if select_model.device_type == "multi":
            lr = model.opt_step(global_steps)
        else:
            lr = adjust_learning_rate(optimizer, configs['base_lr'], configs['end_lr'], global_steps, configs['max_steps'])
        
        print("[Epoch {}] lr: {:.6f}".format(epoch, lr))

        if select_model.device_type == "multi":
            train_loss, train_acc, global_steps, train_time, train_eval_time = train_multiGPU(
                train_loader, model, global_steps, epoch, configs["multi_t"], configs["train_eval"])
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
    from utils.vision import set_loader
    from model.ResNet import resnet18, resnet18_AL, resnet18_SCPL, resnet18_PredSim
    from model.VGG import VGG, VGG_AL, VGG_SCPL, VGG_PredSim, VGG_SCPL_REWRITE
    from model.vanillaCNN import CNN, CNN_AL, CNN_SCPL, CNN_PredSim
    from model.vision_multi import VGG_BP_m, VGG_BP_p_m, VGG_SCPL_m, VGG_DASCPL_m
    from model.vision_multi import resnet18_BP_m, resnet18_BP_p_m, resnet18_SCPL_m, resnet18_DASCPL_m

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
            from model.vision_multi_profiler import VGG_BP_m, VGG_BP_p_m, VGG_SCPL_m, VGG_DASCPL_m
            from model.vision_multi_profiler import resnet18_BP_m, resnet18_BP_p_m, resnet18_SCPL_m, resnet18_DASCPL_m
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