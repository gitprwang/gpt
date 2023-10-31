from util.model import ChemGPT
from config import cfg
import random
import numpy as np
import torch
import os
import torch.distributed as dist
from datetime import timedelta, date
import subprocess
from torch.utils.data import DataLoader
from collections import defaultdict
from util.loss import Loss_log
from util.dataloader import ChemData, ChemDataSet, MyTokenizer
from transformers.trainer_pt_utils import get_parameter_names
from src.data import DataCollatorForDenoisingTasks
import deepspeed
from tqdm import tqdm
import warnings

import sys
import re
import json
import time
import logging
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


warnings.filterwarnings('ignore')

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def get_dump_path(params):
    """
    Create a directory to store the experiment.
    """
    assert len(params.exp_name) > 0
    assert not params.dump_path in ('', None), \
            'Please choose your favorite destination for dump.'
    dump_path = params.dump_path

    # create the sweep path if it does not exist
    when = date.today().strftime('%m%d-')
    sweep_path = os.path.join(dump_path, when + params.exp_name)
    if not os.path.exists(sweep_path):
        subprocess.Popen("mkdir -p %s" % sweep_path, shell=True).wait()

    # create an random ID for the job if it is not given in the parameters.
    if params.exp_id == '':
        chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
        while True:
            exp_id = ''.join(random.choice(chars) for _ in range(10))
            if not os.path.isdir(os.path.join(sweep_path, exp_id)):
                break
        params.exp_id = exp_id

    # create the dump folder / update parameters
    exp_folder = os.path.join(sweep_path, params.exp_id)
    if not os.path.isdir(exp_folder):
        subprocess.Popen("mkdir -p %s" % exp_folder, shell=True).wait()
    return exp_folder

class Runner:
    def __init__(self, args, writer=None, logger=None, rank=0):
        self.rank = rank
        self.args = args
        self.writer = writer
        self.logger = logger
        self.logger_path = get_dump_path(self.args)
        # model choice
        self.model = ChemGPT(self.args)
        # data loading
        
        self.data_init()
        if rank == 0:
            self.logger.info("data init")
            self.logger.info(self.train_set)
        
        set_seed(args.random_seed)
        if rank == 0:
            self.logger.info("data init")

        self.dataloader_init(self.train_set) 
        if rank == 0:
            self.logger.info("data init")
            self.logger.info(self.train_dataloader.dataset)
        
        self.optim_init(self.args)
        if rank == 0:
            self.logger.info("data init")

    def data_init(self):
        # 完成tokenize 和 设置分布式训练的采样器
        zinc_data_path = '/home/Zhouyu/MODEL/task1/test/asd/a.txt'
        uspto_data_path = '/home/Zhouyu/MODEL/task1/test/asd/b.txt'
        rxn_data_path = '/home/Zhouyu/MODEL/task1/test/asd/c.txt'
        
        all_data = ChemData([zinc_data_path], [uspto_data_path, rxn_data_path])
        train_data = ChemDataSet(all_data.train_data)
        

        if self.rank == 0:
            self.logger.info("Loading train dataset...")

        self.train_set = train_data

        if self.rank == 0:
            self.logger.info("Finish loading!")

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_set)

    def dataloader_init(self, train_set=None):
        # 完成dataloader的初始化
        bs = self.args.batch_size

        self.args.workers = min([os.cpu_count(), self.args.batch_size, self.args.workers])
        '''train_collator = DataCollatorForDenoisingTasks(
            self.tokenizer,
            self.mask_ratio,
            self.poisson_lambda,
            self.pad_to_multiple_of,
        )   '''
        self.train_dataloader = self._dataloader_dist(train_set, self.train_sampler, bs, None)
    
    def optim_init(self, opt, total_step=None):
        
        step_per_epoch = len(self.train_dataloader)
        opt.total_steps = int(step_per_epoch * opt.epoch) if total_step is None else int(total_step)
        
        if self.rank == 0 and total_step is None:
            self.logger.info(f"total_steps: {opt.total_steps}")
            self.logger.info(f"weight_decay: {opt.weight_decay}")


    def _dataloader_dist(self, train_set, train_sampler, batch_size, collator):
        torch.multiprocessing.set_start_method('spawn', force=True)
        train_dataloader = DataLoader(
            train_set,
            sampler=train_sampler,
            pin_memory=False,
            num_workers=self.args.workers,
            persistent_workers=True, 
            drop_last=True,
            batch_size=batch_size,
            collate_fn=collator,
        )
        return train_dataloader

    def _dataloader(self, train_set, batch_size, collator):
        train_dataloader = DataLoader(
            train_set,
            num_workers=self.args.workers,
            persistent_workers=True,
            shuffle=(self.args.only_test == 0),
            drop_last=(self.args.only_test == 0),
            batch_size=batch_size,
            collate_fn=collator
        )
        return train_dataloader

    def run(self):
        self.loss_log = Loss_log()
        self.curr_loss = 0.
        self.lr = self.args.lr
        self.curr_loss_dic = defaultdict(float)
        self.loss_weight = [1, 1]
        self.step = 0
        
        decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]
        deepspeed_config_path = '/home/Zhouyu/MODEL/task1/code/code/config.json'
        self.model_engine, optimizer, _, self.scheduler = deepspeed.initialize(self.args, model=self.model,
                                                        model_parameters=optimizer_grouped_parameters, config=deepspeed_config_path)
        
        print("deepspeed initialized")
        with tqdm(total=self.args.epoch) as _tqdm: 
            for i in range(self.args.epoch):
                print("ready to run")
                self.train(_tqdm)
                print("finish")
                _tqdm.update(1)
        print("chenckpoint is about to be saved")
        self.model_engine.save_checkpoint(save_dir=os.path.join(self.logger_path, 'model'), client_state={'checkpoint_step': self.step})


    def loss_output(self, batch):
        input_ids = batch["input_ids"].cuda()
        decoder_input_ids = batch["decoder_input_ids"].cuda()
        attention_mask = input_ids != self.tokenizer.pad_token_id
        labels = batch["labels"].cuda()
        _output = self.model_engine(input_ids=input_ids, 
                            attention_mask=attention_mask,
                            decoder_input_ids=decoder_input_ids, 
                            labels=labels)
        loss = _output.loss
        return loss
    
    # one time train
    def train(self, _tqdm):
        self.model_engine.train()
        print("1")
        curr_loss = 0.
        self.loss_log.acc_init()
        print("2")
        for batch in self.train_dataloader:
            print("3")
            # Forward pass
            loss = self.loss_output(batch)
            # Backward pass
            self.model_engine.backward(loss)
            # Optimizer Step
            self.model_engine.step()

            self.step += 1
            print("4")
            # -------- stat --------
            if is_main_process():
                print("5")
                self.output_statistic(loss, output=None)
                self.lr = self.get_learning_rate()
                _tqdm.set_description(f'Train | step [{self.step}/{self.args.total_steps}] LR [{self.lr:.5f}] Loss {self.loss_log.get_loss():.5f} ')
                if self.step % self.args.eval_step == 0 and self.step > 0:
                    
                    self.loss_log.update(self.curr_loss)
                    self.update_loss_log()
                    self.writer.add_scalars("lr", {"lr": self.lr}, self.step)

            if self.step % 10000 == 0:
                self.model_engine.save_checkpoint(save_dir=os.path.join(self.logger_path, 'model'), client_state={'checkpoint_step': self.step})

    def get_learning_rate(self):
        # with deepspeed's fp16 and dynamic loss scale enabled the optimizer/scheduler steps may
        # not run for the first few dozen steps while loss scale is too large, and thus during
        # that time `get_last_lr` will fail if called during that warm up stage, so work around it:
        try:
            last_lr = self.scheduler.get_last_lr()[-1]
        except:
            last_lr = 0
        # except AssertionError as e:
        #     if "need to call step" in str(e):
        #         logger.warning("tried to get lr value before scheduler/optimizer started stepping, returning lr=0")
        #         last_lr = 0
        #     else:
        #         raise
        return last_lr
    
    def output_statistic(self, loss, output):
        self.curr_loss += loss.item()
        if output is None:
            return
        for key in output['loss_dic'].keys():
            self.curr_loss_dic[key] += output['loss_dic'][key]
        
        if 'loss_weight' in output and output['loss_weight'] is not None:
            self.loss_weight = output['loss_weight']

    def update_loss_log(self):
        vis_dict = {"train_loss": self.curr_loss}
        vis_dict.update(self.curr_loss_dic)
        self.writer.add_scalars("loss", vis_dict, self.step)
        
        # init log loss
        self.curr_loss = 0.
        for key in self.curr_loss_dic:
            self.curr_loss_dic[key] = 0.

def set_seed(seed):
    """
    Freeze every seed for reproducibility.
    torch.cuda.manual_seed_all is useful when using random generation on GPUs.
    e.g. torch.cuda.FloatTensor(100).uniform_()
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'  
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()

class LogFormatter():

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''


def create_logger(filepath, rank):
    """
    Create a logger.
    Use a different log file for each process.
    """
    # create log formatter
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    if filepath is not None:
        if rank > 0:
            filepath = '%s-%i' % (filepath, rank)
        file_handler = logging.FileHandler(filepath, "a", encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    return logger

def initialize_exp(params):
    """
    Initialize the experiment:
    - dump parameters
    - create a logger
    """
    # dump parameters
    exp_folder = get_dump_path(params)
    json.dump(vars(params), open(os.path.join(exp_folder, 'params.pkl'), 'w'), indent=4)

    # get running command
    command = ["python", sys.argv[0]]
    for x in sys.argv[1:]:
        if x.startswith('--'):
            assert '"' not in x and "'" not in x
            command.append(x)
        else:
            assert "'" not in x
            if re.match('^[a-zA-Z0-9_]+$', x):
                command.append("%s" % x)
            else:
                command.append("'%s'" % x)
    command = ' '.join(command)
    params.command = command + ' --exp_id "%s"' % params.exp_id

    # check experiment name
    assert len(params.exp_name.strip()) > 0

    # create a logger
    logger = create_logger(os.path.join(exp_folder, 'train.log'), rank=getattr(params, 'global_rank', 0))
    logger.info("============ Initialized logger ============")
    # logger.info("\n".join("%s: %s" % (k, str(v))
    #                       for k, v in sorted(dict(vars(params)).items())))
    # text = f'# Git Version: {get_code_version()} #'
    # logger.info("\n".join(['=' * 24, text, '=' * 24]))
    logger.info("The experiment will be stored in %s\n" % exp_folder)
    logger.info("Running command: %s" % command)
    logger.info("")
    return logger


cfg = cfg()
cfg.get_args()
cfgs = cfg.update_train_configs()

set_seed(cfgs.random_seed)
init_distributed_mode(args=cfgs)
rank = cfgs.rank

model = ChemGPT(cfgs)
# 统计模型的参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
'''
writer, logger = None, None

if rank == 0:
    logger = initialize_exp(cfgs)
    logger_path = get_dump_path(cfgs)
    cfgs.time_stamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    comment = f'bath_size={cfgs.batch_size} exp_id={cfgs.exp_id}'
    if not cfgs.no_tensorboard and not cfgs.only_test:
        writer = SummaryWriter(log_dir=os.path.join(logger_path, 'tensorboard', cfgs.time_stamp), comment=comment)
    
    cfgs.device = torch.device(cfgs.device)

    # ----------begin-------------
    torch.cuda.set_device(cfgs.gpu)


    runner = Runner(cfgs, writer, logger, rank)

    
    if cfgs.only_test:
        runner.test()
    else: 
        runner.run()

    # --------end--------
    if not cfgs.no_tensorboard and not cfgs.only_test and rank == 0:
        writer.close()
        logger.info("done!")

    if cfgs.dist and not cfgs.only_test:
        dist.barrier()
        dist.destroy_process_group()


'''
