import torch.nn.functional as F
import clip
import copy
import torch.nn as nn
import time
import logging
import json
import gc

from torch import max, eq, no_grad, unsqueeze
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data import RandomSampler
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
from PIL import Image

from data.datasets import *
#from data.old_datasets import get_dataloaders, get_wild_dataloaders
from src.splitter import NonIIDSplitter
from src.server import *
from src.serverbase import *
from src.client import *
from src.clientbase import *
from src.utils import *
from src.hparams_registry import *
from src.splitter import RandomSplitter
from utils.logger import headline
from utils.options import args_parser
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader

"""
The main file function:
1. Load the hyperparameter dict.
2. Initialize logger
3. Initialize data (preprocess, data splits, etc.)
4. Initialize clients. 
5. Initialize Server.
6. Register clients at the server.
7. Start the server.
"""


def main(seed, seed1, iid, num_shards):
    args = args_parser()
    config_file = args.config_file
    with open(config_file) as cf:
        hyparam = json.load(cf)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_config = hyparam["global"]
    server_config = hyparam["server"]
    client_config = hyparam["client"]
    dataset_config = hyparam["dataset"]

    # Preprocess some hyperparameters
    server_config.update({
        'backbone': global_config['backbone'], 'num_channels': global_config['num_channels'],
        'text_feature_dim': global_config['text_feature_dim'], 'dataset_name': dataset_config["name"],
        'n_classes': dataset_config['num_classes'], 'real_scene': args.real_scene
    })
    client_config.update({
        'num_rounds': server_config["num_rounds"], 'text_feature_dim': global_config['text_feature_dim'],
        'dataset_name': dataset_config["name"], 'n_classes': dataset_config['num_classes'], 'real_scene': args.real_scene
    })
    root_dir = os.path.join(args.dataset_path, 'resources')
    num_shards = global_config['num_clients']
    if dataset_config['name'] == "PACS" or dataset_config['name'] == "OfficeHome" or dataset_config['name'] == "VLCS":
        for test_envs in sorted(dataset_config['domain_names']):
            args.hparams_seed = seed
            if args.hparams_seed == 0:
                hparams = default_hparams(global_config['method'], dataset_config['name'])
            else:
                hparams = random_hparams(global_config['method'], dataset_config['name'], seed_hash(args.hparams_seed))
            if args.hparams:
                hparams.update(json.loads(args.hparams))

            set_seeds(seed1, True)
            if not args.real_scene:
                output_path, train_output, save_path = set_output(args, seed1, dataset_config["name"], "train_log",
                                                                  test_envs, global_config['method'])
                _, val_output, _ = set_output(args, seed1, dataset_config["name"], "val_log", test_envs, global_config['method'])
                _, test_output, _ = set_output(args, seed1, dataset_config["name"], "test_log", test_envs, global_config['method'])
            else:
                output_path, train_output, save_path = set_real_output1(args, seed1, dataset_config["name"], "train_log",
                                                                       iid, num_shards, test_envs, global_config['method'])
                _, val_output, _ = set_real_output1(args, seed1, dataset_config["name"], "val_log", iid, num_shards,
                                                   test_envs, global_config['method'])
                _, test_output, _ = set_real_output1(args, seed1, dataset_config["name"], "test_log", iid, num_shards,
                                                    test_envs, global_config['method'])
            # Initialize logger
            # modify log_path to contain current time
            log_path = save_path.replace('checkpoints', 'logs')
            server_config.update({'log_path': log_path})
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            logger = logging.getLogger(__name__)
            logging.basicConfig(
                filename=os.path.join(log_path, "FL.log"),
                level=logging.INFO,
                format="[%(levelname)s](%(asctime)s) %(message)s",
                datefmt="%Y/%m/%d/ %I:%M:%S %p")

            # display and log experiment configuration
            message = "\n[WELCOME] Unfolding configurations...!"
            logging.info(message)
            logging.info(hyparam)
            logging.info(hparams)

            ## Loading datasets
            source_names = [x for x in sorted(dataset_config['domain_names']) if x != test_envs]
            split_scheme_first_part = ''.join([name[0].lower() for name in source_names])
            split_scheme = f'{split_scheme_first_part}-{test_envs[0].lower()}'

            server_config.update({
                'save_path': save_path, 'train_output': train_output, 'val_output': val_output, 'test_output': test_output,
                'class_names': dataset_config['class_names'], 'split_scheme': split_scheme, 'target': test_envs
            })
            client_config.update({'save_path': save_path, 'split_scheme': split_scheme, 'target': test_envs})

            dataset = eval(dataset_config['name'])(version='1.0', root_dir=root_dir, download=False,
                                                   split_scheme=split_scheme)
            grouper = CombinatorialGrouper(dataset=dataset, groupby_fields=dataset.groupby_fields)

            _, val_dataloader = dataset.get_DG_dataset_dataloader('val', sorted(dataset_config['domain_names']), source_names,
                                                                dataset.val_transform, int(hparams['batch_size']))
            test_dataset = dataset.get_subset('test', frac=1.0, transform=dataset.val_transform)
            test_dataloader = get_eval_loader(loader='standard', dataset=test_dataset, batch_size=int(hparams['batch_size']))

            if args.real_scene:
                if global_config["method"] == 'MCGDM':
                    strong_transforms = transforms.Compose([
                        transforms.Resize((dataset.input_shape[1], dataset.input_shape[2])),
                        transforms.RandomResizedCrop(dataset.input_shape[1], scale=(0.7, 1.0)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                        transforms.RandomGrayscale(),
                        transforms.RandAugment(),  # add this
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    train_dataset = dataset.MCGDM_get_subset('train', transform=dataset.train_transform,
                                                             strong_transform=strong_transforms)
                    if num_shards == 1:
                        train_datasets = [train_dataset]
                    elif num_shards > 1:
                        train_datasets = NonIIDSplitter(num_shards=num_shards, iid=iid, seed=seed1).MCGDM_split(
                            dataset.get_subset('train'), dataset.groupby_fields, transform=dataset.train_transform,
                            strong_transform=strong_transforms)
                    else:
                        raise ValueError("num_shards should be greater or equal to 1, we got {}".format(num_shards))
                else:
                    train_dataset = dataset.get_subset('train', transform=dataset.train_transform)
                    if num_shards == 1:
                        train_datasets = [train_dataset]
                    elif num_shards > 1:
                        train_datasets = NonIIDSplitter(num_shards=num_shards, iid=iid, seed=seed1).split(
                            dataset.get_subset('train'), dataset.groupby_fields, transform=dataset.train_transform)
                    else:
                        raise ValueError("num_shards should be greater or equal to 1, we got {}".format(num_shards))

                # initialize client
                clients = []
                #length = []
                for client_idx in tqdm(range(len(train_datasets)), leave=False):
                    train_dataloader = torch.utils.data.DataLoader(train_datasets[client_idx],
                                                                   batch_size=int(hparams['batch_size']), shuffle=True,
                                                                   num_workers=2, pin_memory=True)
                    #length.append(len(train_dataloader.dataset))
                    if client_config["algorithm"] == 'IRMClient':
                        client = eval(client_config["algorithm"])(device, seed1, client_idx, grouper, train_dataloader,
                                                                  client_config, hparams)
                    elif client_config["algorithm"] == 'GroupDROClient':
                        client = eval(client_config["algorithm"])(device, seed1, client_idx, grouper, dataset.metadata_array,
                                                                  train_dataloader, client_config, hparams)
                    else:
                        client = eval(client_config["algorithm"])(device, seed1, client_idx, train_dataloader,
                                                                  client_config, hparams)
                    clients.append(client)
                #print(length)
            else:
                if global_config["method"] == 'MCGDM':
                    strong_transforms = transforms.Compose([
                            transforms.Resize((dataset.input_shape[1],dataset.input_shape[2])),
                            transforms.RandomResizedCrop(dataset.input_shape[1], scale=(0.7, 1.0)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                            transforms.RandomGrayscale(),
                            transforms.RandAugment(),  # add this
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
                    _, train_dataloaders = dataset.MCGDM_get_DG_dataset_dataloader('train',
                                                                             sorted(dataset_config['domain_names']),
                                                                             source_names, dataset.train_transform,
                                                                             int(hparams['batch_size']), strong_transforms)
                else:
                    _, train_dataloaders = dataset.get_DG_dataset_dataloader('train', sorted(dataset_config['domain_names']),
                                                                             source_names, dataset.train_transform,
                                                                             int(hparams['batch_size']))
                # initialize client
                clients = []
                #length = []
                for client_idx in tqdm(range(len(train_dataloaders)), leave=False):
                    #length.append(len(train_dataloaders[client_idx].dataset))
                    if client_config["algorithm"] == 'GroupDROClient':
                        client = eval(client_config["algorithm"])(device, seed1, client_idx, grouper, dataset.metadata_array,
                                                                  train_dataloaders[client_idx], client_config, hparams)
                    else:
                        client = eval(client_config["algorithm"])(device, seed1, client_idx, train_dataloaders[client_idx],
                                                                  client_config, hparams)
                    clients.append(client)
                #print(length)
            message = f"successfully initialize all clients!"
            logging.info(message)
            del message; gc.collect()

            headline("train_step", clients[0].loss_names, train_output)
            headline("val_step", clients[0].loss_names[1], val_output)
            headline("test_step", clients[0].loss_names, test_output)

            central_server = eval(server_config["algorithm"])(device, seed1, server_config, hparams)

            if server_config['algorithm'] == "FedDGServer":
                central_server.set_amploader(global_dataloader)
            if args.resume_file:
                start_round = torch.load(args.resume_file)['round']
            else:
                start_round = 0

            central_server.setup_model(args.resume_file, start_round)

            #if not True:
            central_server.register_clients(clients)
            central_server.register_testloader({
                "val": val_dataloader[0],
                "test": test_dataloader})

            # do federated learning
            central_server.fit()

            # output the best result
            best_test_acc = torch.load(save_path + '/checkpoint_best.pt')['best_test_acc']
            best_round = torch.load(save_path + '/checkpoint_best.pt')['round']
            total_log = "the final result: best_round={:04d}, best_test_acc={:.4f}".format(best_round, best_test_acc)
            Print(total_log, test_output, timestamp=False)

            if not train_output == sys.stdout: train_output.close()
            if not val_output == sys.stdout: val_output.close()
            if not test_output == sys.stdout: test_output.close()
            message = "...done all learning process!\n...exit program!"
            logging.info(message)

    elif dataset_config['name'] == "iWildCam" or dataset_config['name'] == "Camelyon17" or dataset_config['name'] == \
            "FMoW":
        #num_shards = 100
        #iids = [0.5]
        split_scheme = f'0.5_50'
        target = f"{dataset_config['name'].lower()}_{dataset_config['name'].lower()}"
        #for iid in iids:
        args.hparams_seed = seed
        if args.hparams_seed == 0:
            hparams = default_hparams(global_config['method'], dataset_config['name'])
        else:
            hparams = random_hparams(global_config['method'], dataset_config['name'],
                                     seed_hash(args.hparams_seed))
        if args.hparams:
            hparams.update(json.loads(args.hparams))

        set_seeds(seed1, True)

        output_path, train_output, save_path = set_real_output(args, seed1, dataset_config["name"], "train_log", iid, num_shards,
                                                               dataset_config["name"].lower(),  dataset_config["name"].lower(),
                                                               global_config['method'])
        _, val_output, _ = set_real_output(args, seed1, dataset_config["name"], "val_log", iid, num_shards,
                                           dataset_config["name"].lower(), dataset_config["name"].lower(),
                                           global_config['method'])
        _, test_output, _ = set_real_output(args, seed1, dataset_config["name"], "test_log", iid, num_shards,
                                            dataset_config["name"].lower(), dataset_config["name"].lower(),
                                            global_config['method'])

        # Initialize logger
        # modify log_path to contain current time
        log_path = save_path.replace('checkpoints', 'logs')
        server_config.update({'log_path': log_path})

        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logger = logging.getLogger(__name__)
        logging.basicConfig(
            filename=os.path.join(log_path, "FL.log"),
            level=logging.INFO,
            format="[%(levelname)s](%(asctime)s) %(message)s",
            datefmt="%Y/%m/%d/ %I:%M:%S %p")

        # display and log experiment configuration
        message = "\n[WELCOME] Unfolding configurations...!"
        logging.info(message)
        logging.info(hyparam)
        logging.info(hparams)

        server_config.update({
            'save_path': save_path, 'train_output': train_output, 'val_output': val_output,
            'test_output': test_output, 'target':  target, 'source_domains': [], "class_names": [],
            'split_scheme': split_scheme,
        })
        client_config.update({'save_path': save_path, 'target': target})

        if dataset_config['name'].lower() == 'iwildcam':
            dataset = IWildCam(version='2.0', root_dir=root_dir, download=False)
        elif dataset_config['name'].lower() == 'camelyon17':
            dataset = Camelyon17(version='1.0', root_dir=root_dir, download=False)
        elif dataset_config['name'].lower() == 'fmow':
            dataset = FMoW(version='1.1', root_dir=root_dir, download=False)
        grouper = CombinatorialGrouper(dataset=dataset, groupby_fields=dataset.groupby_fields)

        if global_config["method"] == 'MCGDM':
            strong_transforms = transforms.Compose([
                transforms.Resize((dataset.input_shape[1], dataset.input_shape[2])),
                transforms.RandomResizedCrop(dataset.input_shape[1], scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                transforms.RandomGrayscale(),
                transforms.RandAugment(),  # add this
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            train_dataset = dataset.MCGDM_get_subset('train', transform=dataset.train_transform,
                                                     strong_transform=strong_transforms)
            if num_shards == 1:
                train_datasets = [train_dataset]
            elif num_shards > 1:
                train_datasets = NonIIDSplitter(num_shards=num_shards, iid=iid, seed=seed1).MCGDM_split(
                    dataset.get_subset('train'), dataset.groupby_fields, transform=dataset.train_transform,
                    strong_transform=strong_transforms)
            else:
                raise ValueError("num_shards should be greater or equal to 1, we got {}".format(num_shards))
        else:
            train_dataset = dataset.get_subset('train', transform=dataset.train_transform)
            if num_shards == 1:
                train_datasets = [train_dataset]
            elif num_shards > 1:
                train_datasets = NonIIDSplitter(num_shards=num_shards, iid=iid, seed=seed1).split(
                    dataset.get_subset('train'),
                    dataset.groupby_fields, transform=dataset.train_transform)
            else:
                raise ValueError("num_shards should be greater or equal to 1, we got {}".format(num_shards))
        #try:
        #    in_test_dataset = dataset.get_subset('id_test', transform=dataset.val_transform)
        #except ValueError:
        #    in_test_dataset, total_subset = RandomSplitter(ratio=0.2, seed=seed1).split(train_dataset)
        #    in_test_dataset.transform = dataset.val_transform

        out_val_dataset = dataset.get_subset('val', transform=dataset.val_transform)
        #try:
        #    in_val_dataset = dataset.get_subset('id_val', transform=dataset.val_transform)
        #except ValueError:
        #    in_validation_dataset, in_test_dataset = RandomSplitter(ratio=0.5, seed=seed1).split(train_dataset)
        #    in_validation_dataset.transform = dataset.val_transform
        #    in_test_dataset.transform = dataset.val_transform

        out_test_dataset = dataset.get_subset('test', transform=dataset.val_transform)
        out_test_dataloader = get_eval_loader(loader='standard', dataset=out_test_dataset,
                                              batch_size=int(hparams['batch_size']))
        #in_test_dataloader = get_eval_loader(loader='standard', dataset=in_test_dataset,
        #                                     batch_size=int(hparams['batch_size']))
        out_val_dataloader = get_eval_loader(loader='standard', dataset=out_val_dataset,
                                             batch_size=int(hparams['batch_size']))
        #in_val_dataloader = get_eval_loader(loader='standard', dataset=in_val_dataset,
        #                                    batch_size=int(hparams['batch_size']))

        # if not True:
        # initialize client
        #length = []
        clients = []
        for client_idx in tqdm(range(num_shards), leave=False):
            train_dataloader = get_train_loader('standard', train_datasets[client_idx],
                                                batch_size=int(hparams['batch_size']))
            #length.append(len(train_dataloader.dataset))
            if client_config["algorithm"] == 'IRMClient':
                client = eval(client_config["algorithm"])(device, seed1, client_idx, grouper, train_dataloader,
                                                          client_config, hparams)
            elif client_config["algorithm"] == 'GroupDROClient':
                client = eval(client_config["algorithm"])(device, seed1, client_idx, grouper, dataset.metadata_array,
                                                          train_dataloader, client_config, hparams)
            else:
                client = eval(client_config["algorithm"])(device, seed1, client_idx, train_dataloader,
                                                          client_config, hparams)
            clients.append(client)
        message = f"successfully initialize all clients!"
        logging.info(message)
        del message; gc.collect()

        headline("train_step", clients[0].loss_names, train_output)
        headline("val_step", clients[0].loss_names[1], val_output)
        headline("test_step", clients[0].loss_names, test_output)

        # initialize server (model should be initialized in the server. )
        central_server = eval(server_config["algorithm"])(device, seed1, server_config, hparams)

        if server_config['algorithm'] == "FedDGServer":
            central_server.set_amploader(global_dataloader)
        if args.resume_file:
            start_round = torch.load(save_path + '/checkpoint_latest.pt')['round']
        else:
            start_round = 0

        central_server.setup_model(args.resume_file, start_round)

        # if not True:
        central_server.register_clients(clients)
        central_server.register_testloader({
            #"in_val": in_val_dataloader,
            "val": out_val_dataloader,
            #"in_test": in_test_dataloader,
            "test": out_test_dataloader})

        # do federated learning
        central_server.fit()

        # output the best result
        best_test_acc = torch.load(save_path + '/checkpoint_best.pt')['best_test_acc']
        best_round = torch.load(save_path + '/checkpoint_best.pt')['round']
        total_log = "the final result: best_round={:04d}, best_test_acc={:.4f}".format(best_round,
                                                                                            best_test_acc)
        Print(total_log, test_output, timestamp=False)
        if dataset_config['name'] == 'iWildCam':
            best_test_f1_score = torch.load(save_path + '/checkpoint_best_f1_score.pt')['best_test_f1_score']
            best_round = torch.load(save_path + '/checkpoint_best_f1_score.pt')['round']
            total_log = "the final result: best_round={:04d}, best_test_f1_score={:.4f}".format(best_round,
                                                                                                best_test_f1_score)
        if dataset_config['name'] == 'Camelyon17':
            best_test_group_acc = torch.load(save_path + '/checkpoint_best_group_acc.pt')['best_test_group_acc']
            best_round = torch.load(save_path + '/checkpoint_best_group_acc.pt')['round']
            total_log = "the final result: best_round={:04d}, best_test_group_acc={:.4f}".format(best_round,
                                                                                                best_test_group_acc)
        if dataset_config['name'] == 'FMoW':
            best_test_group_acc = torch.load(save_path + '/checkpoint_best_group_acc.pt')['best_test_group_acc']
            best_round = torch.load(save_path + '/checkpoint_best_group_acc.pt')['round']
            total_log = "the final result: best_round={:04d}, best_test_group_acc={:.4f}".format(best_round,
                                                                                                best_test_group_acc)
            Print(total_log, test_output, timestamp=False)
            best_test_wc_acc = torch.load(save_path + '/checkpoint_best_wc_acc.pt')['best_test_wc_acc']
            best_round = torch.load(save_path + '/checkpoint_best_wc_acc.pt')['round']
            total_log = "the final result: best_round={:04d}, best_test_wc_acc={:.4f}".format(best_round,
                                                                                                best_test_wc_acc)
        Print(total_log, test_output, timestamp=False)
        if not train_output == sys.stdout: train_output.close()
        if not val_output == sys.stdout: val_output.close()
        if not test_output == sys.stdout: test_output.close()

    else:
        num_shards = global_config['num_clients']
        #iids = [0.5]
        #for iid in iids:
        for test_envs in sorted(dataset_config['domain_names']):

            source_names = [x for x in sorted(dataset_config['domain_names']) if x != test_envs]
            for out_val_envs in source_names:
                train_source_name = [x for x in sorted(source_names) if x != out_val_envs]

                args.hparams_seed = seed
                if args.hparams_seed == 0:
                    hparams = default_hparams(global_config['method'], dataset_config['name'])
                else:
                    hparams = random_hparams(global_config['method'], dataset_config['name'],
                                             seed_hash(args.hparams_seed))
                if args.hparams:
                    hparams.update(json.loads(args.hparams))

                set_seeds(seed1, True)

                output_path, train_output, save_path = set_real_output(args, seed1, dataset_config["name"], "train_log", iid,
                                                                       num_shards, out_val_envs, test_envs,
                                                                       global_config['method'])
                _, val_output, _ = set_real_output(args, seed1, dataset_config["name"], "val_log", iid, num_shards,
                                                   out_val_envs, test_envs, global_config['method'])
                _, test_output, _ = set_real_output(args, seed1, dataset_config["name"], "test_log", iid, num_shards,
                                                    out_val_envs, test_envs, global_config['method'])

                # Initialize logger
                # modify log_path to contain current time
                log_path = save_path.replace('checkpoints', 'logs')
                server_config.update({'log_path': log_path})
                if not os.path.exists(log_path):
                    os.makedirs(log_path)
                logger = logging.getLogger(__name__)
                logging.basicConfig(
                    filename=os.path.join(log_path, "FL.log"),
                    level=logging.INFO,
                    format="[%(levelname)s](%(asctime)s) %(message)s",
                    datefmt="%Y/%m/%d/ %I:%M:%S %p")

                # display and log experiment configuration
                message = "\n[WELCOME] Unfolding configurations...!"
                logging.info(message)
                logging.info(hyparam)
                logging.info(hparams)

                ## Loading datasets
                # Construct the first part of the split_scheme using a loop
                split_scheme_first_part = ''.join([name[0].lower() for name in train_source_name])
                # Combine the first part with the test environment
                split_scheme = f'{split_scheme_first_part}-{out_val_envs[0].lower()}-{test_envs[0].lower()}'

                server_config.update({
                    'save_path': save_path, 'train_output': train_output, 'val_output': val_output,
                    'test_output': test_output,
                    'source_domains': source_names, 'class_names': dataset_config['class_names'],
                })
                client_config.update({'save_path': save_path, 'target': test_envs})

                dataset = eval(dataset_config['name'])(version='1.0', root_dir=root_dir, download=False,
                                                       split_scheme=split_scheme)
                grouper = CombinatorialGrouper(dataset=dataset, groupby_fields=dataset.groupby_fields)

                train_dataset = dataset.get_subset('train', transform=dataset.train_transform)
                #in_test_dataset = dataset.get_subset('id_test', transform=dataset.val_transform)
                out_val_dataset = dataset.get_subset('val', transform=dataset.val_transform)
                #in_val_dataset = dataset.get_subset('id_val', transform=dataset.val_transform)
                out_test_dataset = dataset.get_subset('test', transform=dataset.val_transform)
                out_test_dataloader = get_eval_loader(loader='standard', dataset=out_test_dataset,
                                                      batch_size=int(hparams['batch_size']))
                #in_test_dataloader = get_eval_loader(loader='standard', dataset=in_test_dataset,
                #                                     batch_size=int(hparams['batch_size']))
                out_val_dataloader = get_eval_loader(loader='standard', dataset=out_val_dataset,
                                                     batch_size=int(hparams['batch_size']))
                #in_val_dataloader = get_eval_loader(loader='standard', dataset=in_val_dataset,
                #                                    batch_size=int(hparams['batch_size']))

                if num_shards == 1:
                    train_datasets = [train_dataset]
                elif num_shards > 1:
                    train_datasets = NonIIDSplitter(num_shards=num_shards, iid=iid, seed=seed1).split(
                        dataset.get_subset('train'), dataset.groupby_fields, transform=dataset.train_transform)
                else:
                    raise ValueError("num_shards should be greater or equal to 1, we got {}".format(num_shards))

                #if not True:
                # initialize client
                clients = []
                length = []
                for client_idx in tqdm(range(num_shards), leave=False):
                    train_dataloader = get_train_loader('standard', train_datasets[client_idx],
                                                        batch_size=int(hparams['batch_size']))
                    length.append(len(train_dataloader.dataset))
                    if client_config["algorithm"] == 'IRMClient' or client_config["algorithm"] == 'MMDClient':
                        client = eval(client_config["algorithm"])(device, seed1, client_idx, grouper,
                                                                  train_dataloader,
                                                                  client_config, hparams)
                    elif client_config["algorithm"] == 'GroupDROClient':
                        client = eval(client_config["algorithm"])(device, seed1, client_idx, grouper,
                                                                  dataset.metadata_array,
                                                                  train_dataloader, client_config, hparams)
                    else:
                        client = eval(client_config["algorithm"])(device, seed1, client_idx,
                                                                  train_dataloader,
                                                                  client_config, hparams)
                    clients.append(client)
                print(length)
                """
                PACS: 
                num_shards=2, [1517, 3556]
                num_shards=3, [1605, 1735, 1733]
                num_shards=50, [102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 103, 103, 
                102, 102, 102, 102, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101,
                101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101]
                num_shards=100, [57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 
                57, 57, 57, 56, 56, 56, 56, 56, 56, 56, 55, 55, 55, 55, 55, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 
                57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 
                57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 
                57, 57]]
                """
                message = f"successfully initialize all clients!"
                logging.info(message)
                del message; gc.collect()

                headline("train_step", clients[0].loss_names, train_output)
                headline("val_step", clients[0].loss_names[1], val_output)
                headline("test_step", clients[0].loss_names, test_output)

                # initialize server (model should be initialized in the server. )
                central_server = eval(server_config["algorithm"])(device, seed1, server_config, hparams)

                if server_config['algorithm'] == "FedDGServer":
                    central_server.set_amploader(global_dataloader)
                if args.resume_file:
                    start_round = torch.load(save_path + '/checkpoint_latest.pt')['round']
                else:
                    start_round = 0

                #if global_config['text_flag']:
                #    text_features = central_server.text_feature()
                #    central_server.setup_model(args.resume_file, start_round, text_features["text_features_ems"])

                #else:
                central_server.setup_model(args.resume_file, start_round)

                #if not True:
                central_server.register_clients(clients)
                central_server.register_testloader({
                    #"in_val": in_val_dataloader,
                    "val": out_val_dataloader,
                    #"in_test": in_test_dataloader,
                    "test": out_test_dataloader})
                # do federated learning
                #if global_config['text_flag']:
                #    central_server.fit(text_features)
                #else:
                central_server.fit()

                # output the best result
                best_test_accuracy = torch.load(save_path + '/checkpoint_best.pt')['best_test_acc']
                best_round = torch.load(save_path + '/checkpoint_best.pt')['round']
                total_log = "the final result: best_round={:04d}, best_test_accuracy={:.4f}".format(best_round,
                                                                                                    best_test_accuracy)
                Print(total_log, test_output, timestamp=False)
                if not train_output == sys.stdout: train_output.close()
                if not val_output == sys.stdout: val_output.close()
                if not test_output == sys.stdout: test_output.close()
    message = "...done all learning process!\n...exit program!"
    logging.info(message)




if __name__ == '__main__':
    for iid in [0]:
        for i in range(6, 7):
            for seed in range(0, 1):
                for num_shards in [3]:
                    main(i, seed, iid, num_shards)
    time.sleep(3)
    exit()
