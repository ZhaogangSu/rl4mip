import json
import logging
import multiprocessing as mp
import os
import os.path as path
import pickle
from functools import partial
from typing import List
import hydra
import numpy as np
from omegaconf import DictConfig
from pandas import DataFrame
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import torch
from datetime import datetime

from rl4mip.DataGenerator.L2O.src import instance2graph as L2O_instance2graph
from rl4mip.DataGenerator.L2O.src import set_cpu_num as L2O_set_cpu_num
from rl4mip.DataGenerator.L2O.src import set_seed as L2O_set_seed
from rl4mip.DataGenerator.L2O.src import solve_instance as L2O_solve_instance

import rl4mip.DataGenerator.L2O.src.tb_writter as L2O_tb_writter
from rl4mip.DataGenerator.L2O.src import G2MILP
from rl4mip.DataGenerator.L2O.src import Benchmark as L2O_Benchmark
from rl4mip.DataGenerator.L2O.src import Generator as L2O_Generator
from rl4mip.DataGenerator.L2O.src import InstanceDataset as L2O_InstanceDataset
from rl4mip.DataGenerator.L2O.src import Trainer as L2O_Trainer


from rl4mip.DataGenerator.ACM.src import instance2graph as ACM_instance2graph
from rl4mip.DataGenerator.ACM.src import set_cpu_num as ACM_set_cpu_num
from rl4mip.DataGenerator.ACM.src import set_seed as ACM_set_seed
from rl4mip.DataGenerator.ACM.src import solve_instance as ACM_solve_instance

import rl4mip.DataGenerator.ACM.src.tb_writter as ACM_tb_writter
from rl4mip.DataGenerator.ACM.src import ACMMILP
from rl4mip.DataGenerator.ACM.src import Benchmark as ACM_Benchmark
from rl4mip.DataGenerator.ACM.src import Generator as ACM_Generator
from rl4mip.DataGenerator.ACM.src import InstanceDataset as ACM_InstanceDataset
from rl4mip.DataGenerator.ACM.src import Trainer as ACM_Trainer


def format_logger():
    logging.basicConfig(
        level=logging.INFO, 
        format='[%(asctime)s][%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

class L20_MIPDataGen:
    # def __init__(self, para):
    #     self.para = para

    def preprocess_(self, file: str, config: DictConfig):
        """
        Preprocesses a single instance.

        Args:
            file: instance file name
            config: config
        """
        sample_path = path.join(config.paths.data_dir, file)
        data, features = L2O_instance2graph(sample_path, config.compute_features)

        with open(path.join(config.paths.dataset_samples_dir, os.path.splitext(file)[0]+".pkl"), "wb") as f:
            pickle.dump(data, f)
        if config.solve_instances:
            solving_results = {"instance": file}
            solving_results.update(L2O_solve_instance(sample_path))
        else:
            solving_results = None
        return features, solving_results

    def make_dataset_features(
            self, 
            features: List[dict],
            solving_results: List[dict],
            config: DictConfig
        ):
        """
        Computes the dataset features.

        Args:
            features: list of instance features
            solving_results: list of solving results
            config: config
        """
        if config.compute_features:
            logging.info(f"Writing instance features to: {config.paths.dataset_features_path}")
            features: DataFrame = DataFrame(features, columns=features[0].keys()).set_index("instance")
            features.to_csv(config.paths.dataset_features_path)

            logging.info(f"Writing dataset statistics to: {config.paths.dataset_stats_path}")
            stats = {
                "rhs_type": config.dataset.rhs_type,
                "rhs_min": np.min(features["rhs_min"]),
                "rhs_max": np.max(features["rhs_max"]),

                "obj_type": config.dataset.obj_type,
                "obj_min": np.min(features["obj_min"]),
                "obj_max": np.max(features["obj_max"]),
                
                "coef_type": config.dataset.lhs_type,
                "coef_min": np.min(features['lhs_min']),
                "coef_max": np.max(features["lhs_max"]),
                "coef_dens": np.mean(features["coef_dens"]),

                "cons_degree_min": int(np.min(features["cons_degree_min"])),
                "cons_degree_max": int(np.max(features["cons_degree_max"])),
            }
            with open(config.paths.dataset_stats_path, "w") as f:
                f.write(json.dumps(stats, indent=2))

        if config.solve_instances:
            logging.info(f"Writting solving results to: {config.paths.dataset_solving_path}")
            solving_results: DataFrame = DataFrame(solving_results, columns=solving_results[0].keys()).set_index("instance")
            solving_results.to_csv(config.paths.dataset_solving_path)

            solving_time = solving_results.loc[:, ["solving_time"]].to_numpy()
            num_nodes = solving_results.loc[:, ["num_nodes"]].to_numpy()
            
            logging.info(f"  mean solving time: {solving_time.mean()}")
            logging.info(f"  mean num nodes: {num_nodes.mean()}")

    def preprocess(self, config: DictConfig):
        """
        直接接受配置对象的预处理方法
        """
        L2O_set_seed(config.seed)
        L2O_set_cpu_num(config.num_workers + 1)

        logging.info("="*10 + "Begin preprocessing" + "="*10)
        logging.info(f"Dataset: {config.dataset.name}.")
        logging.info(f"Dataset dir: {config.paths.data_dir}")

        os.makedirs(config.paths.dataset_samples_dir, exist_ok=True)
        os.makedirs(config.paths.dataset_stats_dir, exist_ok=True)

        files: list = os.listdir(config.paths.data_dir)
        files.sort()
        if len(files) > config.dataset.num_train:
            files = files[:config.dataset.num_train]

        func = partial(self.preprocess_, config=config)
        with mp.Pool(config.num_workers) as pool:
            features, solving_results = zip(*list(tqdm(pool.imap(func, files), total=len(files), desc="Preprocessing")))
        logging.info(f"Preprocessed samples are saved to: {config.paths.dataset_samples_dir}")

        self.make_dataset_features(features, solving_results, config)
        
        logging.info("="*10 + "Preprocessing finished" + "="*10)
        


    def train(self, config: DictConfig):
        """
        Train G2MILP.
        """
        L2O_set_seed(config.seed)
        L2O_set_cpu_num(config.num_workers + 1)
        torch.cuda.set_device(config.cuda)
        L2O_tb_writter.set_logger(config.paths.tensorboard_dir)

        model = G2MILP.load_model(config)
        logging.info(f"Loaded model.")
        logging.info(
            f"  Number of model parameters: {sum([x.nelement() for x in model.parameters()]) / 1000}K.")

        train_set = L2O_InstanceDataset(
            data_dir=config.paths.dataset_samples_dir,
            solving_results_path=config.paths.dataset_solving_path,
        )
        logging.info(f"Loaded dataset.")
        logging.info(f"  Number of training instances: {len(train_set)}.")
    
        trainer = L2O_Trainer(
            model=model,
            train_set=train_set,
            paths=config.paths,
            config=config.trainer,
            generator_config=config.generator,
            benchmark_config=config.benchmarking,
        )

        logging.info("="*10 + "Begin training" + "="*10)

        trainer.train()

        logging.info("="*10 + "Training finished" + "="*10)

        # test
        for mask_ratio in [0.01, 0.05, 0.1]:
            config.generator.mask_ratio = mask_ratio

            # generate
            samples_dir = path.join(config.paths.train_dir,
                                    f"eta-{mask_ratio}/samples")
            generator = L2O_Generator(
                model=model,
                config=config.generator,
                templates_dir=config.paths.dataset_samples_dir,
                save_dir=samples_dir,
            )
            generator.generate()

            # benchmark
            benchmark_dir = path.join(
                config.paths.train_dir, f"eta-{mask_ratio}/benchmark")
            benchmarker = L2O_Benchmark(
                config=config.benchmarking,
                dataset_stats_dir=config.paths.dataset_stats_dir,
            )
            results = benchmarker.assess_samples(
                samples_dir=samples_dir,
                benchmark_dir=benchmark_dir
            )

            info_path = path.join(benchmark_dir, "info.json")
            benchmarker.log_info(
                generator_config=config.generator,
                benchmarking_config=config.benchmarking,
                meta_results=results,
                save_path=info_path,
            )


    def generate(self, config: DictConfig):
        """
        Generate instances using G2MILP.
        """
        L2O_set_seed(config.seed)
        L2O_set_cpu_num(config.num_workers + 1)
        torch.cuda.set_device(config.cuda)

        # model_path = path.join(config.paths.model_dir, "model_best.ckpt")
        # 数据生成修改代码
        model_path = path.join(config.paths.best_model_path, "model_best.ckpt")
        
        model = G2MILP.load_model(config, model_path)
        generator = L2O_Generator(
            model=model,
            config=config.generator,
            templates_dir=config.paths.dataset_samples_dir,
            save_dir=config.paths.samples_dir,
        )
        generator.generate()

        benchmarker = L2O_Benchmark(
            config=config.benchmarking,
            dataset_stats_dir=config.paths.dataset_stats_dir,
        )
        results = benchmarker.assess_samples(
            samples_dir=config.paths.samples_dir,
            benchmark_dir=config.paths.benchmark_dir,
        )

        info_path = path.join(config.paths.benchmark_dir, "info.json")
        benchmarker.log_info(
            generator_config=config.generator,
            benchmarking_config=config.benchmarking,
            meta_results=results,
            save_path=info_path,
        )



class ACM_MIPDataGen:
    def preprocess_(self, file: str, config: DictConfig):
        """
        Preprocesses a single instance.

        Args:
            file: instance file name
            config: config
        """
        sample_path = path.join(config.paths.data_dir, file)
        data, features, community_part = ACM_instance2graph(sample_path, config.compute_features, True, config.dataset.resolution)

        with open(path.join(config.paths.dataset_samples_dir, os.path.splitext(file)[0]+".pkl"), "wb") as f:
            pickle.dump(data, f)

        community_part = np.array(community_part, dtype=object)

        np.save(path.join(config.paths.community_info_dir, os.path.splitext(file)[0]+".npy"), community_part)

        if config.solve_instances:
            solving_results = {"instance": file}
            solving_results.update(ACM_solve_instance(sample_path))
        else:
            solving_results = None
        return features, solving_results

    def make_dataset_features(
            self,
            features: List[dict],
            solving_results: List[dict],
            config: DictConfig
        ):
        """
        Computes the dataset features.

        Args:
            features: list of instance features
            solving_results: list of solving results
            community_partition: list of community information
            config: config
        """
        if config.compute_features:
            logging.info(f"Writing instance features to: {config.paths.dataset_features_path}")
            features: DataFrame = DataFrame(features, columns=features[0].keys()).set_index("instance")
            features.to_csv(config.paths.dataset_features_path)

            logging.info(f"Writing dataset statistics to: {config.paths.dataset_stats_path}")
            stats = {
                "rhs_type": config.dataset.rhs_type,
                "rhs_min": np.min(features["rhs_min"]),
                "rhs_max": np.max(features["rhs_max"]),

                "obj_type": config.dataset.obj_type,
                "obj_min": np.min(features["obj_min"]),
                "obj_max": np.max(features["obj_max"]),
                
                "coef_type": config.dataset.lhs_type,
                "coef_min": np.min(features['lhs_min']),
                "coef_max": np.max(features["lhs_max"]),
                "coef_dens": np.mean(features["coef_dens"]),

                "cons_degree_min": int(np.min(features["cons_degree_min"])),
                "cons_degree_max": int(np.max(features["cons_degree_max"])),
            }
            with open(config.paths.dataset_stats_path, "w") as f:
                f.write(json.dumps(stats, indent=2))

        if config.solve_instances:
            logging.info(f"Writting solving results to: {config.paths.dataset_solving_path}")
            solving_results: DataFrame = DataFrame(solving_results, columns=solving_results[0].keys()).set_index("instance")
            solving_results.to_csv(config.paths.dataset_solving_path)

            solving_time = solving_results.loc[:, ["solving_time"]].to_numpy()
            num_nodes = solving_results.loc[:, ["num_nodes"]].to_numpy()
            
            logging.info(f"  mean solving time: {solving_time.mean()}")
            logging.info(f"  mean num nodes: {num_nodes.mean()}")

        
    def preprocess(self, config: DictConfig):
        """
        Preprocesses the dataset.
        """
        ACM_set_seed(config.seed)
        ACM_set_cpu_num(config.num_workers + 1)

        logging.info("="*10 + "Begin preprocessing" + "="*10)
        logging.info(f"Dataset: {config.dataset.name}.")
        logging.info(f"Dataset dir: {config.paths.data_dir}")

        os.makedirs(config.paths.dataset_samples_dir, exist_ok=True)
        os.makedirs(config.paths.dataset_stats_dir, exist_ok=True)
        os.makedirs(config.paths.community_info_dir, exist_ok=True)

        files: list = os.listdir(config.paths.data_dir)
        files.sort()
        if len(files) > config.dataset.num_train:
            files = files[:config.dataset.num_train]

        func = partial(self.preprocess_, config=config)
        with mp.Pool(config.num_workers) as pool:
            features, solving_results = zip(*list(tqdm(pool.imap(func, files), total=len(files), desc="Preprocessing")))
        logging.info(f"Preprocessed samples are saved to: {config.paths.dataset_samples_dir}")

        self.make_dataset_features(features, solving_results, config)
        
        logging.info("="*10 + "Preprocessing finished" + "="*10)

    def train(self, config: DictConfig):
        """
        Train ACMMILP.
        """
        ACM_set_seed(config.seed)
        ACM_set_cpu_num(config.num_workers + 1)
        torch.cuda.set_device(config.cuda)
        ACM_tb_writter.set_logger(config.paths.tensorboard_dir)

        model = ACMMILP.load_model(config)
        emb_model = ACMMILP.load_model(config)
        logging.info(f"Loaded model.")
        logging.info(
            f"  Number of model parameters: {sum([x.nelement() for x in model.parameters()]) / 1000}K.")

        train_set = ACM_InstanceDataset(
            data_dir=config.paths.dataset_samples_dir,
            community_dir=config.paths.community_info_dir,
            solving_results_path=config.paths.dataset_solving_path,
        )
        logging.info(f"Loaded dataset.")
        logging.info(f"  Number of training instances: {len(train_set)}.")
    
        trainer = ACM_Trainer(
            model=model,
            emb_model=emb_model,
            train_set=train_set,
            paths=config.paths,
            config=config.trainer,
            generator_config=config.generator,
            benchmark_config=config.benchmarking,
        )

        logging.info("="*10 + "Begin training" + "="*10)

        trainer.train()

        logging.info("="*10 + "Training finished" + "="*10)

        # test
        for mask_ratio in [0.05, 0.1, 0.2]:
            config.generator.mask_ratio = mask_ratio

            # generate
            samples_dir = path.join(config.paths.train_dir,
                                    f"eta-{mask_ratio}/samples")
            generator = ACM_Generator(
                model=model,
                emb_model=emb_model,
                config=config.generator,
                templates_dir=config.paths.dataset_samples_dir,
                community_dir=config.paths.community_info_dir,
                save_dir=samples_dir,
            )
            generator.generate()

            # benchmark
            benchmark_dir = path.join(
                config.paths.train_dir, f"eta-{mask_ratio}/benchmark")
            benchmarker = ACM_Benchmark(
                config=config.benchmarking,
                dataset_stats_dir=config.paths.dataset_stats_dir,
            )
            results = benchmarker.assess_samples(
                samples_dir=samples_dir,
                benchmark_dir=benchmark_dir
            )

            info_path = path.join(benchmark_dir, "info.json")
            benchmarker.log_info(
                generator_config=config.generator,
                benchmarking_config=config.benchmarking,
                meta_results=results,
                save_path=info_path,
            )

    def generate(self, config: DictConfig):
        """
        Generate instances using ACMMILP.
        """
        ACM_set_seed(config.seed)
        ACM_set_cpu_num(config.num_workers + 1)
        torch.cuda.set_device(config.cuda)

        # model_path = path.join(config.paths.model_dir, "model_best.ckpt")
        model_path = path.join(config.paths.best_model_path, "model_best.ckpt")
        model = ACMMILP.load_model(config, model_path)

        # emb_model_path = path.join(config.paths.model_dir, "emb_model_best.ckpt")
        emb_model_path = path.join(config.paths.best_model_path, "emb_model_best.ckpt")
        emb_model = ACMMILP.load_model(config, emb_model_path)

        generator = ACM_Generator(
            model=model,
            emb_model=emb_model,
            config=config.generator,
            templates_dir=config.paths.dataset_samples_dir,
            community_dir=config.paths.community_info_dir,
            save_dir=config.paths.samples_dir,
        )
        generator.generate()

        benchmarker = ACM_Benchmark(
            config=config.benchmarking,
            dataset_stats_dir=config.paths.dataset_stats_dir,
        )
        results = benchmarker.assess_samples(
            samples_dir=config.paths.samples_dir,
            benchmark_dir=config.paths.benchmark_dir,
        )

        info_path = path.join(config.paths.benchmark_dir, "info.json")
        benchmarker.log_info(
            generator_config=config.generator,
            benchmarking_config=config.benchmarking,
            meta_results=results,
            save_path=info_path,
        )




class MIPDataGen:
    def __init__(self, method, num_workers):
        self.method = method
        self.num_workers = num_workers
        format_logger()

    def preprocess(self, dataset='setcover', num_train=5):
        if self.method == "L2O":
            config_path  =  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'L2O/conf/preprocess.yaml')
            config = OmegaConf.load(config_path)
            config['num_workers'] = self.num_workers
            config['dataset']['name'] = dataset
            config['dataset']['num_train'] = num_train
            dataGenerator = L20_MIPDataGen()
            dataGenerator.preprocess(config)

        elif self.method == "ACM":
            config_path  =  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ACM/conf/preprocess.yaml')
            config = OmegaConf.load(config_path)
            config['num_workers'] = self.num_workers
            config['dataset']['name'] = dataset
            config['dataset']['num_train'] = num_train
            dataGenerator = ACM_MIPDataGen()
            dataGenerator.preprocess(config)



    def train(self, dataset='setcover', cuda=0,
              batch_size=16, batch_repeat_size=2, steps=4, save_start=3, save_step=1, num_samples=5,
              save_start_stage_1=3, save_start_stage_2=6, update_iters=2, update_iters_stage_2=1):
        if self.method == "L2O":
            config_path  =  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'L2O/conf/train.yaml')
            config = OmegaConf.load(config_path)
            now = datetime.now().strftime("%m-%d-%H:%M:%S")
            config['now'] = now

            config['job_name'] = dataset + '-default'
            config['cuda'] = cuda
            config['num_workers'] = self.num_workers
            config['dataset']['name'] = dataset
            
            config['trainer']['batch_size'] = batch_size
            config['trainer']['batch_repeat_size'] = batch_repeat_size
            config['trainer']['steps'] = steps
            config['trainer']['save_start'] = save_start
            config['trainer']['save_step'] = save_step

            config['generator']['num_samples'] = num_samples
            config['benchmarking']['num_samples'] = num_samples

            dataGenerator = L20_MIPDataGen()
            dataGenerator.train(config)

        elif self.method == "ACM":
            config_path  =  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ACM/conf/train.yaml')
            config = OmegaConf.load(config_path)
            now = datetime.now().strftime("%m-%d-%H:%M:%S")
            config['now'] = now

            config['job_name'] = dataset + '-default'
            config['cuda'] = cuda
            config['num_workers'] = self.num_workers
            config['dataset']['name'] = dataset
        

            config['trainer']['batch_size'] = batch_size
            config['trainer']['batch_repeat_size'] = batch_repeat_size
            config['trainer']['steps'] = steps
            config['trainer']['save_start_stage_1'] = save_start_stage_1
            config['trainer']['save_start_stage_2'] = save_start_stage_2
            config['trainer']['save_step'] = save_step
            config['trainer']['update_iters'] = update_iters
            config['trainer']['update_iters_stage_2'] = update_iters_stage_2

            config['generator']['num_samples'] = num_samples
            config['benchmarking']['num_samples'] = num_samples

            dataGenerator = ACM_MIPDataGen()
            dataGenerator.train(config)


    def generate(self, num_samples=5, mask_ratio=0.01):
        if self.method == "L2O":
            config_path  =  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'L2O/conf/generate.yaml')
            config = OmegaConf.load(config_path)
            now = datetime.now().strftime("%m-%d-%H:%M:%S")
            config['now'] = now
            config['num_workers'] = self.num_workers
            config['generator']['num_samples'] = num_samples
            config['generator']['mask_ratio'] = mask_ratio
            config['benchmarking']['num_samples'] = num_samples

            dataGenerator = L20_MIPDataGen()
            dataGenerator.generate(config)

        elif self.method == "ACM":
            config_path  =  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ACM/conf/generate.yaml')
            config = OmegaConf.load(config_path)
            now = datetime.now().strftime("%m-%d-%H:%M:%S")
            config['now'] = now
            config['num_workers'] = self.num_workers
            config['generator']['num_samples'] = num_samples
            config['generator']['mask_ratio'] = mask_ratio
            config['benchmarking']['num_samples'] = num_samples

            dataGenerator = ACM_MIPDataGen()
            dataGenerator.generate(config)
