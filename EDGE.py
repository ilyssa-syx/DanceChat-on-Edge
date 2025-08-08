import multiprocessing
import os
import pickle
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dance_dataset import AISTPPDataset
from dataset.preprocess import increment_path
from model.adan import Adan
from model.diffusion import GaussianDiffusion
from model.model import DanceDecoder
from vis import SMPLSkeleton


def wrap(x):
    return {f"module.{key}": value for key, value in x.items()}


def maybe_wrap(x, num):
    return x if num == 1 else wrap(x)


class EDGE:
    def __init__(
        self,
        feature_type,
        checkpoint_path="",
        normalizer=None,
        EMA=True,
        learning_rate=4e-4,
        weight_decay=0.02,
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        state = AcceleratorState()
        num_processes = state.num_processes
        use_baseline_feats = feature_type == "baseline"

        pos_dim = 3
        rot_dim = 24 * 6  # 24 joints, 6dof
        self.repr_dim = repr_dim = pos_dim + rot_dim + 4

        feature_dim = 512

        horizon_seconds = 5
        FPS = 30
        self.horizon = horizon = horizon_seconds * FPS

        self.accelerator.wait_for_everyone()

        motiondiffuse_weights = torch.load("../text2motion/checkpoints/t2m/t2m_motiondiffuse/model/latest.tar")
        motiondiffuse = motiondiffuse_weights['encoder']


        checkpoint = None
        if checkpoint_path != "":
            if self.accelerator.is_main_process:
                print(f"Loading checkpoint from: {checkpoint_path}") # 只打印一次这句话
            checkpoint = torch.load(
                checkpoint_path, map_location=self.accelerator.device
            )
            self.normalizer = checkpoint["normalizer"]
            if self.accelerator.is_main_process:
                print("✅ Checkpoint loaded successfully")

        model = DanceDecoder(
            nfeats=repr_dim,
            seq_len=horizon,
            latent_dim=512,
            ff_size=1024,
            num_layers=8,
            num_heads=8,
            dropout=0.1,
            cond_feature_dim=feature_dim,
            activation=F.gelu,
        )

        

        smpl = SMPLSkeleton(self.accelerator.device)
        diffusion = GaussianDiffusion(
            model,
            horizon,
            repr_dim,
            smpl,
            schedule="cosine",
            n_timestep=1000,
            predict_epsilon=False,
            loss_type="l2",
            use_p2=False,
            cond_drop_prob=0.25,
            guidance_weight=2,
        )

        print(
            "Model has {} parameters".format(sum(y.numel() for y in model.parameters()))
        )

        self.model = self.accelerator.prepare(model)
        self.diffusion = diffusion.to(self.accelerator.device)
        optim = Adan(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.optim = self.accelerator.prepare(optim)

        if self.accelerator.is_main_process:
            print("🔧 Initializing text encoder with MotionDiffuse weights...")
        self.model.multi_modal_projector.load_text_encoder_weights(motiondiffuse)
        print('finished')
        # 🔧 修复2：改进模型状态加载，添加优化器状态加载
        if checkpoint_path != "":
            if self.accelerator.is_main_process:
                print("🔄 Loading model and optimizer states...")
            
            # 加载模型状态
            self.model.load_state_dict(
                maybe_wrap(
                    checkpoint["ema_state_dict" if EMA else "model_state_dict"],
                    num_processes,
                )
            )
            
            # 🔧 新增：加载优化器状态
            if "optimizer_state_dict" in checkpoint:
                try:
                    self.optim.load_state_dict(checkpoint["optimizer_state_dict"])
                    if self.accelerator.is_main_process:
                        print("✅ Optimizer state loaded")
                except Exception as e:
                    if self.accelerator.is_main_process:
                        print(f"⚠️  Could not load optimizer state: {e}")
            
            if self.accelerator.is_main_process:
                print("✅ Model checkpoint fully loaded!")

    def eval(self):
        self.diffusion.eval()

    def train(self):
        self.diffusion.train()

    def prepare(self, objects):
        return self.accelerator.prepare(*objects)

    def train_loop(self, opt):
        # load datasets
        train_tensor_dataset_path = os.path.join(
            opt.processed_data_dir, f"train_tensor_dataset.pkl"
        )
        test_tensor_dataset_path = os.path.join(
            opt.processed_data_dir, f"test_tensor_dataset.pkl"
        )
        if (
            not opt.no_cache
            and os.path.isfile(train_tensor_dataset_path)
            and os.path.isfile(test_tensor_dataset_path)
        ):
            train_dataset = pickle.load(open(train_tensor_dataset_path, "rb"))
            test_dataset = pickle.load(open(test_tensor_dataset_path, "rb"))
        else:
            train_dataset = AISTPPDataset(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=True,
                force_reload=opt.force_reload,
            )
            test_dataset = AISTPPDataset(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=False,
                normalizer=train_dataset.normalizer,
                force_reload=opt.force_reload,
            )
            # cache the dataset in case
            if self.accelerator.is_main_process:
                pickle.dump(train_dataset, open(train_tensor_dataset_path, "wb"))
                pickle.dump(test_dataset, open(test_tensor_dataset_path, "wb"))

        # set normalizer
        self.normalizer = test_dataset.normalizer

        # data loaders
        # decide number of workers based on cpu count
        num_cpus = multiprocessing.cpu_count()
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=min(int(num_cpus * 0.75), 32),
            pin_memory=True,
            drop_last=True,
        )
        test_data_loader = DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

        train_data_loader = self.accelerator.prepare(train_data_loader)
        # boot up multi-gpu training. test dataloader is only on main process
        load_loop = (
            partial(tqdm, position=1, desc="Batch")
            if self.accelerator.is_main_process
            else lambda x: x
        )
        if self.accelerator.is_main_process:
            save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
            opt.exp_name = save_dir.split("/")[-1]
            wandb.init(project=opt.wandb_pj_name, name=opt.exp_name)
            save_dir = Path(save_dir)
            wdir = save_dir / "weights"
            wdir.mkdir(parents=True, exist_ok=True)

        self.accelerator.wait_for_everyone()

        start_epoch = 1
        if hasattr(opt, 'checkpoint') and opt.checkpoint and os.path.exists(opt.checkpoint):
            try:
                checkpoint_filename = os.path.basename(opt.checkpoint)
                if 'train-' in checkpoint_filename:
                    epoch_num = int(checkpoint_filename.split('train-')[1].split('.')[0])
                    start_epoch = epoch_num + 1
                    if self.accelerator.is_main_process:
                        print(f"🔄 Resuming training from epoch {start_epoch} (loaded from {opt.checkpoint})")
            except Exception as e:
                if self.accelerator.is_main_process:
                    print(f"⚠️  Could not determine start epoch from checkpoint filename, starting from epoch 1: {e}")
        for epoch in range(start_epoch, opt.epochs + 1):
            avg_loss = 0
            avg_vloss = 0
            avg_fkloss = 0
            avg_footloss = 0
            avg_alignloss = 0
            # train
            self.train()
            for step, (x, cond1, cond2, cond3, filename, wavnames) in enumerate(
                load_loop(train_data_loader)
            ):
                emb1, emb2, emb3 = self.model.get_embeddings(cond1, cond2, cond3)
                total_loss, (loss, v_loss, fk_loss, foot_loss, align_loss) = self.diffusion(
                    x, cond1, cond2, cond3, emb1, emb2, emb3, t_override=None
                )
                self.optim.zero_grad()
                self.accelerator.backward(total_loss)

                self.optim.step()

                # ema update and train loss update only on main
                if self.accelerator.is_main_process:
                    avg_loss += loss.detach().cpu().numpy()
                    avg_vloss += v_loss.detach().cpu().numpy()
                    avg_fkloss += fk_loss.detach().cpu().numpy()
                    avg_footloss += foot_loss.detach().cpu().numpy()
                    avg_alignloss += align_loss.detach().cpu().numpy()

                    if step % opt.ema_interval == 0:
                        self.diffusion.ema.update_model_average(
                            self.diffusion.master_model, self.diffusion.model
                        )
            if (epoch % opt.save_interval) == 0:
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    self.eval()
                    # log
                    avg_loss /= len(train_data_loader)
                    avg_vloss /= len(train_data_loader)
                    avg_fkloss /= len(train_data_loader)
                    avg_footloss /= len(train_data_loader)
                    avg_alignloss /= len(train_data_loader)
                    log_dict = {
                        "Train Loss": avg_loss,
                        "V Loss": avg_vloss,
                        "FK Loss": avg_fkloss,
                        "Foot Loss": avg_footloss,
                        "Align Loss": avg_alignloss
                    }
                    wandb.log(log_dict)
                    ckpt = {
                        "ema_state_dict": self.diffusion.master_model.state_dict(),
                        "model_state_dict": self.accelerator.unwrap_model(
                            self.model
                        ).state_dict(),
                        "optimizer_state_dict": self.optim.state_dict(),
                        "normalizer": self.normalizer,
                    }
                    torch.save(ckpt, os.path.join(wdir, f"train-{epoch}.pt"))
                    
                    # 🔧 修复4：优化sample生成，避免内存问题
                    try:
                        render_count = 1  # 减少到1个sample
                        print(f"🎨 Generating Sample (epoch {epoch})...")
                        
                        # 清理GPU缓存
                        torch.cuda.empty_cache()
                        
                        # 获取测试数据
                        (x, cond1, cond2, cond3, filename, wavnames) = next(iter(test_data_loader))
                        
                        # 🔧 关键修复：使用EDGE的render_sample而不是diffusion的
                        # 构造数据元组，模仿test.py的方式
                        data_tuple = (
                            x[:render_count], 
                            cond1[:render_count], 
                            cond2[:render_count], 
                            cond3[:render_count], 
                            wavnames[:render_count]
                        )
                        
                        # 使用EDGE的render_sample方法，避免accelerator.device问题
                        self.render_sample(
                            data_tuple, 
                            f"epoch_{epoch}", 
                            os.path.join(opt.render_dir, "train_" + opt.exp_name), 
                            render_count=render_count
                        )
                        
                        print(f"✅ Sample generated successfully!")
                        
                    except Exception as e:
                        print(f"⚠️  Sample generation failed (training continues): {e}")
                        # 清理内存后继续
                        torch.cuda.empty_cache()
                    
                    print(f"💾 [MODEL SAVED at Epoch {epoch}]")
        
        if self.accelerator.is_main_process:
            wandb.run.finish()

    def render_sample(
        self, data_tuple, label, render_dir, render_count=-1, fk_out=None, render=True
    ):
        _, cond1, cond2, cond3, wavname = data_tuple
        
        if render_count < 0:
            render_count = len(cond1)
        shape = (render_count, self.horizon, self.repr_dim)
        cond1 = cond1.to(self.accelerator.device)
        cond2 = cond2.to(self.accelerator.device)
        cond3 = cond3.to(self.accelerator.device)
        print('wavname[:render_count]:', wavname[:render_count])
        self.diffusion.render_sample(
            shape,
            cond1[:render_count],
            cond2[:render_count],
            cond3[:render_count],
            self.normalizer,
            label,
            render_dir,
            name=wavname[:render_count],
            sound=True,
            mode="long",
            fk_out=fk_out,
            render=render
        )
