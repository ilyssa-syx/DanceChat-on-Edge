from args import parse_train_opt
from EDGE import EDGE
import os


def train(opt):
    # 修复：将checkpoint路径传给EDGE
    checkpoint_path = ""
    if hasattr(opt, 'checkpoint') and opt.checkpoint and os.path.exists(opt.checkpoint):
        checkpoint_path = opt.checkpoint
        print(f"Loading model with checkpoint: {checkpoint_path}")
    else:
        print("Starting training from scratch")
    
    # 传入checkpoint路径
    model = EDGE(opt.feature_type, checkpoint_path=checkpoint_path)
    model.train_loop(opt)


if __name__ == "__main__":
    opt = parse_train_opt()
    train(opt)