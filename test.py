import glob
import os
from functools import cmp_to_key
from pathlib import Path
from tempfile import TemporaryDirectory
import random
import re
from collections import defaultdict

import jukemirlib
import numpy as np
import torch
from tqdm import tqdm

from args import parse_test_opt
from data.slice import slice_audio
from EDGE import EDGE
from data.audio_extraction.baseline_features import extract as baseline_extract
from data.audio_extraction.jukebox_features import extract as juke_extract


def key_func(path: str) -> int:
    """
    从文件名中提取 slice 后面的数字，用于排序。
    如果没匹配到，返回 0 作为默认。
    """
    basename = os.path.basename(path)
    match = re.search(r'slice(\d+)', basename)
    if match:
        return int(match.group(1))
    else:
        return 0


def stringintcmp_(a, b):
    aa = re.sub(r'_slice\d+.*$', '', os.path.basename(a).split('.')[0])
    bb = re.sub(r'_slice\d+.*$', '', os.path.basename(b).split('.')[0])
    ka, kb = key_func(a), key_func(b)
    if aa < bb:
        return -1
    if aa > bb:
        return 1
    if ka < kb:
        return -1
    if ka > kb:
        return 1
    return 0


stringintkey = cmp_to_key(stringintcmp_)

def get_song_name(filename):
    """
    从文件名中提取歌曲名称，保留 _slice{number} 部分，但去掉其他后缀
    """
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]
    
    # 找到 _slice{number} 的位置
    match = re.search(r'_slice\d+', name_without_ext)
    if match:
        # 返回到 _slice{number} 结束位置的部分
        return name_without_ext[:match.end()]
    else:
        return name_without_ext

def test(opt):
    feature_func = juke_extract if opt.feature_type == "jukebox" else baseline_extract
    sample_length = opt.out_length
    sample_size = int(sample_length / 2.5) - 1

    temp_dir_list = []
    all_cond_juke = []
    all_cond_beat = []
    all_cond_text = []
    all_filenames = []
    
    if opt.use_cached_features:
        print("Using precomputed features")
        
        # 根据你的扁平化文件结构加载特征
        jukebox_dir = os.path.join(opt.feature_cache_dir, "jukebox_feats")
        beat_dir = os.path.join(opt.feature_cache_dir, "beat_feats/npy") 
        text_dir = os.path.join(opt.feature_cache_dir, "text_encodings/npy")
        wav_dir = os.path.join(opt.feature_cache_dir, "wavs_sliced")
        
        juke_files = sorted(glob.glob(f"{jukebox_dir}/*.npy"), key=stringintkey)
        beat_files = sorted(glob.glob(f"{beat_dir}/*.npy"), key=stringintkey)
        text_files = sorted(glob.glob(f"{text_dir}/*.npy"), key=stringintkey)
        wav_files = sorted(glob.glob(f"{wav_dir}/*.wav"), key=stringintkey)
        
        print(f"Directory paths:")
        print(f"  jukebox_dir: {jukebox_dir}")
        print(f"  beat_dir: {beat_dir}")
        print(f"  text_dir: {text_dir}")
        print(f"  wav_dir: {wav_dir}")
        
        print(f"Found {len(juke_files)} juke files, {len(beat_files)} beat files, {len(text_files)} text files, {len(wav_files)} wav files")
        
        if len(juke_files) > 0:
            print(f"Sample juke files: {juke_files[:3]}")
        if len(beat_files) > 0:
            print(f"Sample beat files: {beat_files[:3]}")
        if len(text_files) > 0:
            print(f"Sample text files: {text_files[:3]}")
        if len(wav_files) > 0:
            print(f"Sample wav files: {wav_files[:3]}")
        
        # 确保所有文件数量一致
        if len(juke_files) == 0:
            print("❌ No jukebox feature files found!")
            print(f"   Check if directory exists: {os.path.exists(jukebox_dir)}")
            if os.path.exists(jukebox_dir):
                print(f"   Files in directory: {os.listdir(jukebox_dir)[:10]}")
            return
        
        if len(beat_files) == 0:
            print("❌ No beat feature files found!")
            print(f"   Check if directory exists: {os.path.exists(beat_dir)}")
            if os.path.exists(beat_dir):
                print(f"   Files in directory: {os.listdir(beat_dir)[:10]}")
            return
            
        if len(text_files) == 0:
            print("❌ No text encoding files found!")
            print(f"   Check if directory exists: {os.path.exists(text_dir)}")
            if os.path.exists(text_dir):
                print(f"   Files in directory: {os.listdir(text_dir)[:10]}")
            return
            
        if len(wav_files) == 0:
            print("❌ No wav files found!")
            print(f"   Check if directory exists: {os.path.exists(wav_dir)}")
            if os.path.exists(wav_dir):
                print(f"   Files in directory: {os.listdir(wav_dir)[:10]}")
            return
        
        assert len(juke_files) == len(beat_files) == len(text_files) == len(wav_files), f"File count mismatch: juke={len(juke_files)}, beat={len(beat_files)}, text={len(text_files)}, wav={len(wav_files)}"
        
        # 按歌曲名分组 -> 改为每个slice独立处理
        # songs_dict = defaultdict(lambda: {'juke': [], 'beat': [], 'text': [], 'wav': []})
        
        for juke_file, beat_file, text_file, wav_file in zip(juke_files, beat_files, text_files, wav_files):
            # 确保对应的文件是同一个slice
            juke_song = get_song_name(juke_file)
            beat_song = get_song_name(beat_file)
            text_song = get_song_name(text_file)
            wav_song = get_song_name(wav_file)
            
            # 调试信息
            if len(all_cond_juke) == 0:  # 只打印第一个文件的调试信息
                print(f"Debug first file:")
                print(f"  juke_file: {juke_file} -> song: {juke_song}")
                print(f"  beat_file: {beat_file} -> song: {beat_song}")
                print(f"  text_file: {text_file} -> song: {text_song}")
                print(f"  wav_file: {wav_file} -> song: {wav_song}")
            
            if not (juke_song == beat_song == text_song == wav_song):
                print(f"❌ File mismatch detected:")
                print(f"  juke_song: {juke_song}")
                print(f"  beat_song: {beat_song}")
                print(f"  text_song: {text_song}")
                print(f"  wav_song: {wav_song}")
                continue  # 跳过不匹配的文件，而不是崩溃
            
            
            # 每个slice作为独立的数据项处理
            print(f"Processing slice: {os.path.basename(juke_file)}")
            
            juke_cond = np.load(juke_file)
            beat_cond = np.load(beat_file) 
            text_cond = np.load(text_file)
            print('wav_file:', wav_file)
            # 每个slice单独添加到列表中，需要增加一个维度以符合模型输入格式
            all_cond_juke.append(torch.from_numpy(juke_cond[None, :]))  # [1, features]
            all_cond_beat.append(torch.from_numpy(beat_cond[None, :]))  # [1, features]  
            all_cond_text.append(torch.from_numpy(text_cond[None, :]))  # [1, features]
            all_filenames.append([wav_file])  # 单个文件也要放在列表中
        
        print(f"Total processed slices: {len(all_cond_juke)}")
    
    else:
        print("Computing features for input music")
        for wav_file in glob.glob(os.path.join(opt.music_dir, "*.wav")):
            # create temp folder (or use the cache folder if specified)
            if opt.cache_features:
                songname = os.path.splitext(os.path.basename(wav_file))[0]
                save_dir = os.path.join(opt.feature_cache_dir, songname)
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                dirname = save_dir
            else:
                temp_dir = TemporaryDirectory()
                temp_dir_list.append(temp_dir)
                dirname = temp_dir.name
            # slice the audio file
            print(f"Slicing {wav_file}")
            slice_audio(wav_file, 2.5, 5.0, dirname)
            file_list = sorted(glob.glob(f"{dirname}/*.wav"), key=stringintkey)
            # randomly sample a chunk of length at most sample_size
            rand_idx = random.randint(0, len(file_list) - sample_size)
            juke_cond_list = []
            beat_cond_list = []
            text_cond_list = []
            # generate features
            print(f"Computing features for {wav_file}")
            for idx, file in enumerate(tqdm(file_list)):
                # if not caching then only calculate for the interested range
                if (not opt.cache_features) and (not (rand_idx <= idx < rand_idx + sample_size)):
                    continue
                
                reps, _ = feature_func(file)
                # save reps
                if opt.cache_features:
                    featurename = os.path.splitext(file)[0] + ".npy"
                    np.save(featurename, reps)
                # if in the random range, put it into the list of reps we want
                # to actually use for generation
                if rand_idx <= idx < rand_idx + sample_size:
                    juke_cond_list.append(reps)
            
            juke_cond_tensor = torch.from_numpy(np.array(juke_cond_list))
            all_cond_juke.append(juke_cond_tensor)
            all_filenames.append(file_list[rand_idx : rand_idx + sample_size])

    model = EDGE(opt.feature_type, opt.checkpoint)
    model.eval()

    # directory for optionally saving the dances for eval
    fk_out = None
    if opt.save_motions:
        fk_out = opt.motion_save_dir

    print(f"Generating dances for {len(all_cond_juke)} slices")
    for i in range(len(all_cond_juke)):
        print('all_filenames[i]:', all_filenames[i])
        if opt.use_cached_features:
            data_tuple = (None, all_cond_juke[i], all_cond_beat[i], all_cond_text[i], all_filenames[i])
        else:
            # 如果是实时计算特征，只有jukebox特征
            data_tuple = (None, all_cond_juke[i], all_filenames[i])
        
        model.render_sample(
            data_tuple, "test", opt.render_dir, render_count=-1, fk_out=fk_out, render=not opt.no_render
        )
        print('finished 1 sample')
    print("Done")
    torch.cuda.empty_cache()
    for temp_dir in temp_dir_list:
        temp_dir.cleanup()


if __name__ == "__main__":
    opt = parse_test_opt()
    test(opt)