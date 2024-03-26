# -*- coding: utf-8 -*-
import click
import logging
import os
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor, AutoModel, HubertModel, Wav2Vec2FeatureExtractor
import soundfile as sf

from torch.utils.data import DataLoader



class UASpeechDataset(Dataset):
    def __init__(self, metadata_path, raw_data_dir, model_path, transform=None, target_transform=None):
        self.metadata = pd.read_csv(metadata_path)
        self.processor = AutoFeatureExtractor.from_pretrained(model_path)
        self.raw_data_dir  = raw_data_dir

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # comment when working on binary data without augmentation
        curr_audio_file_path = self.metadata.loc[self.metadata.index[idx], 'path']
        curr_audio_file_path = curr_audio_file_path.replace("/l/users/rzan.alhaddad/noisereduce", "/UASpeech/audio/noisereduce")
        if self.metadata.loc[self.metadata.index[idx], 'split'] != "B2":
            curr_audio_file_path = curr_audio_file_path.replace("noisereduce", "noisereduce_aug")
        audio_path = "./"+self.raw_data_dir +  curr_audio_file_path

        #uncomment when working on binary data without augmentation
        # audio_path = "./"+self.raw_data_dir +  self.metadata.loc[self.metadata.index[idx], 'path']
        speech, samplerate = sf.read(audio_path)
        # seconds = librosa.get_duration(path= audio_path)
        # print(seconds)
        speech = speech[0:250000]

        preprocessed_speech = self.processor(speech, padding="max_length",  max_length = 250000,return_tensors="pt", sampling_rate = samplerate).input_values
        # label = self.metadata.iloc[idx, 3]
        label = self.metadata.loc[self.metadata.index[idx], "Intelligibility_Label_id"]

        return preprocessed_speech, label


def get_train_test_val_set(args):
    # Load the data
    # comment when working on binary data without augmentation
    train_set = UASpeechDataset(os.path.join(args.data_splits_path, "train_df_binary_aug.csv"), args.data_path, args.SSL_model)
    # uncomment when working on binary data without augmentation
    # train_set = UASpeechDataset(os.path.join(args.data_splits_path, "train_df_binary.csv"), args.data_path, args.SSL_model)
    test_set = UASpeechDataset(os.path.join(args.data_splits_path, "test_df_binary.csv"), args.data_path, args.SSL_model)
    val_set = UASpeechDataset(os.path.join(args.data_splits_path, "val_df_binary.csv"), args.data_path, args.SSL_model)

    train_dataloader = DataLoader(train_set, batch_size = args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size = args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size = args.batch_size, shuffle=True)


    return train_dataloader, test_dataloader, val_dataloader


