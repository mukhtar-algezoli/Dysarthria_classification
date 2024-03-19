import numpy as np
from __future__ import print_function
import argparse
import joblib
from tqdm import tqdm
import torch
import os 








def main():
    # Training settings
    parser = argparse.ArgumentParser(description='get Hubert kmeans centriods')
    
    parser.add_argument('--aug_data_path', type=str, default="data/raw" ,help='path to clusters file')
    
    parser.add_argument('--output_path', type=str, default="./output" ,help='path to the output dir')

    parser.add_argument('--seed', type=int, default=101, metavar='S',
                        help='random seed (default: 101)')
    
    parser.add_argument('--model_path', type=str, default="./models/",
                        help='path to kmeans model')
    
    parser.add_argument('--severity_label', default='S0', const='S0', nargs='?', choices=['S0', 'S1', 'S2', 'S3'],help='Severity label for the severity. []')
    
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print("loading kmeans model...")

    km_model = joblib.load("/content/hubert_base_ls960_L9_km500.bin")

    print(f"reading file {args.aug_data_path} and getting centriods...")

    with open(args.aug_data_path, "r") as f:
        counter = 0
        for line in tqdm(f):
            temp_arr = []
            clusters = line.split("#")[1:]
            for cluster in clusters:
                temp_arr.append(km_model.cluster_centers_[int(cluster)])
            temp_arr = np.array(temp_arr)
            np.save(f"/content/outputs/{args.severity_label}_{counter}.npy", temp_arr)
            counter += 1
            # print("clusters_printed")


if __name__ == '__main__':
    main()