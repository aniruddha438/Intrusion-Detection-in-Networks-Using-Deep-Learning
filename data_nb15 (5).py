# import torch
# import torch.utils.data as data
# import numpy as np
# import pandas as pd
# import os
# from sklearn.preprocessing import MinMaxScaler
# import paths_1 as cf
# class Intrusion_Dataset(data.Dataset):
#     def __init__(self, mode='train'):
#         self.mode = mode
        
#         # 🔹 Load CSV file instead of npz
#         csv_path = os.path.join(cf.data_dir, cf.data_filename)   # e.g., "UNSW_NB15.csv"
#         df = pd.read_csv(csv_path)

#         # 🔹 Separate features & labels
#         X = df.iloc[:, :-2].values   # all columns except last
#         y = df.iloc[:, -2].values    # last column is class label

#         # 🔹 Normalize features (important for NN training)
#         scaler = MinMaxScaler()
#         X = scaler.fit_transform(X)

#         # 🔹 Combine features + labels into one array
#         data_all = np.hstack((X, y.reshape(-1,1)))

#         # 🔹 Convert to torch tensor
#         data_all = torch.from_numpy(data_all).float()

#         # 🔹 Separate normal & abnormal
#         normal_data = data_all[data_all[:, -1] == 0]
#         abnormal_data = data_all[data_all[:, -1] == 1]

#         # 🔹 Train/test split
#         train_normal_mark = int(normal_data.shape[0] * cf.train_ratio)
#         train_abnormal_mark = int(abnormal_data.shape[0] * cf.train_ratio)

#         train_normal_data = normal_data[:train_normal_mark, :]
#         train_abnormal_data = abnormal_data[:train_abnormal_mark, :]
#         self.train_data = np.concatenate((train_normal_data, train_abnormal_data), axis=0)
#         np.random.shuffle(self.train_data)

#         test_normal_data = normal_data[train_normal_mark:, :]
#         test_abnormal_data = abnormal_data[train_abnormal_mark:, :]
#         self.test_data = np.concatenate((test_normal_data, test_abnormal_data), axis=0)
#         np.random.shuffle(self.test_data)

#     def __len__(self):
#         if self.mode == 'train':
#             return self.train_data.shape[0]
#         else:
#             return self.test_data.shape[0]
        
#     def __getitem__(self, index):
#         if self.mode == 'train':
#             x = self.train_data[index, :-1]
#             y = self.train_data[index, -1]
#         else:
#             x = self.test_data[index, :-1]
#             y = self.test_data[index, -1]
#         return x, y
    
#     def set_mode(self, mode):
#         self.mode = mode


import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import paths_1 as cf
def prepare_nb15_balanced(csv_path,
                          label_col="Label",
                          attack_col="Attack",
                          n_train_normal=50_000,
                          total_subset=200_000,
                          per_attack=2000,      # target per attack class in test
                          n_test_normals=20_000,# how many benign to include in test
                          random_state=42):
    rng = np.random.default_rng(random_state)

    # 1. load
    df = pd.read_csv(csv_path)

    # 2. drop object-type cols
    drop_cols = df.select_dtypes(include=['object']).columns.tolist()
    if attack_col in df.columns:
        drop_cols.append(attack_col)  # keep aside only for balancing
    df_features = df.drop(columns=drop_cols, errors="ignore")

    # 3. separate
    y = df[label_col].values.astype(int)
    X = df_features.drop(columns=[label_col]).values.astype(np.float64)
    attack_types = df[attack_col].astype(str).values

    # 4. split normal vs abnormal
    normal_mask = (y == 0)
    abnormal_mask = (y != 0)

    X_normal, y_normal = X[normal_mask], y[normal_mask]
    X_abnormal, y_abnormal = X[abnormal_mask], y[abnormal_mask]
    
    attack_types_abnormal = attack_types[abnormal_mask]

    print(f"Normal samples: {len(X_normal)}, Attack samples: {len(X_abnormal)}")

    # 5. training normals
    n_train_normal = min(n_train_normal, len(X_normal))
    train_idx = rng.choice(len(X_normal), size=n_train_normal, replace=False)
    X_train, y_train = X_normal[train_idx], y_normal[train_idx]

    # 6. balanced attack sampling for test
    X_test_list, y_test_list = [], []
    for attack in np.unique(attack_types_abnormal):
        mask = (attack_types_abnormal == attack)
        X_cls, y_cls = X_abnormal[mask], y_abnormal[mask]

        n_take = min(per_attack, len(X_cls))
        idx = rng.choice(len(X_cls), size=n_take, replace=False)
        X_test_list.append(X_cls[idx])
        y_test_list.append(y_cls[idx])

        print(f"Attack {attack:<15} → picked {n_take} / {len(X_cls)}")

    # 7. add some normals to test
    remain_idx = np.setdiff1d(np.arange(len(X_normal)), train_idx)
    n_test_normals = min(n_test_normals, len(remain_idx))
    test_norm_idx = rng.choice(remain_idx, size=n_test_normals, replace=False)
    X_test_list.append(X_normal[test_norm_idx])
    y_test_list.append(y_normal[test_norm_idx])

    # 8. combine and shuffle
    X_test = np.vstack(X_test_list)
    y_test = np.hstack(y_test_list)
    perm = rng.permutation(len(X_test))
    X_test, y_test = X_test[perm], y_test[perm]

    # 9. clean inf/nan
    def clean(arr):
        arr = np.array(arr, dtype=np.float64)
        mask = ~np.isfinite(arr)
        if mask.any():
            arr[mask] = np.nan
            col_mean = np.nanmean(arr, axis=0)
            col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
            inds = np.where(np.isnan(arr))
            arr[inds] = np.take(col_mean, inds[1])
        return arr

    X_train = clean(X_train)
    X_test = clean(X_test)

    # 10. scale
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Final shapes -> Train:", X_train.shape, " Test:", X_test.shape)
    return X_train, y_train, X_test, y_test
# class Intrusion_Dataset(data.Dataset):
#     def __init__(self, mode='train'):
#         self.mode = mode
        
#         # 🔹 Load CSV file
#         csv_path = os.path.join(cf.data_dir, cf.data_filename)
#         df = pd.read_csv(csv_path)

#         # 🔹 Remove object type columns (e.g., ipv4)
#         object_cols = df.select_dtypes(include=['object']).columns
#         df = df.drop(columns=object_cols)
#         # 🔹 Handle infinite or extremely large values
        


#         # 🔹 Separate features & labels
#         X = df.iloc[:, :-2].values   # all columns except last
#         y = df.iloc[:, -2].values    # assume last column is class label
#         X = np.where(np.isfinite(X), X, np.nan)
#         col_mean = np.nanmean(X, axis=0)
#         inds = np.where(np.isnan(X))
#         X[inds] = np.take(col_mean, inds[1])

#         # 🔹 Separate normal and abnormal samples
#         normal_mask = y == 0   # 0 = normal
#         abnormal_mask = y == 1 # 1 = abnormal

#         X_normal, y_normal = X[normal_mask], y[normal_mask]
#         X_abnormal, y_abnormal = X[abnormal_mask], y[abnormal_mask]

#         # 🔹 Take 50,000 normal samples for training (or all if less)
#         n_train_normal = min(50000, X_normal.shape[0])
#         subset_idx = np.random.choice(X_normal.shape[0], n_train_normal, replace=False)
#         X_train = X_normal[subset_idx]
#         y_train = y_normal[subset_idx]

#         # 🔹 Test set: remaining normal + all abnormal
#         remaining_idx = np.setdiff1d(np.arange(X_normal.shape[0]), subset_idx)
#         X_test_normal = X_normal[remaining_idx]
#         y_test_normal = y_normal[remaining_idx]

#         X_test = np.vstack((X_test_normal, X_abnormal))
#         y_test = np.hstack((y_test_normal, y_abnormal))

#         # 🔹 Shuffle test set
#         test_indices = np.arange(X_test.shape[0])
#         np.random.shuffle(test_indices)
#         X_test = X_test[test_indices]
#         y_test = y_test[test_indices]

#         # 🔹 Scale features: fit scaler only on training normal data
#         scaler = MinMaxScaler()
#         X_train = scaler.fit_transform(X_train)
#         X_test = scaler.transform(X_test)

#         # 🔹 Convert to torch tensors
#         self.train_data = torch.from_numpy(np.hstack((X_train, y_train.reshape(-1,1)))).float()
#         self.test_data = torch.from_numpy(np.hstack((X_test, y_test.reshape(-1,1)))).float()

#     def __len__(self):
#         if self.mode == 'train':
#             return self.train_data.shape[0]
#         else:
#             return self.test_data.shape[0]
        
#     def __getitem__(self, index):
#         if self.mode == 'train':
#             x = self.train_data[index, :-1]
#             y = self.train_data[index, -1]
#         else:
#             x = self.test_data[index, :-1]
#             y = self.test_data[index, -1]
#         return x, y
    
#     def set_mode(self, mode):
#         self.mode = mode
class Intrusion_Dataset(data.Dataset):
    def __init__(self, mode='train'):
        self.mode = mode
        csv_path = os.path.join(cf.data_dir, cf.data_filename)

        # Call the helper
        X_train, y_train, X_test, y_test = prepare_nb15_balanced(csv_path)

        # Scale with training fit
        # scaler = MinMaxScaler()
        # X_train = scaler.fit_transform(X_train)
        # X_test = scaler.transform(X_test)

        self.train_data = torch.from_numpy(
            np.hstack((X_train, y_train.reshape(-1, 1)))
        ).float()
        self.test_data = torch.from_numpy(
            np.hstack((X_test, y_test.reshape(-1, 1)))
        ).float()

    def __len__(self):
        return self.train_data.shape[0] if self.mode == 'train' else self.test_data.shape[0]

    def __getitem__(self, index):
        data = self.train_data if self.mode == 'train' else self.test_data
        x, y = data[index, :-1], data[index, -1]
        return x, y

    def set_mode(self, mode):
        self.mode = mode
