import os
import torch

import pandas as pd
import numpy as np
from datetime import datetime

from torch.utils.data import DataLoader
from srcs.utils import Datasets

class dataLoader:
    def __init__(self, cfg):
        self._init_cfg(cfg)

        print(f"[INFO] : Feature mode is {self.feature_mode}")
        if (f'{self.feature_mode}_train_data.npz' not in os.listdir(r"./Datas")) or (f'{self.feature_mode}_test_data.npz' not in os.listdir(r"./Datas")):
            print(f"[INFO] : We don't have feature files. Start Processing...")
            self._init_processing()
        else:
            train_dat_npz                       = np.load(f'./Datas/{self.feature_mode}_train_data.npz', allow_pickle=True)
            test_dat_npz                        = np.load(f'./Datas/{self.feature_mode}_test_data.npz' , allow_pickle=True)

            self.train_x_npy, self.train_y_npy  = train_dat_npz["x"], train_dat_npz["y"]
            self.test_x_npy, self.test_y_npy    = test_dat_npz["x"], test_dat_npz["y"]

            train_dat_npz.close()
            test_dat_npz.close()

    def _init_cfg(self, cfg):
        
        ##### (1) 필요한 파라미터 정의 #####
        self.scaling           = cfg.dataset.scaling                                                              # Data Scaling Flag 
        if self.scaling:
            self.feature_mode   = f'scaled_{cfg.dataset.feature_mode}'
        else:
            self.feature_mode       = cfg.dataset.feature_mode
        self.predict_type       = cfg.dataset.predict_type                                                        # "Classification", "Regression"
        self.label_mode         = cfg.dataset.label_mode                                                          # "optim_u", "optim_b" 

        self.time_interval      = int(cfg.dataset.time_interval)

        self.trn_batch_size     = cfg.model.batch_size
        self.val_batch_size     = cfg.dataset.val_batch_size

        self.srt_mrk_t          = int(cfg.dataset.market_start_time)                                              # Market Open Time  : 900
        self.end_mrk_t          = int(cfg.dataset.market_end_time)                                                # Market Close Time : 1530
        self.trn_split_point    = datetime.strptime(cfg.dataset.train_split_point, "%m%d%y").strftime("%Y-%m-%d") # "2016-07-29"
        self.test_st_dt         = datetime.strptime(cfg.dataset.test_start_date, "%m%d%y").strftime("%Y-%m-%d")   # "2021-01-04"

        ##### (2) Feature/Target에 따라 필요한 데이터 로드 #####      
        self.fx_close_dat       = pd.read_csv(os.path.join(cfg.dataset.fx_data_dir, "usdkrw_futures_12yr_reshape.csv"), index_col=0).fillna(method="ffill")
        self.target_dat         = pd.read_csv(os.path.join(cfg.dataset.fx_data_dir, f"{self.label_mode}_ver1.csv"), index_col=0)


    def _init_processing(self):

        ##### (1) Train & Test Split #####
        trn_dat_bef        = self.fx_close_dat.loc[ :1500,                       :  self.trn_split_point].iloc[:,:-1]
        trn_dat_aft        = self.fx_close_dat.loc[ :1530, self.trn_split_point  :       self.test_st_dt]
        test_dat           = self.fx_close_dat.loc[ :1530, self.test_st_dt       :                      ]

        trn_target_bef     = self.target_dat.loc[ :1500,                      : self.trn_split_point].iloc[:,:-1]
        trn_target_aft     = self.target_dat.loc[ :1530, self.trn_split_point :      self.test_st_dt]
        test_target        = self.target_dat.loc[ :1530, self.test_st_dt      :                     ]
    
        ##### (2) Data Reformatting #####
        train_x_1, train_y_1, _        = self._base_arrange_data(input_dat=trn_dat_bef, target_dat=trn_target_bef)
        train_x_2, train_y_2, _        = self._base_arrange_data(input_dat=trn_dat_aft, target_dat=trn_target_aft)
        train_x, train_y               = np.concatenate((train_x_1, train_x_2), axis=0), np.concatenate((train_y_1, train_y_2))
        test_x, test_y, self.index_arr = self._base_arrange_data(input_dat=test_dat   , target_dat=test_target)

        ##### (3-1) Data Sorting : get available_loc #####
        available_train_x_loc               = np.where(np.sum(np.isnan(train_x), axis=1) == 0)[0]
        available_test_x_loc                = np.where(np.sum(np.isnan(test_x), axis=1) == 0)[0]
        available_train_y_loc               = np.where(np.isnan(train_y) == False)[0]
        available_test_y_loc                = np.where(np.isnan(test_y) == False)[0]

        ##### (3-2) Data Sorting : extraction #####
        available_train_loc                 = sorted(list(set(available_train_x_loc) & set(available_train_y_loc)))
        available_test_loc                  = sorted(list(set(available_test_x_loc) & set(available_test_y_loc)))

        self.train_x_npy, self.train_y_npy  = train_x[available_train_loc, :], train_y[available_train_loc]    
        self.test_x_npy , self.test_y_npy   = test_x [available_test_loc , :], test_y [available_test_loc]
        
        ##### (4*) Data Scaling  #####
        if self.scaling:
            train_set                  = self.fx_close_dat.loc[ : ,                       :  self.test_st_dt].iloc[:,:-1]  
            self.max_p, self.min_p     = train_set.max().max(), train_set.min().min()

            self.train_x_npy = (self.train_x_npy - self.min_p) / (self.max_p - self.min_p)
            self.train_y_npy = (self.train_y_npy - self.min_p) / (self.max_p - self.min_p)
            self.test_x_npy  = (self.test_x_npy  - self.min_p) / (self.max_p - self.min_p)
            self.test_y_npy  = (self.test_y_npy  - self.min_p) / (self.max_p - self.min_p) 

        ##### (5) Data 저장 #####
        np.savez_compressed(f"./Datas/{self.feature_mode}_train_data.npz", x = self.train_x_npy, y = self.train_y_npy)
        np.savez_compressed(f"./Datas/{self.feature_mode}_test_data.npz" , x = self.test_x_npy , y = self.test_y_npy)

    def load_data(self):
        trn_val_rat                         = int(self.train_x_npy.shape[0] * 0.8)
        
        if (self.train_x_npy.ndim < 3) or (self.test_x_npy.ndim < 3):
            self.train_x_npy, self.test_x_npy = np.expand_dims(self.train_x_npy, 2), np.expand_dims(self.test_x_npy, 2)

        self.train_x_npy, self.val_x_npy    = self.train_x_npy[:trn_val_rat], self.train_x_npy[trn_val_rat:]
        self.train_y_npy, self.val_y_npy    = self.train_y_npy[:trn_val_rat], self.train_y_npy[trn_val_rat:]

        self.train_x,  self.train_y         = torch.tensor(self.train_x_npy), torch.tensor(self.train_y_npy) 
        self.val_x  ,  self.val_y           = torch.tensor(self.val_x_npy), torch.tensor(self.val_y_npy)  
        self.test_x ,  self.test_y          = torch.tensor(self.test_x_npy), torch.tensor(self.test_y_npy)

        train_dset, val_dset, test_dset     = Datasets(self.train_x, self.train_y), Datasets(self.val_x, self.val_y), Datasets(self.test_x, self.test_y)

        self.train_dataloader               = DataLoader(train_dset, batch_size = self.trn_batch_size)
        self.val_dataloader                 = DataLoader(val_dset  , batch_size = self.val_batch_size)
        self.test_dataloader                = DataLoader(test_dset , batch_size = self.val_batch_size)

        print(f"[INFO] train shape : {self.train_x.shape} \t {self.train_y.shape}")
        print(f"[INFO] val shape   : {self.val_x.shape}   \t {self.val_y.shape}")
        print(f"[INFO] test shape  : {self.test_x.shape}  \t {self.test_y.shape}")


    def _base_arrange_data(self, input_dat, target_dat, indexer=None, m_factor = None):
    
        input_data_npy, index_arr = self.convert_frame_to_numpy(data = input_dat , seq_len = self.time_interval, timestep = 1)
        target_data_npy, _        = self.convert_frame_to_numpy(data = target_dat, seq_len = self.time_interval, timestep = 1)
        target_data_npy           = target_data_npy[:, 0, :]

        input_data    = np.concatenate([input_data_npy[i, :, j] for j in range(input_data_npy.shape[2])
                                    for i in range(input_data_npy.shape[0])])

        input_data    = input_data.reshape(-1, self.time_interval)
        target_data   = target_data_npy.transpose().reshape(-1)

        return input_data, target_data, index_arr
        
    def convert_frame_to_numpy(self, data, seq_len, timestep, use_columns=None, type = None):
        """DataFrame을 3dArray로 변환.

        Parameters
        ----------
        data : DataFrame
            변환하고자 하는 2차원 시계열 DataFrame
        seq_len : int
            하나의 2dArray에 들어갈 데이터 길이
        timestep : int
            2dArray 간의 timestep
        use_columns : None or list
            list : DataFrame의 여러 Column중 변환하고자 하는 컬럼

        Returns
        -------
        3dArray(Value), 2dArray(TimeLocation Array)
            3dArray(Value) : 만일 seq_len = 10, timestep=1 이라면 10일의 2d_array를 1일 간격으로 쌓는다.
            2dArray(TimeLocation Array) : 3dArray의 각 배치의 index위치를 알려줌
        """

        # 오류 검사
        # input data는 DataFrame혹은 Series 이어야 합니다.
        input_data_type = data.__class__
        pandas_available_type = [pd.DataFrame([]).__class__]
        series_available_type = [pd.Series([]).__class__]
        available_type = pandas_available_type + series_available_type

        if input_data_type not in available_type:
            raise AttributeError("지원하지 않는 Input Data형식 입니다. 지원 형식 : DataFrame, Series")

        # set_mode variable setting
        # Input Data 자료형에 따른 mode 설정
        if input_data_type in pandas_available_type:
            set_mode = "DataFrame"
        if input_data_type in series_available_type:
            set_mode = "Series"

        # 오류 검사
        # seq_len은 int type이어야 합니다.
        if not isinstance(seq_len, int):
            raise AttributeError("seq_len 변수는 Int이어야 합니다.")

        # 오류 검사
        # timestep은 int type이어야 합니다.
        if not isinstance(timestep, int):
            raise AttributeError("timestep 변수는 Int이어야 합니다.")

        # 오류 검사
        # use_columns를 설정할 경우 dataframe의 column안에 존재해야 함.
        if (use_columns is not None) & (set_mode == "DataFrame"):
            for usable_columns in use_columns:
                if usable_columns not in data.columns:
                    raise AttributeError("{}의 데이터가 존재하지 않습니다.".format(usable_columns))

            dataframe = data.loc[self.srt_mrk_t : self.end_mrk_t - 1][use_columns]

        else:
            dataframe = data.loc[self.srt_mrk_t : self.end_mrk_t - 1]

        # batch size 결정
        if set_mode == "DataFrame":
            row_num = len(dataframe)
            col_num = len(dataframe.columns)
        elif set_mode == "Series":
            row_num = len(dataframe)
            col_num = 1

        num_of_batch = (row_num - seq_len) / timestep + 1
        num_of_batch = int(num_of_batch) 

        # array 미리 선정
        reformat_value_array = np.zeros((num_of_batch, seq_len, col_num))
        reformat_index_array = np.zeros((num_of_batch, seq_len)).astype(int)
        forward_constant = 0   
        idx_delay        = 0   

        if set_mode == "DataFrame":
            index_size = range(len(dataframe.index))
            indexer = np.array(index_size)
            dataframe = np.array(dataframe, dtype=float)
        elif set_mode == "Series":
            index_size = range(len(dataframe.index))
            indexer = np.array(index_size)
            dataframe = np.array(dataframe, dtype=float).reshape(-1, 1)



        # 3차원 Array로 dataframe 변형
        for batch_num in range(num_of_batch):
            if batch_num == 0:
                reformat_value_array[batch_num, :, :] = dataframe[
                                                        forward_constant : seq_len + idx_delay
                                                        ]
                if set_mode == "DataFrame" or set_mode == "Series":
                    reformat_index_array[batch_num, :] = indexer[
                                                        forward_constant : seq_len + idx_delay
                                                        ]
            else:
                reformat_value_array[batch_num, :, :] = dataframe[
                                                        forward_constant : seq_len + idx_delay + (batch_num * timestep)
                                                        ]
                if set_mode == "DataFrame" or set_mode == "Series":
                    reformat_index_array[batch_num, :] = indexer[
                                                        forward_constant : seq_len + idx_delay + (batch_num * timestep)
                                                        ]

            forward_constant += timestep

        return reformat_value_array, reformat_index_array
