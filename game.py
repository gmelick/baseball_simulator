import pandas as pd
import numpy as np
import random
import itertools
import joblib
from datetime import datetime
from multiprocessing import Pool, shared_memory
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.layers import Input, Conv1D, BatchNormalization, ReLU, GlobalAveragePooling1D, Dense
from keras.models import Model
from keras.metrics import PrecisionAtRecall

_global_data = {}
sit_index_file = 'situation_index.pkl'
sim_config_file = 'sim_cfg.pkl'
people_link = 'https://statsapi.mlb.com/api/v1/people'

def numpy_conv1d(x, kernel, bias):
    """x: (seq_len, in_ch), kernel: (k_size, in_ch, out_ch)"""
    seq_len, in_ch = x.shape
    k_size, _, out_ch = kernel.shape
    pad = k_size // 2  # 'same' padding
    x_padded = np.pad(x, ((pad, pad), (0, 0)))
    out = np.zeros((seq_len, out_ch), dtype=np.float32)
    for i in range(seq_len):
        out[i] = x_padded[i:i + k_size].reshape(-1) @ kernel.reshape(-1, out_ch) + bias
    return out

def numpy_batch_norm(x, gamma, beta, mean, var, eps=1e-3):
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

def numpy_relu(x):
    return np.maximum(0, x)

def numpy_gap(x):
    """Global Average Pooling over sequence dimension."""
    return x.mean(axis=0)

def numpy_dense(x, kernel, bias):
    return x @ kernel + bias

def numpy_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def numpy_forward(x, weights):
    """
    x: raw input array of shape (12,) — will be reshaped to (12, 1)
    weights: dict from extract_model_weights()
    """
    x = x.reshape(12, 1).astype(np.float32)
    if 'conv1d' not in weights:
        start_index = int(list(weights.keys())[0][list(weights.keys())[0].rfind('_')+1:])
    else:
        start_index = 0

    # Three Conv -> BN -> ReLU blocks
    for i in range(1, 4):
        if 'conv1d' in weights:
            conv_key = f"conv1d" if i == 1 else f"conv1d_{i-1}"  # Keras default naming
        else:
            conv_key = f'conv1d_{i+start_index-1}'
        if 'batch_normalization' in weights:
            bn_key = f"batch_normalization" if i == 1 else f"batch_normalization_{i-1}"
        else:
            bn_key = f'batch_normalization_{i+start_index-1}'
        x = numpy_conv1d(x, weights[conv_key]["kernel"], weights[conv_key]["bias"])
        x = numpy_batch_norm(x, **weights[bn_key])
        x = numpy_relu(x)

    x = numpy_gap(x)                                              # (64,)
    if 'dense' in weights:
        x = numpy_dense(x, weights["dense"]["kernel"], weights["dense"]["bias"])  # (1,)
    else:
        x = numpy_dense(x, weights[f"dense_{int(start_index / 3)}"]["kernel"], weights[f"dense_{int(start_index / 3)}"]["bias"])
    return float(numpy_sigmoid(x))


# ------------------ Shared memory helpers ------------------
def dataframe_to_shared(df: pd.DataFrame):
    # Convert DataFrame to structured array (contiguous)
    arr = np.empty(len(df), dtype=[(col, df[col].dtype) for col in df.columns])
    for col in df.columns:
        arr[col] = df[col].to_numpy(copy=False)

    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    shm_arr[:] = arr
    return (arr.shape, arr.dtype, shm.name), shm

def extract_model_weights(model):
    weights = {}
    for layer in model.layers:
        if not layer.get_weights():
            continue
        w = layer.get_weights()
        name = layer.name
        if isinstance(layer, Conv1D):
            weights[name] = {"kernel": w[0], "bias": w[1]}
        elif isinstance(layer, BatchNormalization):
            # gamma, beta, moving_mean, moving_variance
            weights[name] = {"gamma": w[0], "beta": w[1], "mean": w[2], "var": w[3]}
        elif isinstance(layer, Dense):
            weights[name] = {"kernel": w[0], "bias": w[1]}
    return weights

def weights_to_shared(model):
    weights_dict = {f"w{i}": w for i, w in enumerate(model.get_weights())}
    shm_dict = {}
    meta_dict = {}
    for name, arr in weights_dict.items():
        shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        shm_arr[:] = arr
        shm_dict[name] = shm
        meta_dict[name] = (arr.shape, arr.dtype, shm.name)
    return meta_dict, shm_dict

def shared_to_arrays(meta):
    shape, dtype, shm_name = meta
    shm = shared_memory.SharedMemory(name=shm_name)
    shm_arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return {col: shm_arr[col] for col in shm_arr.dtype.names}, shm

def load_weights_from_shared_memory(meta_dict):
    weights = []
    for i in range(len(meta_dict)):
        key = f"w{i}"
        shape, dtype, shm_name = meta_dict[key]
        shm = shared_memory.SharedMemory(name=shm_name)
        arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        weights.append(arr.copy())  # copy to avoid accidental write-through
    return weights

# ------------------ Situation Index Helpers ------------------
def build_situation_index(arrays):
    strikes = arrays['strikes']
    outs = arrays['outs']
    balls = arrays['balls']
    on1 = arrays['on_1b_na'].astype(np.int8)
    on2 = arrays['on_2b_na'].astype(np.int8)
    on3 = arrays['on_3b_na'].astype(np.int8)

    # Encode into a single integer code per row
    codes = (strikes * (3*4*2*2*2) +
             outs * (4*2*2*2) +
             balls * (2*2*2) +
             on1 * (2*2) +
             on2 * 2 +
             on3)

    # Sort once, then group indices
    order = np.argsort(codes, kind="mergesort")  # stable sort
    codes_sorted = codes[order]

    situation_index = {}
    start = 0
    for code, group in itertools.groupby(codes_sorted):
        end = start + sum(1 for _ in group)
        idx = order[start:end]
        situation_index[code] = idx
        start = end

    return situation_index

def decode_situation_code(code):
    on3 = code % 2
    code //= 2
    on2 = code % 2
    code //= 2
    on1 = code % 2
    code //= 2
    balls = code % 4
    code //= 4
    outs = code % 3
    code //= 3
    strikes = code
    return strikes, outs, balls, bool(on1), bool(on2), bool(on3)

# ------------------ Example model builder ------------------
def build_model():
    input_layer = Input((12, 1))

    conv1 = Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)

    conv2 = Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = ReLU()(conv2)

    conv3 = Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = ReLU()(conv3)

    gap = GlobalAveragePooling1D()(conv3)

    output_layer = Dense(1, activation='sigmoid')(gap)

    model = Model(inputs=input_layer, outputs=output_layer)

    model.compile(
        optimizer="adam",
        loss='binary_crossentropy',
        metrics=[PrecisionAtRecall(.5)],
    )
    return model

# ------------------ Worker setup ------------------
def _init_worker(plays_meta):
    global _global_data
    arrays, shared_mems = shared_to_arrays(plays_meta)
    _global_data["plays"] = arrays
    _global_data["_shared_mems"] = shared_mems

    # Store config and models once per worker
    _global_data.update(joblib.load(sim_config_file))
    _global_data["situation_index"] = joblib.load(sit_index_file)

# ------------------ Simulation entry ------------------
def run_simulation(_):
    global _global_data
    game = Game(
        game_date=_global_data["game_date"],
        home_team=_global_data["home_team"],
        away_team=_global_data["away_team"],
        game_pk=_global_data["game_pk"],
        away_pitcher=_global_data["away_pitcher"],
        home_pitcher=_global_data["home_pitcher"],
        away_lineup=_global_data["away_lineup"],
        home_lineup=_global_data["home_lineup"],
        away_bullpen=_global_data["away_bullpen"],
        home_bullpen=_global_data["home_bullpen"],
        away_bullpen_appearances=_global_data["away_bullpen_apps"],
        home_bullpen_appearances=_global_data["home_bullpen_apps"],
        pitcher_hand_map=_global_data["pitcher_hand_map"]
    )
    return game.sim_game(_global_data["plays"], _global_data["situation_index"], _global_data["starter_model_weights"], _global_data["reliever_model_weights"])

# ------------------ Game class ------------------
class Game:
    def __init__(self, game_date, home_team, away_team, game_pk, away_pitcher, home_pitcher, away_lineup, home_lineup,
                 away_bullpen, home_bullpen, away_bullpen_appearances, home_bullpen_appearances, pitcher_hand_map):
        self.game_date = game_date
        self.game_pk = game_pk
        self.home_team, self.away_team = home_team, away_team
        self.starting_pitchers = [home_pitcher, away_pitcher]
        self.home_pitcher, self.away_pitcher = home_pitcher, away_pitcher
        self.home_lineup, self.away_lineup = home_lineup, away_lineup
        self.home_bullpen, self.away_bullpen = home_bullpen, away_bullpen
        self.pitcher_hand_map = pitcher_hand_map
        self.original_bullpen_apps = [home_bullpen_appearances, away_bullpen_appearances]
        self.home_bullpen_appearances, self.away_bullpen_appearances = home_bullpen_appearances, away_bullpen_appearances
        self.home_bullpen_apps_np = home_bullpen_appearances[['inning','Difference','game_pk','pitcher']].to_records(index=False)
        self.away_bullpen_apps_np = away_bullpen_appearances[['inning','Difference','game_pk','pitcher']].to_records(index=False)
        self.linescore = [0] * 20
        self.linescore_index = 0
        self.home_lineup_index, self.away_lineup_index = 0, 0
        self.inning = 1
        self.is_top_of_inning = True
        self.balls, self.strikes, self.outs = 0, 0, 0
        self.baserunners = [None, None, None]
        self.home_score, self.away_score = 0, 0
        self.pitcher = self.home_pitcher
        self.batter = self.away_lineup[0]
        self.pitcher_hand = self.pitcher_hand_map[self.pitcher]
        all_pitchers = [self.home_pitcher, self.away_pitcher] + self.home_bullpen + self.away_bullpen
        all_batters = self.home_lineup + self.away_lineup
        self.pitcher_index = {pid: i for i, pid in enumerate(all_pitchers)}
        self.batter_index = {bid: i for i, bid in enumerate(all_batters)}
        self.pitcher_stats = np.zeros((len(self.pitcher_index), 11), dtype=np.int32)
        self.batter_stats = np.zeros((len(self.batter_index), 11), dtype=np.int32)
        self.home_pitcher_list, self.away_pitcher_list = [self.home_pitcher], [self.away_pitcher]
        self._hook_buffer = np.zeros(12, dtype=np.float32)
        self.plays = None
        self.situation_index = None
        self.starter_model_weights = None
        self.reliever_model_weights = None

    def simulate(self, n_sims=100, processes=12, chunksize=5, hook_model_starters=None, hook_model_relievers=None):
        plays = pd.read_feather(
            f'{self.game_date.year}\\{self.game_date.strftime("%Y_%m_%d")}\\{self.game_date.strftime("%Y_%m_%d")}_{self.game_pk}.feather'
        )

        starter_model_weights = extract_model_weights(hook_model_starters)
        reliever_model_weights = extract_model_weights(hook_model_relievers)

        sim_cfg = dict(
            game_date=self.game_date,
            game_pk=self.game_pk,
            home_team=self.home_team,
            away_team=self.away_team,
            home_pitcher=self.home_pitcher,
            away_pitcher=self.away_pitcher,
            home_lineup=self.home_lineup,
            away_lineup=self.away_lineup,
            home_bullpen=self.home_bullpen,
            away_bullpen=self.away_bullpen,
            home_bullpen_apps=self.home_bullpen_appearances,
            away_bullpen_apps=self.away_bullpen_appearances,
            pitcher_hand_map=self.pitcher_hand_map,
            starter_model_weights=starter_model_weights,
            reliever_model_weights=reliever_model_weights
        )
        joblib.dump(sim_cfg, sim_config_file)

        situation_index = build_situation_index({
            'strikes': plays['strikes'].to_numpy(),
            'balls': plays['balls'].to_numpy(),
            'outs': plays['outs'].to_numpy(),
            'on_1b_na': plays['on_1b_na'].to_numpy().astype(np.int8),
            'on_2b_na': plays['on_2b_na'].to_numpy().astype(np.int8),
            'on_3b_na': plays['on_3b_na'].to_numpy().astype(np.int8),
        })
        joblib.dump(situation_index, sit_index_file)

        plays.drop(columns=['strikes', 'balls', 'outs', 'on_1b_na', 'on_2b_na', 'on_3b_na'], inplace=True)
        plays_meta, shm_block = dataframe_to_shared(plays)

        print(f'Started at {datetime.now()}')
        with Pool(processes=processes, initializer=_init_worker, initargs=(plays_meta,)) as pool:
            results = pool.map(run_simulation, range(n_sims), chunksize=chunksize)
        print(f'Finished at {datetime.now()}')

        shm_block.close()
        shm_block.unlink()

        output_path = f'{self.game_date.year}\\{self.game_date.strftime("%Y_%m_%d")}\\{self.game_date.strftime("%Y_%m_%d")}_{self.game_pk}.csv'
        with open(output_path, "w+") as file:
            file.write('T1,B1,T2,B2,T3,B3,T4,B4,T5,B5,T6,B6,T7,B7,T8,B8,T9,B9,T10+,B10+,Home Score,Away Score')
            for i in range(12):
                file.write(
                    f',Home Pitcher {i + 1},HP_{i + 1}_BF,HP_{i + 1}_Outs,HP_{i + 1}_H,HP_{i + 1}_R,HP_{i + 1}_K,HP_{i + 1}_BB,'
                    f'HP_{i + 1}_HBP,HP_{i + 1}_HR,HP_{i + 1}_Pitches,HP_{i + 1}_Strikes,HP_{i + 1}_ER')
            for i in range(12):
                file.write(
                    f',Away Pitcher {i + 1},AP_{i + 1}_BF,AP_{i + 1}_Outs,AP_{i + 1}_H,AP_{i + 1}_R,AP_{i + 1}_K,AP_{i + 1}_BB,'
                    f'AP_{i + 1}_HBP,AP_{i + 1}_HR,AP_{i + 1}_Pitches,AP_{i + 1}_Strikes,AP_{i + 1}_ER')
            for i in range(9):
                file.write(
                    f',Home Batter {i + 1},HB_{i + 1}_PA,HB_{i + 1}_AB,HB_{i + 1}_H,HB_{i + 1}_2B,HB_{i + 1}_3B,HB_{i + 1}_HR,'
                    f'HB_{i + 1}_K,HB_{i + 1}_BB,HB_{i + 1}_HBP,HB_{i + 1}_R,HB_{i + 1}_RBI')
            for i in range(9):
                file.write(
                    f',Away Batter {i + 1},AB_{i + 1}_PA,AB_{i + 1}_AB,AB_{i + 1}_H,AB_{i + 1}_2B,AB_{i + 1}_3B,AB_{i + 1}_HR,'
                    f'AB_{i + 1}_K,AB_{i + 1}_BB,AB_{i + 1}_HBP,AB_{i + 1}_R,AB_{i + 1}_RBI')
            file.write('\n')
            for r in results:
                file.write(r + '\n')

    def sim_game(self, plays, situation_index, starter_model_weights, reliever_model_weights):
        self.home_bullpen_appearances, self.away_bullpen_appearances = self.original_bullpen_apps[0], self.original_bullpen_apps[1]
        self.linescore = [0] * 20
        self.linescore_index = 0
        self.home_lineup_index, self.away_lineup_index = 0, 0
        self.inning = 1
        self.is_top_of_inning = True
        self.baserunners = [None, None, None]
        self.home_score, self.away_score = 0, 0
        self.home_pitcher, self.away_pitcher = self.starting_pitchers[0], self.starting_pitchers[1]
        self.pitcher = self.home_pitcher
        self.batter = self.away_lineup[0]
        self.pitcher_hand = self.pitcher_hand_map[self.pitcher]
        # Pitcher Stats: 0=BF, 1=Outs, 2=Hits, 3=Runs, 4=K, 5=BB, 6=HBP, 7=HR, 8=Pitches, 9=Strikes, 10=ER
        self.pitcher_stats = np.zeros((len(self.pitcher_index), 11), dtype=np.int32)
        # Batter Stats: 0=PA, 1=AB, 2=Hits, 3=2B, 4=3B, 5=HR, 6=K, 7=BB, 8=HBP, 9=Runs, 10=RBI
        self.batter_stats = np.zeros((len(self.batter_index), 11), dtype=np.int32)
        self.home_pitcher_list, self.away_pitcher_list = [self.home_pitcher], [self.away_pitcher]
        self.starter_model_weights = starter_model_weights
        self.reliever_model_weights = reliever_model_weights
        self.plays = plays
        self.situation_index = situation_index

        while self.inning <= 9 or self.home_score == self.away_score:
            self.simulate_half_inning()
            if self.inning >= 9 and self.home_score > self.away_score:
                break
            self.simulate_half_inning()
            self.inning += 1

        out = [','.join(map(str, self.linescore)), str(self.home_score), str(self.away_score)]

        for pitcher_list in [self.home_pitcher_list, self.away_pitcher_list]:
            count = 0
            for pid in pitcher_list:
                idx = self.pitcher_index[pid]
                stats = ','.join(map(str, self.pitcher_stats[idx]))
                out.append(f"{pid},{stats}")
                count += 1
            while count < 12:
                out.append(',' * 11)
                count += 1

        for lineup in [self.home_lineup, self.away_lineup]:
            for bid in lineup:
                idx = self.batter_index[bid]
                stats = ','.join(map(str, self.batter_stats[idx]))
                out.append(f"{bid},{stats}")

        return ','.join(out)

    def simulate_half_inning(self):
        while self.outs < 3:
            self.simulate_at_bat()
            if self.inning >= 9 and not self.is_top_of_inning and self.home_score > self.away_score:
                break
        if self.is_top_of_inning:
            self.pitcher = self.away_pitcher
            self.pitcher_hand = self.pitcher_hand_map[self.pitcher]
            self.batter = self.home_lineup[self.home_lineup_index]
        else:
            self.pitcher = self.home_pitcher
            self.pitcher_hand = self.pitcher_hand_map[self.pitcher]
            self.batter = self.away_lineup[self.away_lineup_index]
        self.outs = 0
        self.baserunners = [None, None, None]
        self.is_top_of_inning = not self.is_top_of_inning
        self.linescore_index += 1

    def simulate_at_bat(self):
        self.strikes, self.balls = 0, 0
        self.pitcher_stats[self.pitcher_index[self.pitcher], 0] += 1
        self.batter_stats[self.batter_index[self.batter], 0] += 1
        while self.strikes < 3 and self.balls < 4:
            pitch_res = self.simulate_pitch()
            if pitch_res > 0:
                break
        self.next_batter()
        self.check_pitcher_hook()
        self.strikes = 0
        self.balls = 0

    def simulate_pitch(self):
        code = (self.strikes * (3 * 4 * 2 * 2 * 2) +
                self.outs * (4 * 2 * 2 * 2) +
                self.balls * (2 * 2 * 2) +
                int(self.baserunners[0] is None) * (2 * 2) +
                int(self.baserunners[1] is None) * 2 +
                int(self.baserunners[2] is None))
        candidates = self.situation_index[code]
        similarities = self.plays[f'P_{self.pitcher}'][candidates] * self.plays[f'B_{self.pitcher_hand}_{self.batter}'][candidates]
        mask = similarities >= similarities.mean()
        weights = similarities[mask] / similarities[mask].sum()
        chosen_idx = np.random.choice(candidates[mask], p=weights)

        # Update stats via NumPy arrays
        self.pitcher_stats[self.pitcher_index[self.pitcher], 8] += 1
        if self.plays['Strike'][chosen_idx]:
            self.pitcher_stats[self.pitcher_index[self.pitcher], 9] += 1
            if not self.plays['Foul'][chosen_idx] or self.strikes < 2:
                self.strikes += 1
            if self.strikes == 3:
                self.pitcher_stats[self.pitcher_index[self.pitcher], 4] += 1
                self.batter_stats[self.batter_index[self.batter], 1] += 1
                self.batter_stats[self.batter_index[self.batter], 6] += 1
                return_val = 1
            else:
                return_val = 0
        elif self.plays['Ball'][chosen_idx]:
            self.balls += 1
            if self.balls == 4:
                self.pitcher_stats[self.pitcher_index[self.pitcher], 5] += 1
                self.batter_stats[self.batter_index[self.batter], 7] += 1
                return_val = 1
            else:
                return_val = 0
        else:
            if self.plays['Hit'][chosen_idx]:
                self.pitcher_stats[self.pitcher_index[self.pitcher], 2] += 1
                self.batter_stats[self.batter_index[self.batter], 2] += 1
            if self.plays['HBP'][chosen_idx]:
                self.pitcher_stats[self.pitcher_index[self.pitcher], 6] += 1
                self.batter_stats[self.batter_index[self.batter], 8] += 1
            else:
                self.batter_stats[self.batter_index[self.batter], 1] += 1
            if self.plays['Home_Run'][chosen_idx]:
                self.pitcher_stats[self.pitcher_index[self.pitcher], 7] += 1
                self.batter_stats[self.batter_index[self.batter], 5] += 1
                self.batter_stats[self.batter_index[self.batter], 9] += 1
            if self.plays['Triple'][chosen_idx]:
                self.batter_stats[self.batter_index[self.batter], 4] += 1
            if self.plays['Double'][chosen_idx]:
                self.batter_stats[self.batter_index[self.batter], 3] += 1
            return_val = 1

        self.outs += self.plays['outs_on_pitch'][chosen_idx]
        runs = self.plays['runs_on_pitch'][chosen_idx]
        if runs > 0:
            self.score(runs)
        self.pitcher_stats[self.pitcher_index[self.pitcher], 1] += self.plays['outs_on_pitch'][chosen_idx]
        self.pitcher_stats[self.pitcher_index[self.pitcher], 3] += runs
        self.pitcher_stats[self.pitcher_index[self.pitcher], 10] += self.plays['earned_runs_on_pitch'][chosen_idx]
        self.batter_stats[self.batter_index[self.batter], 10] += self.plays['rbis_on_pitch'][chosen_idx]

        if self.plays['1b_runner_score'][chosen_idx]:
            self.batter_stats[self.batter_index[self.baserunners[0]], 9] += 1
        if self.plays['2b_runner_score'][chosen_idx]:
            self.batter_stats[self.batter_index[self.baserunners[1]], 9] += 1
        if self.plays['3b_runner_score'][chosen_idx]:
            self.batter_stats[self.batter_index[self.baserunners[2]], 9] += 1

        self.update_baserunners(self.plays, chosen_idx)

        if self.outs >= 3:
            return 1
        return return_val

    def score(self, num_runs):
        if self.is_top_of_inning:
            self.away_score += num_runs
        else:
            self.home_score += num_runs
        if self.inning >= 10:
            if self.is_top_of_inning:
                self.linescore[-2] += num_runs
            else:
                self.linescore[-1] += num_runs
        else:
            self.linescore[self.linescore_index] += num_runs

    def next_batter(self):
        if self.is_top_of_inning:
            self.away_lineup_index = (self.away_lineup_index + 1) % 9
            self.batter = self.away_lineup[self.away_lineup_index]
        else:
            self.home_lineup_index = (self.home_lineup_index + 1) % 9
            self.batter = self.home_lineup[self.home_lineup_index]

    def check_pitcher_hook(self):
        stats = self.pitcher_stats[self.pitcher_index[self.pitcher]]
        self._hook_buffer[:11] = stats
        self._hook_buffer[11] = 1 if self.outs == 3 else 0
        if stats[0] >= 3 or self.outs == 3:
            weights = self.starter_model_weights if self.pitcher in self.starting_pitchers else self.reliever_model_weights
            prob = numpy_forward(self._hook_buffer[np.newaxis, :, np.newaxis], weights)
            if prob >= np.random.random():
                self.substitute_pitcher()

    def substitute_pitcher(self):
        inning = min(max(self.inning, 5), 9)
        diff = min(abs(self.home_score - self.away_score), 5)
        pitcher_list = self.home_pitcher_list if self.is_top_of_inning else self.away_pitcher_list
        bullpen_apps_np = self.home_bullpen_apps_np if self.is_top_of_inning else self.away_bullpen_apps_np
        if len(pitcher_list) >= 12 or bullpen_apps_np.size == 0:
            return
        mask = ((bullpen_apps_np['inning'] == inning) & (bullpen_apps_np['Difference'] == diff))
        pool = bullpen_apps_np[mask]
        if pool.size == 0:
            pool = bullpen_apps_np
        weights = np.cumsum(pool['game_pk'])
        pick = random.randint(0, int(weights[-1]) - 1)
        new_pitcher = pool['pitcher'][np.searchsorted(weights, pick)]

        self.pitcher = new_pitcher
        self.pitcher_hand = self.pitcher_hand_map[new_pitcher]
        if self.is_top_of_inning:
            self.home_pitcher = new_pitcher
            self.home_pitcher_list.append(new_pitcher)
            self.home_bullpen_apps_np = self.home_bullpen_apps_np[self.home_bullpen_apps_np['pitcher'] != new_pitcher]
        else:
            self.away_pitcher = new_pitcher
            self.away_pitcher_list.append(new_pitcher)
            self.away_bullpen_apps_np = self.away_bullpen_apps_np[self.away_bullpen_apps_np['pitcher'] != new_pitcher]

    def update_baserunners(self, plays, idx):
        post1, post2, post3 = plays['post_on_1b'][idx], plays['post_on_2b'][idx], plays['post_on_3b'][idx]
        on1, on2, on3 = plays['on_1b'][idx], plays['on_2b'][idx], plays['on_3b'][idx]
        batter = plays['batter'][idx]

        if post3 == -1:
            self.baserunners[2] = None
        elif post3 == on2:
            self.baserunners[2] = self.baserunners[1]
        elif post3 == on1:
            self.baserunners[2] = self.baserunners[0]
        elif post3 == batter:
            self.baserunners[2] = self.batter

        if post2 == -1:
            self.baserunners[1] = None
        elif post2 == on1:
            self.baserunners[1] = self.baserunners[0]
        elif post2 == batter:
            self.baserunners[1] = self.batter

        if post1 == -1:
            self.baserunners[0] = None
        elif post1 == batter:
            self.baserunners[0] = self.batter
