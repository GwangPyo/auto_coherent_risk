from tqdm import tqdm
import pandas as pd
import os
import numpy as np


def _evaluate(env, model, steps=1000, verbose=True):
    scores = []
    success = []
    iterator = tqdm(range(steps)) if verbose else range(steps)
    for _ in iterator:
        obs = env.reset()
        done = False
        score = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            score += reward
            if done and "is_success" in info.keys():
                success.append(info["is_success"])
        scores.append(score)
    if len(success) > 0:
        result = {"scores": scores, "success": success}
    else:
        result = {"scores": scores}
    return result


def result_to_csv(path, result_dict):
    df = pd.DataFrame.from_dict(result_dict)
    df.to_csv(path)
    return df


def evaluate(env, model, save_path, steps=1000, verbose=True):
    result = _evaluate(env, model, steps, verbose)
    df = result_to_csv(save_path, result)
    print(df)


def read_csv(path, keys):
    df = pd.read_csv(path)
    if isinstance(keys, str):
        frame = df[keys]
        return np.mean(frame)
    try:
        rets = {}
        for k in keys:
            rets[k] = read_csv(path, k)
        return rets
    except TypeError:
        exit(-1)


def read_csv_in_folders(folder, keys, suffix=".csv"):
    filenames = os.listdir(folder)
    results = []
    for name in filenames:
        try:
            match = (name[-len(suffix):] == suffix)
            if match:
                results.append((name, read_csv(path=f"{folder}/{name}", keys=keys)))
        except IndexError:
            pass
    results.sort(key=lambda x: x[0])
    return results
