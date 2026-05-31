from pathlib import Path

import numpy as np


def draw_features(features: dict, features_name: list, save_dir: str):
    """
    features: np.array in list
    """
    # 放在里面是有原因的，就是不希望影响到外面的设置
    import matplotlib
    matplotlib.use("Agg")
    # matplotlib.rc("font", family='AR PL UMing CN')
    from matplotlib import pyplot as plt

    Path(save_dir).mkdir(parents=True, exist_ok=True)  # 确保父目录存在
    nbr = len(features_name)
    for well_name in features:
        fig = plt.figure(figsize=(120, 30))  # 12000个像素，很大
        fig.suptitle(well_name)
        for i, key in zip(range(nbr), features_name):
            if key in features[well_name]:
                plt.subplot(nbr, 1, i + 1)
                x = np.arange(len(features[well_name][key]))
                y = features[well_name][key][:, 0]
                plt.plot(x, y)
                plt.title(key)
                plt.tight_layout()

        save_path = Path(save_dir) / (well_name + ".svg")
        plt.savefig(str(save_path), dpi=6000, format='svg')  # 保存成svg, dpi 6000 很大
        plt.close()
