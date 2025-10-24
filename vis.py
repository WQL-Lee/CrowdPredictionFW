import numpy as np
from matplotlib import pyplot as plt
# plt.rcParams['figure.constrained_layout.use'] = True

from matplotlib import ticker
import json
from utils import read_json
import os
import glob
from pathlib import Path
import pandas as pd
import matplotlib.gridspec as gridspec

class Visualize:
    def __init__(self, vis_config = "config/vis/CrowdCNNGRU.json"):
        vis_config= Path(vis_config)
        if vis_config.is_file():
            config = read_json(vis_config)
            self.name = config["name"]
            self.saved_dir = config["saved_dir"]
            self.tgt_pred = config["tgt_pred"]
        else:
            print("Warning: visulization configuration file is not found in {}.".format(vis_config))
            exit(-1)

    def tgt_pred_vis(self):
        assert self.tgt_pred is not None, "Error: No parameter for visulizing target and prediction!"
        saved_sub_dir = self.tgt_pred["saved_sub_dir"]
        input_path = self.tgt_pred["input_path"]
        area_titles = self.tgt_pred["title"]
        num_areas = len(area_titles)

        if os.path.isdir(input_path):
            result_files = glob.glob(os.path.join(input_path, "*.json"))
            result_files.sort()
        else:
            result_files = [input_path]
        
        saved_sub_dir_path = os.path.join(self.saved_dir, saved_sub_dir)
        if not os.path.exists(saved_sub_dir_path):
            os.makedirs(saved_sub_dir_path)
        
        for filepath in result_files:
            with open(filepath, 'r', encoding='utf-8') as js:
                result = json.load(js)

            target=np.zeros((len(result),num_areas ))
            pred=np.zeros((len(result),num_areas ))

            for i,data in enumerate(result):
                target[i] = data['target'][1]
                pred[i] = data['prediction'][1]

            input_filename = os.path.basename(filepath)
            filename_wo_ext, _ = os.path.splitext(input_filename)
            saved_filename = f"{filename_wo_ext}.pdf"
            saved_filepath = os.path.join(saved_sub_dir_path, saved_filename)
            self.plot_subplots(pred, target, saved_filepath, f"Performance of the model {self.name}", sub_titles=area_titles)


    def plot_subplots(self, pred, tgt, saved_filepath, fig_title="Main Title", sub_titles=None, xlabel="X Label", ylabel="Y Label", extra_text="Additional information below the title."):
        assert pred.shape == tgt.shape, "The shape of the predicted result and target result is unmatched!"
        
        num_subplots = tgt.shape[1]  # 根据列数决定子图数量
        num_rows = (num_subplots + 1) // 2  # 计算行数，每行最多两个子图
        
        # 图像尺寸设置
        fig_width_per_subplot = 10  # 每个子图的宽度
        fig_height_per_subplot = 4  # 每个子图的高度
        
        if num_subplots == 1:
            fig_width = fig_width_per_subplot
            fig_height = fig_height_per_subplot
        else:
            fig_width = fig_width_per_subplot * 2
            fig_height = fig_height_per_subplot * num_rows
        
        # 使用 GridSpec 设置子图间距
        fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
        
        # 设置大标题，字体加粗
        fig.suptitle(fig_title, fontsize=20, fontweight='bold', y= 1.03)  # 大标题加粗
        
        # 添加额外的文本在大标题下面
        # fig.text(0.5, 1.005, extra_text, ha='center', fontsize=12, color='gray')  # 额外的文本放在大标题下
        
        # 调整子图和标题之间的距离
        # fig.subplots_adjust(top=0.95)  # 通过调整 top 参数，增加大标题与子图的距离
        # fig.subplots_adjust(left=0.2, right=0.9, top=0.95, bottom=0.1)  # 通过调整 top 参数，增加大标题与子图的距离
        
        # 创建一个 GridSpec 网格
        gs = gridspec.GridSpec(num_rows, 2, figure=fig, wspace=0.1, hspace=0.15)  # 设置子图间距

        if num_subplots == 1:
            ax = fig.add_subplot(gs[0, 0])
            ax.plot(tgt[:, 0], label=f'ground truth', c="r")
            ax.plot(pred[:, 0], label=f'prediction', c="b")
            if sub_titles:
                ax.set_title(sub_titles[0], fontsize=14, fontweight='bold')  # 小标题加粗
            ax.legend(loc='upper right')
        else:
            for i in range(num_subplots):
                row_i = i // 2
                col_j = i % 2
                ax = fig.add_subplot(gs[row_i, col_j])
                ax.plot(tgt[:, i], label=f'ground truth', c="r")
                ax.plot(pred[:, i], label=f'prediction', c="b")
                if sub_titles:
                    ax.set_title(sub_titles[i], fontsize=14, fontweight='bold')  # 小标题加粗
                ax.legend(loc='upper right')

        plt.savefig(fname=saved_filepath, bbox_inches='tight')
        # plt.show()
        plt.close()

if __name__=="__main__":
    visualize = Visualize("config/vis/A3TGCN.jsonc")
    visualize.tgt_pred_vis()


