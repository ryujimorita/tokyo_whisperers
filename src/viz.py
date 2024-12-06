from typing import Literal
import os
import shutil
from pathlib import Path
import pandas as pd
from matplotlib.pylab import plt
from loguru import logger

FIG_SIZE = (18, 9)


class Visualization():
    """Class for visualizing loss, evaluation metrics etc."""
    
    def __init__(self, output_dir:str):
        # turn interactive plotting off
        plt.ioff()

        self.output_dir = Path(output_dir)

        # create image folder
        image_folder = self.output_dir / "images"

        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        else:
            shutil.rmtree(image_folder)
            os.makedirs(image_folder)
        
        # self.df = pd.read_csv(train_history_path)
        self.df = self._prepare_data()

        self.column_mapping = {
            "loss": ["train_loss", "eval_loss"],
            "metrics": ["eval_wer", "eval_cer"],
            "learning_rate": ["learning_rate"]
        }

    def _prepare_data(self):
        """Load the train_history.csv and clean up."""
        df = pd.read_csv(self.output_dir / "train_history.csv")
        
        # get only columns we use for plot
        use_columns = [
            "loss", "learning_rate", "epoch", "step", "eval_loss", "eval_wer", 
            "eval_cer"
        ]
        df = df[use_columns].rename(columns={'loss': 'train_loss'})

        # handle duplicate rows for the same step by aggregating values
        df = df.groupby('step').mean(numeric_only=True)

        return df

    def save_image(self, plot_type:Literal["loss", "metrics", "learning_rate"]):
        """Save given type of line chart"""
        defined_plot_type = set(self.column_mapping.keys())

        assert plot_type in defined_plot_type, \
        f"plot_type needs to be the one from {defined_plot_type}"
        
        # plot and label the training and validation loss values
        column_list = self.column_mapping[plot_type]
        fig = plt.figure()
        
        for column_name in column_list:
            # eval has too many NaN and it won't show up. this is messy but works
            temp = self.df[column_name].dropna()
            
            plt.plot(temp.index, temp, label=column_name)
        
        # add in a title and axes labels
        title = " & ".join(column_list) 
        plt.title(f"{plot_type.capitalize()} : {title}")
        plt.xlabel("Step")
        plt.ylabel(plot_type)
        
        # display the plot
        plt.legend(loc='best')
        
        # save the figure if save_path is provided
        save_path = self.output_dir / f"images/{plot_type}.png"
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
        # close the figure so it never gets displayed
        plt.close(fig)

    def save_all(self):
        for plot_type in self.column_mapping.keys():
            self.save_image(plot_type)