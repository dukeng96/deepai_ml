import matplotlib.pyplot as plt
import pandas as pd

style = 'ggplot' # 'seaborn-white' 'dark_background', 'bmh', 'ggplot', 'fivethirtyeight'

plt.style.use(style)

plt.rcParams.update({'font.family': 'serif',

                     'font.serif': 'Ubuntu',

                     'font.monospace':'Ubuntu Mono',

                     'font.size':12,

                     'axes.labelsize':16,

                     'axes.labelweight':'bold',

                     'axes.titlesize':16,

                     'xtick.labelsize':12,

                     'ytick.labelsize':12,

                     'legend.fontsize':10,

                     'figure.titlesize':12})

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)