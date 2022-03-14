import json
# import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# % matplotlib inline

sns.set_style("whitegrid")
sns.set_context("paper")
# 设置风格、尺度

# import warnings
# warnings.filterwarnings('ignore')
# 不发出警告
import os.path
path = 'alice_ae_user_correlation'

if not os.path.isfile(path+f'.tsv'):
    # print ("File exist")
    Proj_dir = os.path.abspath(os.path.join(os.getcwd(), "../../../"))

    brain2txt = json.load(open(Proj_dir +'/'+path+ f'.json', 'r'))
    df = pd.DataFrame(columns=['user', 'correlation'])
    for k, v in brain2txt.items():
        # 插入数据
        # df['user'] = k
        print(k, np.mean(v))
        for i in range(len(v)):
            df = df.append({"user": k, "correlation": v[i]}, ignore_index=True)
            # df[f'{i}'] = v[i]
    df.sort_values(by="user", ascending=False)
    df.to_csv(path+f'.tsv', index=False, sep='\t')
else:
    df = pd.read_csv(path+f'.tsv', sep='\t', header=0)
    df.columns = ['user', 'correlation']
    df.sort_values(by="user", ascending=False)

    # df = pd.DataFrame(columns=['user', 'correlation'])


sns.stripplot(x="user",          # x → 设置分组统计字段
              y="correlation",   # y → 数据分布统计字段
              # 这里xy数据对调，将会使得散点图横向分布
              data= df,        # data → 对应数据
              jitter = True,    # jitter → 当点数据重合较多时，用该参数做一些调整，也可以设置间距如：jitter = 0.1
              size = 5, edgecolor = 'w',linewidth=1,marker = 'o'  # 设置点的大小、描边颜色或宽度、点样式
              )
plt.show()
# plt.savefig("")

sns.catplot(x="user", y="correlation",data=df, kind="box")
plt.savefig(path+f'.png')
plt.show()
