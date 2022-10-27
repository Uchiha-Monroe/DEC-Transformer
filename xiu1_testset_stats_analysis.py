import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle

with open('all_file_dict_test', 'rb') as f:
    afd = pickle.load(f)

all_classes = afd.keys()
new_dict = {
    "class": [],
    "len": []
}

for single_class in all_classes:
    for single_doc in afd[single_class]:
        doc_class = single_class
        doc_len = len(single_doc)
        new_dict["class"].append(doc_class)
        new_dict["len"].append(doc_len)

df_test = pd.DataFrame(new_dict)


fig1, ax1 = plt.subplots(figsize=(10, 5))
fig1.subplots_adjust(bottom=0.3)
ax1 = sns.boxplot(
    data=df_test,
    x="class",
    y="len"
)

ax1.tick_params(axis='x', labelrotation=90)
ax1.set_xlabel("")

fig1.savefig(
    fname="experiment_figures/test_box_plot.eps",
    bbox_inches=None,
    pad_inches=0.2
)



fig2, ax2 = plt.subplots(figsize=(10, 5))
fig2.subplots_adjust(bottom=0.3)

ax2 = sns.countplot(data=df_test, x="class")
ax2.set_xlabel("")
ax2.tick_params(axis='x', labelrotation=90)

fig2.savefig(
    fname="experiment_figures/test_count_plot.eps",
    bbox_inches=None,
    pad_inches=0.2    
)

print("Done!")