# -*- coding: utf-8 -*-
"""
Created on Tue May 10 14:28:04 2022

@author: Nadja
"""

import os
os.chdir(".\\")
from functions import *
import matplotlib.lines as mlines
os.chdir(".\\Data")
data1 = np.load("Training_Babies.npz")
X = data1["T1"]
Labels = data1["Labels"]
# read in the probability masks from all 3 views
data_a = np.load("Results_axial_crossvalidation.npz")
data_c = np.load("Results_coronal.npz")
data_s = np.load("Results_sagittal.npz")


# np.savez_compressed("All_directions_no_th.npz", reconstruction_cor=reconstruction_cor_noth,
c = data_c["reconstruction_cor"]
s = data_s["reconstruction_sag"]
a = data_a["Results"]

# Here we sum up the probability masks from all three directions as described in Figure 5 of the paper
combi= a + c + s
# find the best threshold for final output mask
all_data=find_best_dice(combi,Labels) 
labels = ["1/3","2/3","1","1.25","4/3","1.45","5/3", "2","7/3", "8/3"]
fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))

# rectangular box plot
bplot1 = ax1.boxplot(all_data,
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels,showfliers=False,showmeans=True)  # will be used to label x-ticks

# fill with colors
colors = ["lemonchiffon","lemonchiffon","lemonchiffon","lemonchiffon","lemonchiffon","lemonchiffon","lemonchiffon","lemonchiffon"]

for patch, color in zip(bplot1['boxes'], colors):
    patch.set_facecolor(color)
blue_line = mlines.Line2D([], [], color='green', marker='^',linestyle="None",
                          markersize=5, label='mean')
orange_line = mlines.Line2D([], [], color='orange', marker='_',linestyle="None",
                          markersize=5, label='median')
plt.legend(handles=[blue_line,orange_line],loc = "lower left",numpoints=1)
ax1.yaxis.grid(True)
#ax1.set_xlabel('Method')
ax1.set_ylabel('Dice score')
ax1.set_xlabel("Threshold")
plt.show()   
#%-%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%% Plot dcie of all 4 mthods
dice_a = evaluate_dice(np.round(a),Labels)[0]
dice_c = evaluate_dice(np.round(c),Labels)[0]
dice_s = evaluate_dice(np.round(c),Labels)[0]
combi[combi<5/3]=0
combi[combi>0]=1
e = evaluate_dice(combi,Labels)[0]
all_data = [dice_a,dice_s,dice_c,e]
labels = ["axial","sagittal","coronal","combi"]
fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(3, 5))
# rectangular box plot
bplot1 = ax1.boxplot(all_data,
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels,showfliers=False,showmeans=True)  # will be used to label x-ticks
#ax1.set_title('Dice Scores')
blue_line = mlines.Line2D([], [], color='green', marker='^',linestyle="None",
                          markersize=5, label='mean')
orange_line = mlines.Line2D([], [], color='orange', marker='_',linestyle="None",
                          markersize=5, label='median')
plt.legend(handles=[blue_line,orange_line],loc = "lower left",numpoints=1)
# fill with colors
colors = ['lightblue', 'lightblue', 'lightblue',"lemonchiffon"]
for patch, color in zip(bplot1['boxes'], colors):
    patch.set_facecolor(color)
ax1.yaxis.grid(True)
ax1.set_xlabel('Method')
ax1.set_ylabel('Dice score')
plt.show()


#%% plot F1- score for alle methods
#%%%%%%%% Plot dcie of all 4 mthods
f1_a = evaluate_recall(np.round(a),Labels)
f1_c = evaluate_recall(np.round(c),Labels)
f1_s = evaluate_recall(np.round(s),Labels)

e = evaluate_recall(combi,Labels)
all_data = [f1_a,f1_s,f1_c,e]
labels = ["axial","sagittal","coronal","combi"]

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(3, 5))

# rectangular box plot
bplot1 = ax1.boxplot(all_data,
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels,showfliers=False,showmeans=True)  # will be used to label x-ticks
#ax1.set_title('Dice Scores')
blue_line = mlines.Line2D([], [], color='green', marker='^',linestyle="None",
                          markersize=5, label='mean')
orange_line = mlines.Line2D([], [], color='orange', marker='_',linestyle="None",
                          markersize=5, label='median')
plt.legend(handles=[blue_line,orange_line],loc = "lower left",numpoints=1)

colors = ['lightblue', 'lightblue', 'lightblue',"lemonchiffon"]

for patch, color in zip(bplot1['boxes'], colors):
    patch.set_facecolor(color)
ax1.yaxis.grid(True)
ax1.set_xlabel('Method')
ax1.set_ylabel('Recall')
plt.show()

os.chdir("C://Users//nadja//Documents//PLIC_programm/Results_2/crossvalidation")

data= np.load("All_directions_no_th.npz")
# np.savez_compressed("All_directions_no_th.npz", reconstruction_cor=reconstruction_cor_noth,
combi=data["combi"]
all_data1=find_best_precision(combi,Labels) 
all_data=[]
#for i in range(0,len(all_data1)):
#    a=all_data1[i][all_data1[i]<3.5]
#    all_data.append(a)
all_data=np.asarray(all_data1)
    
labels = ["1/3","2/3","1","4/3","5/3", "2","7/3", "8/3"]

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))

# rectangular box plot
bplot1 = ax1.boxplot(all_data,
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels,showfliers=False,showmeans=True)  # will be used to label x-ticks

# fill with colors
colors = ["lemonchiffon","lemonchiffon","lemonchiffon","lemonchiffon","lemonchiffon","lemonchiffon","lemonchiffon","lemonchiffon"]

for patch, color in zip(bplot1['boxes'], colors):
    patch.set_facecolor(color)
blue_line = mlines.Line2D([], [], color='green', marker='^',linestyle="None",
                          markersize=5, label='mean')
orange_line = mlines.Line2D([], [], color='orange', marker='_',linestyle="None",
                          markersize=5, label='median')
plt.legend(handles=[blue_line,orange_line],loc = "lower left",numpoints=1)


ax1.yaxis.grid(True)
#ax1.set_xlabel('Method')
ax1.set_ylabel('Precision')
ax1.set_xlabel("Threshold")
plt.show()   


'%%%%%Jaccard plots'
jc_a = np.asarray(evaluate_jc(np.round(a),Labels))

jc_c = np.asarray(evaluate_jc(np.round(c),Labels))

jc_s = np.asarray(evaluate_jc(np.round(s),Labels))

e = np.asarray(evaluate_jc(combi,Labels))
#e = e[e<3.5]


all_data = [jc_a,jc_s,jc_c,e]
labels = ["axial","sagittal","coronal","combi"]

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(3, 5))

# rectangular box plot
bplot1 = ax1.boxplot(all_data,
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels,showfliers=False,showmeans=True)  # will be used to label x-ticks
colors = ['lightblue', 'lightblue', 'lightblue',"lemonchiffon"]

for patch, color in zip(bplot1['boxes'], colors):
    patch.set_facecolor(color)
blue_line = mlines.Line2D([], [], color='green', marker='^',linestyle="None",
                          markersize=5, label='mean')
orange_line = mlines.Line2D([], [], color='orange', marker='_',linestyle="None",
                          markersize=5, label='median')
plt.legend(handles=[blue_line,orange_line],loc = "lower left",numpoints=1)
ax1.yaxis.grid(True)
ax1.set_xlabel('Method')
ax1.set_ylabel('Jaccard index')
plt.show()


os.chdir("C://Users//nadja//Documents//PLIC_programm/Results_2/crossvalidation")

data= np.load("All_directions_no_th.npz")
# np.savez_compressed("All_directions_no_th.npz", reconstruction_cor=reconstruction_cor_noth,
combi=data["combi"]
all_data1=find_best_jc(combi,Labels) 
all_data=[]
for i in range(0,len(all_data1)):
    a=all_data1[i][all_data1[i]<3.5]
    all_data.append(a)
all_data=np.asarray(all_data)
    
labels = ["1/3","2/3","1","4/3","5/3", "2","7/3", "8/3"]

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))

# rectangular box plot
bplot1 = ax1.boxplot(all_data,
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels,showfliers=False,showmeans=True)  # will be used to label x-ticks

# fill with colors
colors = ["lemonchiffon","lemonchiffon","lemonchiffon","lemonchiffon","lemonchiffon","lemonchiffon","lemonchiffon","lemonchiffon"]

for patch, color in zip(bplot1['boxes'], colors):
    patch.set_facecolor(color)
blue_line = mlines.Line2D([], [], color='green', marker='^',linestyle="None",
                          markersize=5, label='mean')
orange_line = mlines.Line2D([], [], color='orange', marker='_',linestyle="None",
                          markersize=5, label='median')
plt.legend(handles=[blue_line,orange_line],loc = "lower left",numpoints=1)


ax1.yaxis.grid(True)
#ax1.set_xlabel('Method')
ax1.set_ylabel('Jaccard index')
ax1.set_xlabel("Threshold")
plt.show()   

#-------------------------------------------------------------#




