#http://alleninstitute.github.io/AllenSDK/_static/examples/nb/brain_observatory.html#Experiment-Containers

#Initialize
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import pprint

#Download
boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
print('== Experiments Details ==\n')

# Download a list of all targeted areas
targeted_structures = boc.get_all_targeted_structures()
all_tar = targeted_structures 
print("all targeted structures: " + str(targeted_structures)+"\n")
#print("all targeted structures: ")
#print([str(i+1) + ' : ' + x.encode('utf8') for i,x in enumerate(targeted_structures)])

# Download a list of all imaging depths
depths = boc.get_all_imaging_depths()
print("all imaging depths: " + str(depths)+"\n")

# Download a list of all cre driver lines 
cre_lines = boc.get_all_cre_lines()
print("all cre lines:\n")
pprint.pprint(cre_lines)
print("\n")

# Download a list of all stimuli
stims = boc.get_all_stimuli()
print("all stimuli:\n")
pprint.pprint(stims)
print("\n")

"""
# Download experiment containers for VISp experiments
# visp_ecs = boc.get_experiment_containers(targeted_structures=['VISp'])
# print("all VISp experiment containers: %d" % len(visp_ecs))
"""

print("Please selected a targeted area")
print ("\n".join("{}: {}".format(i+1,x) for i,x in enumerate(stims)))
#choice = input("Please enter the stimuli of your choice(1-8) : ") # --------------------- choose stimuli types 
choice = 6
choice = choice  -1 
print("\nDownloading data for %s" %stims[choice])

# Goal : Want to download all experiments for natural scenes
#        And also, get the data for each type of stimuli


import allensdk.brain_observatory.stimulus_info as stim_info

exps = boc.get_ophys_experiments(targeted_structures= all_tar,stimuli=[getattr(stim_info,stims[choice].upper())], cre_lines=cre_lines)
print("Experiment with %s:"%stims[choice])
pprint.pprint(exps)
print("Total = %d" %len(exps))

# Me : Note -- what's the different between experiment containers and the ophys_experiments

#Note : Info for each experiment --- Example
# ecs[0]
# {'imaging_depth': 175, 'experiment_container_id': 511510998, 'specimen_name': u'Cux2-CreERT2;Camk2a-tTA;Ai93-229105', 'reporter_line': u'Ai93(TITL-GCaMP6f)', 'targeted_structure': u'VISal', 'cre_line': u'Cux2-CreERT2', 'session_type': u'three_session_B', 'donor_name': u'229105', 'id': 506156402, 'age_days': 105.0} --> dict 

# For each experiment --> call by ID
expData = boc.get_ophys_experiment_data(exps[0]['id']) 

#get dF/F trace
time, df_f = expData.get_dff_traces()

#Note : for df_f ---> len(df_f) = # of cells  then df_f[0] = traces for that cell --> unit of time? 
# Another issue is the data did not arrange properly ---> if you want to order them by area or depth --> have to write codfe to do this

#print dF/F trace 
from matplotlib import pyplot as plt

"""
dsi_cell_id = dsi_cell['cell_specimen_id']
time, raw_traces = data_set.get_fluorescence_traces(cell_specimen_ids=[dsi_cell_id])
_, neuropil_traces = data_set.get_neuropil_traces(cell_specimen_ids=[dsi_cell_id])
_, corrected_traces = data_set.get_corrected_fluorescence_traces(cell_specimen_ids=[dsi_cell_id])
_, dff_traces = data_set.get_dff_traces(cell_specimen_ids=[dsi_cell_id])

# plot raw and corrected ROI trace
#plt.figure(figsize=(14,4))
plt.title("Raw Fluorescence Trace")
plt.plot(time, raw_traces[0])
plt.show()

plt.figure(figsize=(14,4))
plt.title("Neuropil-corrected Fluorescence Trace")
plt.plot(time, corrected_traces[0])
plt.show()


plt.figure(figsize=(14,4))
plt.title("dF/F Trace")
# warning: dF/F can occasionally be one element longer or shorter 
# than the time stamps for the original traces.
plt.plot(time[:len(df_f[0])], df_f[0])
plt.show()
"""



# ROI Masks --- look at the cell
"""
ROI Masks
If you want to take a look at the cell visually, you can open the NWB file and extract a pixel mask. 
You can also pull out the maximum intensity projection of the movie for context
https://neurodatawithoutborders.github.io/
"""

import numpy as np
#import pdb; pdb.set_trace()

#Test Plotting ROI
data_set = boc.get_ophys_experiment_data(506156402)

# get the specimen IDs for a few cells
cids = data_set.get_cell_specimen_ids()[:15:5]

# get masks for specific cells
roi_mask_list = data_set.get_roi_mask(cell_specimen_ids=cids)

# plot each mask
f, axes = plt.subplots(1, len(cids)+2, figsize=(15, 3))

# i = 2; ax = axes[i]; roi_mask = roi_mask_list[i]; cid = cids[i]  #For testing
for ax, roi_mask, cid in zip(axes[:-2], roi_mask_list, cids):
    ax.imshow(roi_mask.get_mask_plane(), cmap='gray')
    ax.set_title('cell %d' % cid)

# make a mask of all ROIs in the experiment    
all_roi_masks = data_set.get_roi_mask_array()
combined_mask = all_roi_masks.max(axis=0)

axes[-2].imshow(combined_mask, cmap='gray')
axes[-2].set_title('all ROIs')

# show the movie max projection
max_projection = data_set.get_max_projection()
axes[-1].imshow(max_projection, cmap='gray')
axes[-1].set_title('max projection')

plt.show()

"""
Natural Scenes
The natural scenes analysis object is again similar to the others. 
In addition to computing the sweep_response and mean_sweep_response arrays,
NaturalScenes reports the cell's preferred scene, running modulation, time to peak response, and other metrics.
"""

# Me : Goal = "predict cell activity" 
# Here(Kachio) predict image from cell activity
import pdb; pdb.set_trace()

from allensdk.brain_observatory.natural_scenes import NaturalScenes

data_set = boc.get_ophys_experiment_data(510938357)
import time
download_time = time.time()
ns = NaturalScenes(data_set)
print("done analyzing natural scenes") # save ns in Brain_Observatory folder
print "Time %s" %(time.time() - download_time)
import pickle 
def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

##############################################################################################
# https://alleninstitute.github.io/AllenSDK/_static/examples/nb/brain_observatory_stimuli.html 
# #511510667
##############################################################################################
#Natural Scene stimulus
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from allensdk.brain_observatory.natural_scenes import NaturalScenes
boc = BrainObservatoryCache(manifest_file='boc/manifest.json')

import pprint

# PROOBLEM with download NWB file -->  https://github.com/AllenInstitute/AllenSDK/issues/22 
# http://api.brain-map.org/api/v2/data/OphysExperiment/501498760.xml?include=well_known_files
# http://api.brain-map.org/api/v2/well_known_file_download/514422179
# /api/v2/well_known_file_download/514422179

# Download Experiment Container for specific ID
ids=511510667
exps = boc.get_ophys_experiments(experiment_container_ids=[511510667]) 
print("Experiments for experiment_container_id %d: %d\n" % (ids, len(exps)))
pprint.pprint(exps)

import time
download_time = time.time()
data_set = boc.get_ophys_experiment_data(501498760)  #501498760
print "Download complete: Time %s" %(time.time() - download_time)
scene_nums = [4, 83]



# read in the array of images
scenes = data_set.get_stimulus_template('natural_scenes') # ---> Here = scenes data

# display a couple of the scenes
fig, axes = plt.subplots(1,len(scene_nums))
for ax,scene in zip(axes, scene_nums):
    ax.imshow(scenes[scene,:,:], cmap='gray')
    ax.set_axis_off()
    ax.set_title('scene %d' % scene)
