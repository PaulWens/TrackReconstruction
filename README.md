# TrackReconstruction
Bayesian track reconstruction for high-resolution animal movement data

This repository includes several versions of the model described in Ref. [1]. There was a small bug in the code attached to the paper which stopped the model from working in more recent JAGS versions. It also include an example data set (humpback whale mn12_178).

Data processing and visualision is in Matlab and mat2jags in used to interface with JAGS. JAGS itself is required to fit the models.

Start with "reconstruct_track_examples.m" and let me know if you run into problems (pjw [at] hi [dot] is)

% Ref [1]. Wensveen PJ, Thomas L, Miller PJO 2015. A path reconstruction method integrating dead-reckoning and position fixes applied to humpback whales. Movement Ecology 3:31

