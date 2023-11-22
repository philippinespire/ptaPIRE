#### draw a deme ####

import msprime
import demesdraw
import matplotlib.pyplot as plt
import numpy

t_historic_samp=110
gentime=3
ne_ancestral=1000
r_modern=-0.1
t_recent_change=80
t_ancestral_change=300
r_ancestral=0.01

ne_historic=ne_ancestral/numpy.exp(-(r_ancestral)*((t_ancestral_change-t_historic_samp)/gentime))
ne_contemp=ne_historic/numpy.exp(-(r_modern)*(t_recent_change/gentime))
dem=msprime.Demography()
dem.add_population(name="C",initial_size=ne_contemp)
dem.add_population_parameters_change(time=0, growth_rate=r_modern)
dem.add_population_parameters_change(time=(t_recent_change/gentime), growth_rate=0)
dem.add_population_parameters_change(time=(t_historic_samp/gentime), growth_rate=r_ancestral)
dem.add_population_parameters_change(time=(t_ancestral_change/gentime), growth_rate=0)

graph = msprime.Demography.to_demes(dem)
fig, ax = plt.subplots()  # use plt.rcParams["figure.figsize"]
demesdraw.tubes(graph, ax=ax, seed=1)
plt.show()
