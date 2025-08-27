import ipyparallel as ipp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PTA

ipyclient = ipp.Client(cluster_id="PTA-nhap1244_muhi_sigmalo_pta0020")
print(len(ipyclient))

for it in range(1,61):
	simname="npops7_nhap1244_muhi_sigmalo_it"+str(it)
	sim = PTA.DemographicModel_2D_Temporal(simname)
	sim.set_param("npops", 7)
	sim.set_param("nsamps", [12,44])
	sim.set_param("muts_per_gen", 0.00000001)
	sim.set_param("generation_time", [3,1,1,1,0.5,2,2])
	sim.set_param("ne_ancestral", [100000,4000000])
	sim.set_param("r_modern_mu", [0, 0.1])
	sim.set_param("r_modern_sigma", [0,0.05])
	sim.set_param("num_replicates",[25,268,258,266,34,94,221])
	sim.set_param("length",1500)
	print(sim.get_params())
	nsims = 10000
	sim.simulate(nsims=nsims,ipyclient=ipyclient)

print("Simulations finished!")
