####### function for simulating joint sfs with temporal sampling and recent + historic size changes #######
####### now including recombination (n RAD-like loci, assumed to be unlinked) ##############
####### requires: msprime, numpy ################
####### note: "n_albatross" and "n_contemporary" refer to historic and contemporary sample sizes, respectively ########


def jsfs_sim(n_contemp=2,n_albatross=2,n_loci=1000,t_historic_samp=110,gentime=3,mu=0.00001,ne_ancestral=1000,r_modern=-0.1,t_recent_change=80,t_ancestral_change=15000,r_ancestral=0):
     ne_historic=ne_ancestral/numpy.exp(-(r_ancestral)*((t_ancestral_change-t_historic_samp)/gentime))
     ne_contemp=ne_historic/numpy.exp(-(r_modern)*(t_recent_change/gentime))
     contemp_list=list(range(n_contemp*2))
     albatross_list=[x+n_contemp*2 for x in list (range(n_albatross*2))]
     contemporary_sampset=msprime.SampleSet(n_contemp)
     albatross_sampset=msprime.SampleSet(n_albatross, time=round(t_historic_samp/gentime))
     dem=msprime.Demography()
     dem.add_population(name="C",initial_size=ne_contemp)
     dem.add_population_parameters_change(time=0, growth_rate=r_modern)
     dem.add_population_parameters_change(time=(t_recent_change/gentime), growth_rate=0)
     dem.add_population_parameters_change(time=(t_historic_samp/gentime), growth_rate=r_ancestral)
     dem.add_population_parameters_change(time=(t_ancestral_change/gentime), growth_rate=0)
     history=msprime.DemographyDebugger(demography=dem)
     print(history)
     n_sites=n_loci*100-1
     rateseq=[0,0.5]*n_loci
     unlinkedloci_rates=rateseq[:-1]
     loci_startpoints=[x*100 for x in list(range(n_loci))]
     loci_endpoints=[(x+1)*100 -1 for x in list(range(n_loci))]
     loci_boundaries=sorted(loci_startpoints+loci_endpoints)
     rate_map=msprime.RateMap( position=loci_boundaries,rate=unlinkedloci_rates)
     ts = msprime.sim_ancestry(
         samples = [contemporary_sampset,albatross_sampset],
         demography=dem,
         recombination_rate=rate_map,
         sequence_length=n_sites
     )
     print(ts.draw_text())
     mts=msprime.sim_mutations(ts, rate=mu)
     jsfs=mts.allele_frequency_spectrum(sample_sets=[contemp_list,albatross_list],mode="site")
     print(jsfs)
