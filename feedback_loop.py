"""Simulates an example network of 450 cell receiving two kinds of exernal input as defined in the configuration file"""
import numpy as np
import os
import sys
from bmtk.simulator import bionet
from bmtk.simulator.bionet.modules.sim_module import SimulatorMod
from bmtk.analyzer.spike_trains import plot_raster
from bmtk.simulator.bionet.biocell import BioCell
from bmtk.simulator.bionet.pointprocesscell import PointProcessCell
from bmtk.utils.reports.spike_trains import PoissonSpikeGenerator
from bmtk.simulator.bionet.io_tools import io

from neuron import h
import math


pc = h.ParallelContext()


class FeedbackLoop(SimulatorMod):
    def __init__(self):
        self._spike_events = {}
        self._synapses = {}
        self._netcons = {}
        self._spike_records = {}
        self._vect_stims = {}
        self._spikes = {}

        self._block_length_ms = 0.0
        self._n_cells = 0
        
        self.blad_fr = 0.0
        self.pag_fr = 0.0
        self._prev_glob_press = 0.0
        self._glob_press = 0.0 
        self.times = []
        self.b_vols = []
        self.b_pres = []
        
        self.void = False 
        self.fill = True
        
        #self.voidtime
        

    def _set_spike_detector(self, sim):
        for gid, cell in sim.net.get_local_cells().items():
            tvec = sim._spikes[gid]
            self._spike_records[gid] = tvec

    def initialize(self, sim):
        self._block_length_ms = sim.nsteps_block*sim.dt
        self._n_cells = len(sim.net.get_local_cells())

        self._spikes = h.Vector()  # start off with empty input
        vec_stim = h.VecStim()
        vec_stim.play(self._spikes)
        self._vect_stim = vec_stim

        for gid, cell in sim.net.get_local_cells().items():
            self._spike_events[gid] = np.array([])
            # For each cell we setup a network connection, NetCon object, that stimulates input as a series of spike
            # events mimicking a synapse. For this simple example each cell recieves only 1 virtual synapse/netcon.
            # To have more than 1 netcon in each cell you can add an extra internal loop, and _synapses and _netcons
            # will be a dictionary of lists
            if isinstance(cell, BioCell) and gid < 20: ################################################################## CHANGE TO , 20 FOR PAG
                # For biophysicaly detailed cells we use an Synapse object that is placed at the soma. If you want to
                # place it at somewhere different than the soma you can use the following code:
                #   seg_x, sec_obj = cell.morphology.find_sections(
                #       sections_names=[axon, soma, dend, apic],
                #       distance_ranges=[0.0, 1000.0]
                #   )
                #   syn = h.Exp2Syn(seg_x, sec=sec_obj
                syn = h.Exp2Syn(0.5, sec=cell.hobj.soma[0])
                syn.e = 0.0
                syn.tau1 = 0.1
                syn.tau2 = 0.3
                self._synapses[gid] = syn

                # create a NetCon connection on the synpase using the array of spike-time values
                nc = h.NetCon(vec_stim, syn)
                nc.threshold = sim.net.spike_threshold
                nc.weight[0] = 0.2
                nc.delay = 1.0
                self._netcons[gid] = nc

        self._set_spike_detector(sim)
        pc.barrier()

    def step(self, sim, tstep):
        pass
    
    def block(self, sim, block_interval):
        block_length = sim.nsteps_block*sim.dt/1000.0
        t = sim.h.t-block_length*1000.0
        
        #### BLADDER EQUATIONS ####    
    # Grill, et al. 2016
        def blad_vol(vol):
            f = 1.5*20*vol -10 #-10 #math.exp(48*vol-64.9) + 8
            return f

        # Grill function returning pressure in units of cm H20
	    # Grill, et al. 2016
        def pressure(fr,v,x):
            p = 0.2*fr +  .5*v - .2*x + 12
            p = max(p,0.0)
            return p 

        # Grill function returning bladder afferent firing rate in units of Hz
	    # Grill, et al. 2016
        def blad_aff_fr(p):
            #fr1 = -3.0E-08*p**5 + 1.0E-5*p**4 - 1.5E-03*p**3 + 7.9E-02*p**2 - 0.6*p
            p_mmHg = 0.735559*p
            
            if p_mmHg < 5:
                fr1 = 0
            elif p_mmHg < 10:
                fr1 = 0.3*p_mmHg
            elif p_mmHg < 30:
                fr1 = 0.9*p_mmHg - 6
            else:
                fr1 = -3.0E-08*p**5 + 1.0E-5*p**4 - 1.5E-03*p**3 + 7.9E-02*p**2 - 0.6*p
            fr1 = max(fr1,0.0)
            return fr1 # Using scaling factor of 5 here to get the correct firing rate range

    ### STEP 1: Calculate PGN Firing Rate ###
        io.log_info(f'Timestep {block_interval[0]*sim.dt} to {block_interval[1]*sim.dt} ms')
        io.log_info('PGN node_id\t  Hz')
        summed_fr = 0
        for gid, tvec in self._spike_records.items():
            # self._spike_records is a dictionary of the recorded spikes for each cell in the previous block of
            #  time. When self._set_spike_detector() is called it will reset/empty the spike times. If you want to
            #  print/save the actual spike-times you can call self._all_spikes[gid] += list(tvec)
            if gid < 80 and gid > 69: #PGN gids 
                n_spikes = len(tvec)
                fr = n_spikes / (self._block_length_ms/1000.0)
                summed_fr += fr
                io.log_info(f'{gid}\t\t{fr}')
        avg_fr = summed_fr / 10.0
        io.log_info(f'PGN firing rate avg: {summed_fr / 10.0} Hz')
        
        # Grill 
        PGN_fr = max(2.0E-03*avg_fr**3 - 3.3E-02*avg_fr**2 + 1.8*avg_fr - 0.5, 0.0)
        io.log_info("Grill PGN fr = {0} Hz".format(PGN_fr))

    ### STEP 2: Calculate IMG Firing Rate ###
        io.log_info('IMG node_gid\t  Hz')
        summed_fr = 0
        for gid, tvec in self._spike_records.items():
            # self._spike_records is a dictionary of the recorded spikes for each cell in the previous block of
            #  time. When self._set_spike_detector() is called it will reset/empty the spike times. If you want to
            #  print/save the actual spike-times you can call self._all_spikes[gid] += list(tvec)
            if gid < 100 and gid > 89: #IMG gids 
                n_spikes = len(tvec)
                fr = n_spikes / (self._block_length_ms/1000.0)
                summed_fr += fr
                io.log_info(f'{gid}\t\t{fr}')
        IMG_avg_fr = summed_fr / 10.0
        io.log_info(f'IMG firing rate avg: {avg_fr} Hz')
        
    ### STEP 3: Volume Calculations ###
        v_init = 0.0       # TODO: get biological value for initial bladder volume
        fill = 1.75 	 	# ml/min (Asselt et al. 2017) 175 microL / min  Herrara 2010 for rat baseline 
        fill /= (1000 * 60) # Scale from ml/min to ml/ms
        void = 46.0 		# 4.344 ml/min approximated from Herrera 2010; can also use 4.6 ml/min (Streng et al. 2002)
        void /= (1000 * 60) # Scale from ml/min to ml/ms
        max_v = 1.65 		# 1.65 ml based of Herrara 2010; 1.5 ml (Grill et al. 2019) #0.76
        vol = v_init
        
        prev_vol = v_init
        block_len_ms = sim.nsteps_block*sim.dt #block length in milliseconds 
        
        # if first timestep where there are no recorded bladder volumes
        if not self.b_vols:
            prev_vol = v_init
        else:
            prev_vol = self.b_vols[-1]
        
        # To switch back from voiding to filling
        if prev_vol == v_init:
            self.void = False
            self.fill = True 
            
        # Voiding
        if self.void:
            vol = prev_vol - void*block_len_ms #max_v - void*(60000-t)*100
        # Filling
        elif self.fill and prev_vol < max_v:  #make this better
            # if first timestep where there are no recorded bladder volumes 
            if not self.b_vols:
                vol = v_init
            else:
                vol = prev_vol + fill*block_len_ms #fill*t*20 + v_init Vinay
        # If max volume reached
        else:
            vol = prev_vol 
            
        
        # Maintain minimum volume
        if vol < v_init:
            vol = v_init
        
        # Grill
        grill_vol = blad_vol(vol)
        
    ### STEP 4: Pressure and Bladder Afferent FR Calculations ###
        x = 0 #50.0*(1.0/(1.0 + math.exp(75*(vol-0.67*max_v))) - 0.5)
        p = pressure(PGN_fr, grill_vol, IMG_avg_fr)
        self.blad_fr = blad_aff_fr(p)
        
    ### STEP 5: Update the input spikes each cell recieves in the next time block
        # Calculate the start and stop times for the next block
        next_block_tstart = block_interval[1]*sim.dt
        next_block_tstop = next_block_tstart+self._block_length_ms

        # For this simple example we just create a randomized series of spike for the next time block for each of the
        #  14 cells. The stimuli input rate (self._current_input_rate) is increamented by 10 Hz each block, for more
        #  realistic simulations you can use the firing-rates calcualted above to adjust the incoming stimuli.
        #print("Calculated Bladder Afferent Firing Rate: {0}".format(self.blad_fr))
        psg = PoissonSpikeGenerator()
        psg.add(
            node_ids= [0,1,2,3,4,5,6,7,8,9],
            firing_rate= self.blad_fr,
            times=(next_block_tstart/1000.0 + 0.01, next_block_tstop/1000.0),
            population= 'Bladaff',
        )
        
        psg.add_spikes([0,1,2,3,4,5,6,7,8,9], [next_block_tstop, next_block_tstop, next_block_tstop, next_block_tstop, next_block_tstop, next_block_tstop, next_block_tstop, next_block_tstop, next_block_tstop, next_block_tstop], population = "Bladaff")
        psg.to_csv("spikes.csv")

        for gid, cell in sim.net.get_local_cells().items():
            if gid < 10:
                spikes = psg.get_times(gid, population='Bladaff')
                spikes = np.sort(spikes)
                #print("HEllo: \n {0}".format(spikes))
                if len(spikes) == 0:
                    continue

            # The next block of code is where we update the incoming/virtual spike trains for each cell, by adding
            # each spike to the cell's netcon (eg synapse). The only caveats is the spike-trains array must
            #  1. Have atleast one spike
            #  2. Be sorted
            #  3. first spike must occur after the delay.
            # Otherwise an error will be thrown.
                self._spike_events[gid] = np.concatenate((self._spike_events[gid], spikes))
                nc = self._netcons[gid]
                for t in spikes:
                    nc.event(t)
                    
        if self.blad_fr > 10:
            io.log_info("!!!PAG FIRING ACTIVATED!!!")
            self.pag_fr = 15
            
            # To switch from filling to voiding
            self.void = True 
            self.fill = False 
            
            # PAG Firing Rate Update 
            psg = PoissonSpikeGenerator()
            psg.add(
                node_ids= [0,1,2,3,4,5,6,7,8,9],
                firing_rate= self.pag_fr,
                times=(next_block_tstart/1000.0 + 0.01, next_block_tstop/1000.0),
                population= 'PAGaff',
            )
        
            psg.add_spikes([0,1,2,3,4,5,6,7,8,9], [next_block_tstop, next_block_tstop, next_block_tstop, next_block_tstop, next_block_tstop, next_block_tstop, next_block_tstop,       next_block_tstop, next_block_tstop, next_block_tstop], population = "PAGaff")
            psg.to_csv("spikes_pag.csv")
            #self._current_input_rate += 10.0

            for gid, cell in sim.net.get_local_cells().items():
                if gid < 20 and gid > 9:
                    spikes = psg.get_times(gid - 10, population='PAGaff')
                    spikes = np.sort(spikes)
                    #print("HEllo: \n {0}".format(spikes))
                    if len(spikes) == 0:
                        continue

            # The next block of code is where we update the incoming/virtual spike trains for each cell, by adding
            # each spike to the cell's netcon (eg synapse). The only caveats is the spike-trains array must
            #  1. Have atleast one spike
            #  2. Be sorted
            #  3. first spike must occur after the delay.
            # Otherwise an error will be thrown.
                    self._spike_events[gid] = np.concatenate((self._spike_events[gid], spikes))
                    nc = self._netcons[gid]
                    for t in spikes:
                        nc.event(t)
            
#            # EUS Aff Firing Rate Update
#            psg = PoissonSpikeGenerator()
#            psg.add(
#                node_ids= [0,1,2,3,4,5,6,7,8,9],
#                firing_rate= self.pag_fr,
#                times=(next_block_tstart/1000.0 + 0.01, next_block_tstop/1000.0),
#                population= 'EUSaff',
#            )
#        
#            psg.add_spikes([0,1,2,3,4,5,6,7,8,9], [next_block_tstop, next_block_tstop, next_block_tstop, next_block_tstop, next_block_tstop, next_block_tstop, next_block_tstop,       next_block_tstop, next_block_tstop, next_block_tstop], population = "EUSaff")
#            psg.to_csv("spikes_eus.csv")
#
#            for gid, cell in sim.net.get_local_cells().items():
#                if gid < 30 and gid > 19:
#                    spikes = psg.get_times(gid - 20, population='EUSaff')
#                    spikes = np.sort(spikes)
#                    #print("HEllo: \n {0}".format(spikes))
#                    if len(spikes) == 0:
#                        continue
#
#            # The next block of code is where we update the incoming/virtual spike trains for each cell, by adding
#            # each spike to the cell's netcon (eg synapse). The only caveats is the spike-trains array must
#            #  1. Have atleast one spike
#            #  2. Be sorted
#            #  3. first spike must occur after the delay.
#            # Otherwise an error will be thrown.
#                    self._spike_events[gid] = np.concatenate((self._spike_events[gid], spikes))
#                    nc = self._netcons[gid]
#                    for t in spikes:
#                        nc.event(t)
        else:
            self.pag_fr = 0

        self._set_spike_detector(sim)
        pc.barrier()
        
    ### STEP 6: Save Calculations ####
        p_mmHg = 0.735559*p
        self._prev_glob_press = self._glob_press
        self._glob_press = p_mmHg

        #io.log_info('PGN firing rate = %.2f Hz' %fr)
        io.log_info('Volume = %.4f ml' %vol)
        io.log_info('Pressure = %.2f mmHg' %p_mmHg)
        io.log_info('Calculated bladder afferent firing rate for the next time step = {:.2f} Hz \n \n'.format(self.blad_fr))

        # Save values in appropriate lists
        self.times.append(t)
        self.b_vols.append(vol)
        self.b_pres.append(p_mmHg)

    def finalize(self, sim):
        pass

    def save_aff(self, path):
        populations = {'Bladaff':'_high_level_neurons','PAGaff':'_pag_neurons'}
        for pop_name, node_name in populations.items():
            spiketrains = SpikeTrains(population=pop_name)
            for gid in getattr(self,node_name):
                spiketrains.add_spikes(gid,self._spike_events[gid],population=pop_name)
            spiketrains.to_sonata(os.path.join(path,pop_name+'_spikes.h5'))
            spiketrains.to_csv(os.path.join(path,pop_name+'_spikes.csv'))
