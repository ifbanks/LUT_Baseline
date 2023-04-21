import sys, os
from bmtk.simulator import bionet
from bmtk.utils.reports.spike_trains import SpikeTrains
from feedback_loop import FeedbackLoop
import numpy as np
import pandas as pd
from plotting import plot_figure, plotting_calculator
from bmtk.analyzer.compartment import plot_traces

num = {
'Bladaff' : 10,
'PAGaff'  : 10,
'EUSaff'  : 10,
'IND'     : 10,
'Hypo'    : 10,
'INmplus' : 10,
'INmminus': 10,
'PGN'     : 10,
'FB'      : 10,
'IMG'     : 10,
'MPG'     : 10,
'EUSmn'   : 10,
'Bladmn'  : 10
}
gids = {}
ind = 0
for pop,n in num.items():
    gids[pop] = ind
    ind += n

def run(config_file=None,sim=None,conf=None):
    if config_file is not None:
        conf = bionet.Config.from_json(config_file, validate=True)
        dt = conf['run']['dt']
        n_steps = np.ceil(conf['run']['tstop']/dt+1).astype(np.int)
        fbmod = None
    if sim is not None:
        n_steps = sim.n_steps
        dt = sim.dt
        fbmod = sim._sim_mods[[isinstance(mod,FeedbackLoop) for mod in sim._sim_mods].index(True)]
    output_dir = conf.output_dir
    print(n_steps,dt)

    spikes_df = pd.read_csv(os.path.join(output_dir,'output/spikes.csv'), sep=' ')
    print(spikes_df['node_ids'].unique())
    spike_trains = SpikeTrains.from_sonata(os.path.join(output_dir,'output/spikes.h5'))

    #plotting
    window_size = 1000
    pops = ['Bladaff','PGN','PAGaff','EUSmn','INmminus','INmplus','IND', 'IMG', 'FB']
    windows = [window_size]*len(pops)
    means = {}
    stdevs = {}
    for pop,win in zip(pops,windows):
        means[pop], stdevs[pop] = plotting_calculator(spike_trains, n_steps, dt, win, gids, num, pop)
    
    plot_figure(means, stdevs, n_steps, dt, tstep=window_size, fbmod=fbmod)


if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        run(config_file=sys.argv[-1])
    else:
        run(config_file='jsons/simulation_config.json')

#    plot_traces(config_file = 'jsons/simulation_config.json', report_name = 'membrane_report', node_ids = [5], title = 'Bladder Afferent Membrane Voltage', show_legend = False)
#    plot_traces(config_file = 'jsons/simulation_config.json', report_name = 'membrane_report', node_ids = [15], title = 'PAG Membrane Voltage', show_legend = False)
#    plot_traces(config_file = 'jsons/simulation_config.json', report_name = 'membrane_report', node_ids = [25], title = 'EUS Membrane Voltage', show_legend = False)
#    plot_traces(config_file = 'jsons/simulation_config.json', report_name = 'membrane_report', node_ids = [35], title = 'IND Afferent Membrane Voltage', show_legend = False)
#    plot_traces(config_file = 'jsons/simulation_config.json', report_name = 'membrane_report', node_ids = [55], title = 'INM+ Afferent Membrane Voltage', show_legend = False)
#    plot_traces(config_file = 'jsons/simulation_config.json', report_name = 'membrane_report', node_ids = [65], title = 'INM- Afferent Membrane Voltage', show_legend = False)
#    plot_traces(config_file = 'jsons/simulation_config.json', report_name = 'membrane_report', node_ids = [75], title = 'PGN Membrane Voltage', show_legend = False)
#    plot_traces(config_file = 'jsons/simulation_config.json', report_name = 'membrane_report', node_ids = [85], title = 'FB Membrane Voltage', show_legend = False)
#    plot_traces(config_file = 'jsons/simulation_config.json', report_name = 'membrane_report', node_ids = [105], title = 'MPG Membrane Voltage', show_legend = False)
