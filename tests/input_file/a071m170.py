import gvar as gv
import numpy as np
import os
# import fitter.fastfit as ffit

file_params = {}
file_params['data_dir'] = '/home/gbradley/c51_corr_analysis/tests/data/E7/' #all configurations
# file_params['out_dir'] = 'path/to/concateddsets/'

fit_states = ['pion','proton','gA','gV']
bs_seed = 'a071m170'

def ensemble(params):
    cfg     = params['ENS_BASE']+'-'+params['STREAM']+'.ama'

params = dict()
params['run_ff'] = True

params['cfg_i'] = 600  # initial config number
params['cfg_f'] = 3396 # final config number
params['cfg_d'] = 4    # config step value 

def parse_cfg_arg(cfg_arg,params):
    allowed_cfgs = range(params['cfg_i'],params['cfg_f']+1,params['cfg_d'])
    if not cfg_arg:
        ci = params['cfg_i']
        cf = params['cfg_f']
        dc = params['cfg_d']
    else:
        if len(cfg_arg) == 3:
            cfgs = range(cfg_arg[0], cfg_arg[1]+1, cfg_arg[2])
        elif len(cfg_arg) == 1:
            cfgs = range(cfg_arg[0], cfg_arg[0]+1, 1)
        if not all([cfg in allowed_cfgs for cfg in cfgs]):
            print('you selected configs not allowed for %s:' %params['ENS_S'])
            print('       allowed cfgs = range(%d, %d, %d)' %(params['cfg_i'], params['cfg_f'], params['cfg_d']))
            sys.exit('  your choice: cfgs = range(%d, %d, %d)' %(cfgs[0],cfgs[-1],cfgs[1]-cfgs[0]))
        elif len(cfg_arg) == 1:
            ci = int(cfg_arg[0])
            cf = int(cfg_arg[0])
            dc = 1
        elif len(cfg_arg) == 3:
            ci = int(cfg_arg[0])
            cf = int(cfg_arg[1])
            dc = int(cfg_arg[2])
        else:
            print('unrecognized use of cfg arg')
            print('cfg_i [cfg_f cfg_d]')
            sys.exit()
    return range(ci,cf+1,dc)

params['seed'] = dict()
params['seed']['0'] = '0'
params['seed']['a'] = 'a'
params['seed']['b'] = 'b'
params['seed']['c'] = 'c'

# isotropic clover ens info 
params['ENS_ABBR'] =  'a071m170'
params['ENS']      =  'E7'
params['NL']        = '72'
params['NT']        = '192'


params['t_seps']  = [13,15,17,19,21]
params['flavs']   = ['U','D']
params['snk_mom'] = ['0 0 0']
params['SS_PS']   = 'SS'
params['particles'] = ['proton','proton_SP','pion','pion_SP','NUCL']
params['curr_4d'] = ['A3','V4','A1','A2','A4','V1','V2','V3','P','S']
params['curr_0p'] = ['A3','V4','A1','A2','A4','V1','V2','V3','S','T34','T12','T13','T14','T23','T24']


corr_lst = {
    'proton':{
        'dsets':[
            '2pt/proton/src10.0_snk10.0/proton/AMA',
            '2pt/proton_SP/src10.0_snk10.0/proton/AMA'],
        'weights'  :[1],
        't_reverse':[False],
        'fold'     :True,
        'snks'     :['S','P'],
        'srcs'     :['S'],
        'xlim'     :[0,48.5],
        'ylim'     :[0.12,0.169],
        'colors'   :{'SS':'#70bf41','PS':'k'},
        'type'     :'exp',
        'ztype'    :'z_snk z_src',
        'z_ylim'   :[0.055,0.26],
        # fit params
        'n_state'  :3,
        'tsep'     : 0,
        'T'        :96,
        't_range'  :np.arange(5,48),
        't_sweep'  :range(2,28),
        'n_sweep'  :range(1,6),
        'eff_ylim' :[0.133,0.1349]
    },
}

priors = gv.BufferDict()
x = dict()

priors['pion_E_0']  = gv.gvar(0.14, .006)
priors['pion_zS_0'] = gv.gvar(5e-3, 5e-4)
priors['pion_zP_0'] = gv.gvar(0.125,  0.015)

priors['proton_E_0']  = gv.gvar(0.35, .025)
priors['proton_zS_0'] = gv.gvar(2.0e-5, 1.e-5)
priors['proton_zP_0'] = gv.gvar(2.5e-3, 1.e-3)

for corr in corr_lst:#[k for k in corr_lst if 'mres' not in k]:
    for n in range(1,10):
        # use 2 mpi splitting for each dE

        # E_n = E_0 + dE_10 + dE_21 +...
        # use log prior to force ordering of dE_n
        priors['log(%s_dE_%d)' %(corr,n)] = gv.gvar(np.log(2*priors['pion_E_0'].mean), 0.7)

        # for z_P, no suppression with n, but for S, smaller overlaps
        priors['%s_zP_%d' %(corr,n)] = gv.gvar(priors['%s_zP_0' %(corr)].mean, 2*priors['%s_zP_0' %(corr)].sdev)
        zS_0 = priors['%s_zS_0' %(corr)]
        if n <= 2:
            priors['%s_zS_%d' %(corr,n)] = gv.gvar(zS_0.mean, 2*zS_0.sdev)
        else:
            priors['%s_zS_%d' %(corr,n)] = gv.gvar(zS_0.mean/2, zS_0.sdev)

    for snk in corr_lst[corr]['snks']:
        sp = snk+corr_lst[corr]['srcs'][0]
        state = corr+'_'+sp
        x[state] = dict()
        x[state]['state'] = corr
        for k in ['type', 'T', 'n_state', 't_range', 'eff_ylim', 'ztype']:
            if k in corr_lst[corr]:
                x[state][k] = corr_lst[corr][k]
        if 't0' in corr_lst[corr]:
            x[state]['t0'] = corr_lst[corr]['t0']
        else:
            x[state]['t0'] = 0
        if 'mres' not in corr:
            x[state]['color'] = corr_lst[corr]['colors'][sp]
            x[state]['snk']   = snk
            x[state]['src']   = corr_lst[corr]['srcs'][0]
        else:
            x[state]['color'] = corr_lst[corr]['colors']
# def ens_base():
#     ens,stream = os.getcwd().split('/')[-1].split('_')
#     return ens,stream 
# print(ens_base())





