import gvar as gv
import numpy as np
import os

fit_states = ['proton','pion','A3','V4']

params = dict()
params['ENS_ABBR'] = '5682'
params['snk_mom'] = ['+0 +0 +0']
snk_mom = params['snk_mom'][0]
m0,m1,m2 = snk_mom.split()
params['M0']=m0
params['M1']=m1
params['M2']=m2
# params['zero_mom'] = 'proton'
params['momentum'] = 'qz%s_qy%s_qx%s' %(m0,m1,m2)
params['t_seps']  = [8,10,12,14]
params['flavs']   = 'U'
params['gamma'] = 'g8'
# params['spins']   = ['up_up','dn_dn']
# params['SS_PS']   = 'SS'
# params['corr'] = ['proton']
# params['curr_4d'] = ['A3','V4','A1','A2','A4','V1','V2','V3','P','S']
# #params['curr_p']  = ['A3','V4','A1','A2','A4','V1','V2','V3','P','S']
# params['curr_0p'] = ['A3','V4','A1','A2','A4','V1','V2','V3','S','T34','T12','CHROMO_MAG']

directory = './data/C13/'
N_cnf = len([name for name in os.listdir(directory) if os.path.isfile(name)])
dirs = os.listdir(directory)
cnf_abbr = [files.split(".ama.h5",1)[0] for files in dirs]
datatag = []
for abbr in cnf_abbr:
    datatag.append(abbr.replace("-","."))
# get data file from given abbr in params     
data_file = []
for dirpath,_,filenames in os.walk(directory):
    for f in filenames:
        # data_file_list.append(os.path.abspath(os.path.join(dirpath, f)))
        if params['ENS_ABBR'] in f:
            data_file.append(os.path.abspath(os.path.join(dirpath, f)))
            # print(data_file[0]) 
h5_group_2pt = []
h5_group_3pt = []
for states in fit_states:
    if states in ['proton','pion']:
        h5_group_2pt.append('/2pt/'+states+'/src5.0_snk5.0/'+states+'/C13.b_'+params['ENS_ABBR']+'/')
h5_group_3pt.append('/3pt_tsep'+str(params['t_seps'][0])+'/NUCL_'+params['flavs']+'_MIXED_NONREL_l0_'+params['gamma']+'/src5.0_snk5.0/'+params['momentum']+'/C13.b_'+params['ENS_ABBR']+'/')


corr_lst = {
    # PION
    'pion':{
        'corr_array':False,
        'stack'      : False,
        'q_bilinear' : False,
        'vector':False,
        'axial': False,
        'dsets':['2pt/pion/src5.0_snk5.0/pion/C13.b_5682/AMA',
                '2pt/pion_SP/src5.0_snk5.0/pion/C13.b_5682/AMA'],
        'weights'  :[1],
        't_reverse':[False],
        'fold'     :True,
        'snks'     :['S', 'P'],
        'srcs'     :['S'],
        'xlim'     :[0,48.5],
        'ylim'     :[0.12,0.169],
        'colors'   :{'SS':'#70bf41','PS':'k'},
        'type'     :'cosh',
        'ztype'    :'z_snk z_src',
        'z_ylim'   :[0.055,0.26],
        # fit params
        'n_state'  :3,
        'T'        :96,
        't_range'  :np.arange(5,48),
        't_sweep'  :range(2,28),
        'n_sweep'  :range(1,6),
        'eff_ylim' :[0.133,0.1349]
    },
   
    # PROTON
    'proton':{
        'corr_array': False,
        'stack'      : True,
        'axial': False,
        'vector':False,
        'q_bilinear' : False,
        'dsets':['2pt/proton/src5.0_snk5.0/proton/C13.b_5682/AMA',
                '2pt/proton_SP/src5.0_snk5.0/proton/C13.b_5682/AMA'],
        'weights'  :[1.],
        't_reverse':[False],
        'phase'    :[1],
        'fold'     :False,
        'snks'     :['S', 'P'],
        'srcs'     :['S'],
        'xlim'     :[0,25.5],
        'ylim'     :[0.425,0.575],
        'colors'   :{'SS':'#70bf41','PS':'k'},
        'type'     :'exp',
        'ztype'    :'z_snk z_src',
        'z_ylim'   :[0.,0.0039],
        # fit params
        'n_state'  :3,
        't_range'  :np.arange(5,17),
        't_sweep'  :range(2,16),
        'n_sweep'  :range(1,6),
    },
    # nucleon 3pt axial (gamma_1)
    'A3':{
        'corr_array':False,
        'stack': False,
        'axial': True,
        'vector':False,
        'dsets':['3pt_tsep8/NUCL_D_MIXED_NONREL_l0_g1/src5.0_snk5.0/qz+0_qy+0_qx+0/C13.b_5682/AMA',
                '3pt_tsep8/NUCL_U_MIXED_NONREL_l0_g1/src5.0_snk5.0/qz+0_qy+0_qx+0/C13.b_5682/AMA'],
        'weights'  :[1.],
        't_reverse':[False],
        'phase'    :[1],
        'fold'     :False,
        'snks'     :['S', 'P'],
        'srcs'     :['S'],
        'xlim'     :[0,25.5],
        'ylim'     :[0.425,0.575],
        'colors'   :{'SS':'#70bf41','PS':'k'},
        'type'     :'exp',
        'ztype'    :'z_snk z_src',
        'z_ylim'   :[0.,0.0039],
        # fit params
        'n_state'  :3,
        't_range'  :np.arange(5,17),
        't_sweep'  :range(2,16),
        'n_sweep'  :range(1,6),
    },
    #nucleon 3pt vector charge (gamma_8)
    'V4':{
        'corr_array':False,
        'stack':False,
        'q_bilinear' : True,
        'axial': False,
        'vector':True,
        'stack': False,
        'dsets':['3pt_tsep8/NUCL_D_MIXED_NONREL_l0_g8/src5.0_snk5.0/qz+0_qy+0_qx+0/C13.b_5682/AMA',
                '3pt_tsep8/NUCL_U_MIXED_NONREL_l0_g8/src5.0_snk5.0/qz+0_qy+0_qx+0/C13.b_5682/AMA'],
        'weights'  :[1.],
        't_reverse':[False],
        'phase'    :[1],
        'fold'     :False,
        'snks'     :['S', 'P'],
        'srcs'     :['S'],
        'xlim'     :[0,25.5],
        'ylim'     :[0.425,0.575],
        'colors'   :{'SS':'#70bf41','PS':'k'},
        'type'     :'exp',
        'ztype'    :'z_snk z_src',
        'z_ylim'   :[0.,0.0039],
        # fit params
        'n_state'  :3,
        't_range'  :np.arange(5,17),
        't_sweep'  :range(2,16),
        'n_sweep'  :range(1,6),
    },
}

priors = gv.BufferDict()
x      = dict()

priors['proton_E_0']  = gv.gvar(0.5, .06)
priors['proton_zS_0'] = gv.gvar(2.0e-5, 1.e-5)
priors['proton_zP_0'] = gv.gvar(2.5e-3, 1.e-3)

priors['pion_E_0']  = gv.gvar(0.14, .006)
priors['pion_zS_0'] = gv.gvar(5e-3, 5e-4)
priors['pion_zP_0'] = gv.gvar(0.125,  0.015)




for corr in corr_lst:#[k for k in corr_lst if 'mres' not in k]:
    if corr not in ['mres','A3','V4']:
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
    elif corr in ['A3','V4']:

        priors['A3_00'] = gv.gvar(1.2, 0.2)
        priors['V4_00'] = gv.gvar(1.0, 0.2)
        pt3_nstates = corr_lst[corr]['n_state']
        sum_nstates = 5 
        form_fac_nstates = max(np.array([pt3_nstates, sum_nstates]))

        for i in range(form_fac_nstates):
            for j in range(form_fac_nstates):
                if i+j >= 1:
                    if j < i:  
                        priors['A3_'+str(j)+str(i)] = gv.gvar(0, 1)
                        priors['V4_'+str(j)+str(i)] = gv.gvar(0, 1)

                    elif j == i:
                        priors['A3_'+str(j)+str(i)] = gv.gvar(0, 1)
                        priors['V4_'+str(j)+str(i)] = gv.gvar(1, 0.2)

        for i in range(form_fac_nstates-1):
            priors['sum_A3_'+str(i)] = gv.gvar(0, 1)
            priors['sum_V4_'+str(i)] = gv.gvar(0, 1)

        priors['sum_A3_'+str(sum_nstates-1)] = gv.gvar(0, 1)
        priors['sum_V4_'+str(sum_nstates-1)] = gv.gvar(1, 0.2)

    # x-params
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



# params['t_seps']  = [8,10,12,14]
# params['flavs']   = ['U','D']
# params['spins']   = ['up_up','dn_dn']
# params['snk_mom'] = ['0 0 0']
# params['SS_PS']   = 'SS'
# params['particles'] = ['proton','proton_SP']
# params['curr_4d'] = ['A3','V4','A1','A2','A4','V1','V2','V3','P','S']
# #params['curr_p']  = ['A3','V4','A1','A2','A4','V1','V2','V3','P','S']
# params['curr_0p'] = ['A3','V4','A1','A2','A4','V1','V2','V3','S','T34','T12','CHROMO_MAG']

# snk_mom = params['snk_mom'][0]
# params['M0']=m0
# params['M1']=m1
# params['M2']=m2
# params['MOM'] = 'px%spy%spz%s' %(m0,m1,m2)

# state = dict()
# state['pp'] = 'proton'
# state['np'] = 'proton_np'
# spins = ['up_up','dn_dn']

# for s0 in srcs:
#     t0 = s0.split('t')[1]
#     params['SRC'] = s0
#     if verbose:
#         print(charge,tsep,no,src)
#     coherent_formfac_name = coherent_ff_base % params
#     coherent_formfac_file  = base_dir+'/formfac/'+no + '/'+coherent_formfac_name+'.h5'
#     src_h5 = h5.open_file(coherent_formfac_file,'r')
#     for par in ['pp','np']:
#         if par == 'np':
#             tsep_s = '-'+tsep
#         else:
#             tsep_s = tsep
#         for spin in spins:
#             h5_path  = '/'+state[par]+'_%(FS)s_t0_'+t0+'_tsep_'+tsep_s+'_sink_mom_px0_py0_pz0/'
#             h5_path += charge+'/'+sources.src_split(s0)+'/px0_py0_pz0/local_current'
#             try:
#                 FS   = 'UU_'+spin
#                 tmp  = src_h5.get_node(h5_path % {'FS':FS}).read()
#                 FS   = 'DD_'+spin
#                 tmp -= src_h5.get_node(h5_path % {'FS':FS}).read()
#                 data_tmp[spin+'_'+par].append(tmp)
#             except Exception as e:
#                 print('bad data read')
#                 print(e)
#     src_h5.close()
#     for par in ['pp','np']:
#     for spin in spins:
#         data[spin+'_'+par].append(np.mean(np.array(data_tmp[spin+'_'+par]),axis=0))
