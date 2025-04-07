# import psutil
# p = psutil.Process()
# p.cpu_affinity([0])

import os
# change your gpu here
# on pcdev2, 0: A100, 1: RTX3080
# on pcdev12, just check with the nvidia-smi output
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# this commend can adjust the fraction of memory JAX reallocate
#os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = 0.75

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import jax.profiler

from jimgw.jim import Jim
from jimgw.single_event.detector import ET
from jimgw.single_event.likelihood import HeterodynedTransientLikelihoodFD, TransientLikelihoodFD
from jimgw.single_event.waveform import RippleIMRPhenomD
from jimgw.prior import Uniform, PowerLaw, AlignedSpin, Composite

from ripple import ms_to_Mc_eta

from astropy.time import Time
import pandas as pd
import utils

from tap import Tap

psds_dir = "/home/twouters2/jim_gpu_benchmarks/psds/"

class InjectionRecoveryParser(Tap):
    # detector setup
    f_sampling: int = 1024
    # u can check if it is consistent with the flow and mass chosen
    # with /home/tsun-ho.pang/Projects/et_corr_jax/scripts/chirptime.py
    duration: int = 64 
    fmin: float = 5.0
    fref: float = 5.0
    ifos: list[str] = ["ET"]
    # Noise parameters, seed, psd and csd
    noise_seed: int = 13245
    psds: dict[str, str] = {"E1": psds_dir + "ET_D_psd.txt",
                            "E2": psds_dir + "ET_D_psd.txt",
                            "E3": psds_dir + "ET_D_psd.txt",}
    
    csds: dict[str, str] = {"E12": psds_dir + "ET_D_csd_zero.txt",
                            "E23": psds_dir + "ET_D_csd_zero.txt",
                            "E13": psds_dir + "ET_D_csd_zero.txt"}

    # Injection parameters
    m1: float = 39.0
    m2: float = 32.0
    s1_z: float = -0.07
    s2_z: float = -0.07
    dist_mpc: float = 410 * 1000 / 260
    tc: float = 0.
    phic: float = 0.1
    iota: float = 0.5
    psi: float = 0.7
    ra: float = 1.2
    dec: float = 0.3
    trigger_time: float = 1126259462.4
    post_trigger_duration: float = 2.0

    # Sampler parameters
    sampling_seed: int = 42
    n_chains: int = 1000
    n_loop_training: int = 2
    n_loop_production: int = 2
    n_local_steps: int = 200
    n_global_steps: int = 200
    learning_rate: float = 0.001
    max_samples: int = 50000
    momentum: float = 0.9
    num_epochs: int = 50
    batch_size: int = 50000
    use_global: bool = True
    keep_quantile: float = 0.0
    train_thinning: int = 10
    output_thinning: int = 30
    num_layers: int = 10
    hidden_size: list[int] = [128,128]
    num_bins: int = 8

    # Output parameters
    output_path: str = f"./outdir/"
    downsample_factor: int = 10


args = InjectionRecoveryParser().parse_args()

print("Injecting signals")
# setup waveform
waveform = RippleIMRPhenomD(f_ref=args.fref)
# key for noise generation and sampling
key = jax.random.PRNGKey(args.noise_seed)
# creating frequency grid
freqs = jnp.linspace(
    args.fmin,
    args.f_sampling / 2,  # maximum frequency being halved of sampling frequency
    args.duration * args.f_sampling
    )
# convert injected mass to Mc and eta
Mc, eta = ms_to_Mc_eta(jnp.array([args.m1, args.m2]))
# setup the timing setting for the injection
epoch = args.duration - args.post_trigger_duration
gmst = Time(args.trigger_time, format='gps').sidereal_time('apparent', 'greenwich').rad
# array of injection parameters
true_params = {
    'M_c':       float(Mc),      # chirp mass
    'eta':       float(eta),     # symmetric mass ratio 0 < eta <= 0.25
    's1_z':      args.s1_z,      # aligned spin of priminary component s1_z.
    's2_z':      args.s2_z,      # aligned spin of priminary component s2_z.
    'd_L':       args.dist_mpc,  # luminosity distance
    't_c':       args.tc,        # timeshift w.r.t. trigger time
    'phase_c':   args.phic,      # merging phase
    'iota':      args.iota,
    'psi':       args.psi,
    'ra':        args.ra,
    'dec':       args.dec
    }
detector_param = {
    'ra':     args.ra,
    'dec':    args.dec,
    'gmst':   gmst,
    'psi':    args.psi,
    'epoch':  epoch,
    't_c':    args.tc
    }
print(f"The injected parameters are {true_params}")
# generating the geocenter waveform
h_sky = waveform(freqs, true_params)
# setup ifo list
ifos = []
for ifo in args.ifos:
    eval(f'ifos.append({ifo})')
# inject signal into ifos
for idx, ifo in enumerate(ifos):
    key, subkey = jax.random.split(key)
    ifo.inject_signal(
        subkey,
        freqs,
        h_sky,
        detector_param,
        psd_file_dict=args.psds,
        csd_file_dict=args.csds
    )
print("Signal injected")

print("Start prior setup")
# priors without transformation 
Mc_prior    = Uniform(10, 80, naming=['M_c'])
s1z_prior   = AlignedSpin(0.99, naming=['s1_z'])
s2z_prior   = AlignedSpin(0.99, naming=['s2_z'])
dL_prior    = Uniform(0.5e3, 3e3, naming=['d_L'])
tc_prior    = Uniform(-0.1, 0.1, naming=['t_c'])
phic_prior  = Uniform(0., 2. * jnp.pi, naming=['phase_c'])
psi_prior   = Uniform(0.0, jnp.pi, naming=["psi"])
ra_prior    = Uniform(0.0, 2 * jnp.pi, naming=["ra"])
# priors with transformations
q_prior = Uniform(
    0.125,
    1,
    naming=['q'],
    transforms={
        'q': (
            'eta',
            lambda params: params['q'] / (1 + params['q']) ** 2
            )
        }
    )
cos_iota_prior = Uniform(
    0.0,
    1.0,
    naming=["cos_iota"],
    transforms={
        "cos_iota": (
            "iota",
            lambda params: jnp.arccos(
                jnp.arcsin(jnp.sin(params["cos_iota"] / 2 * jnp.pi)) * 2 / jnp.pi
            ),
        )
    },
)
sin_dec_prior = Uniform(
    -1.0,
    1.0,
    naming=["sin_dec"],
    transforms={
        "sin_dec": (
            "dec",
            lambda params: jnp.arcsin(
                jnp.arcsin(jnp.sin(params["sin_dec"] / 2 * jnp.pi)) * 2 / jnp.pi
            ),
        )
    },
)
# compose the prior
prior_list = [
        Mc_prior,
        q_prior,
        s1z_prior,
        s2z_prior,
        dL_prior,
        tc_prior,
        phic_prior,
        cos_iota_prior,
        psi_prior,
        ra_prior,
        sin_dec_prior,
]
complete_prior = Composite(prior_list)
bounds = jnp.array([[p.xmin, p.xmax] for p in complete_prior.priors])
print("Finished prior setup")

print("Replacing the CSD to zero arrays for the likelihood")
for i in range(3):
    for j in range(3):
        if i != j:
            csd = ifos[0].psd[:,i,j]
            ifos[0].psd = ifos[0].psd.at[:, i, j].set(jnp.zeros(shape=csd.shape))
        else:
            continue
print("Initializing likelihood")
# this one is with relative binning
likelihood = HeterodynedTransientLikelihoodFD(
    ifos,
    prior=complete_prior,
    bounds=bounds,
    n_bins=100,
    waveform=waveform,
    trigger_time=args.trigger_time,
    duration=args.duration,
    post_trigger_duration=args.post_trigger_duration,
    popsize=400,
    n_loops=2000,
    ref_params=true_params,
    )
# this one is without relative binning
#likelihood = TransientLikelihoodFD(
#    ifos,
#    waveform=waveform,
#    trigger_time=args.trigger_time,
#    duration=args.duration,
#    post_trigger_duration=args.post_trigger_duration,
#    )

mass_matrix = jnp.eye(len(prior_list))
for idx, prior in enumerate(prior_list):
    if prior.naming[0] in ['t_c']:
        print(f'Modified the mass matrix for {prior.naming}')
        mass_matrix = mass_matrix.at[idx, idx].set(1e-3)
#    elif prior.naming[0] in ['ra', 'sin_dec']:
#        print(f'Modified the mass matrix for {prior.naming}')
#        mass_matrix = mass_matrix.at[idx, idx].set(1e3)
local_sampler_arg = {'step_size': mass_matrix * 3e-3}

import optax
total_epochs = args.num_epochs * args.n_loop_training
start = int(total_epochs / 10)
start_lr = 1e-3
end_lr = 1e-5
power = 4.0
schedule_fn = optax.polynomial_schedule(start_lr, end_lr, power, total_epochs-start, transition_begin=start)

jim = Jim(
    likelihood, 
    complete_prior,
    n_loop_training=args.n_loop_training,
    n_loop_production = args.n_loop_production,
    n_local_steps=args.n_local_steps,
    n_global_steps=args.n_global_steps,
    n_chains=args.n_chains,
    n_epochs=args.num_epochs,
    learning_rate=schedule_fn,
    max_samples = args.max_samples,
    momentum = args.momentum,
    batch_size = args.batch_size,
    use_global=args.use_global,
    keep_quantile= args.keep_quantile,
    train_thinning = args.train_thinning,
    output_thinning = args.output_thinning,
    local_sampler_arg = local_sampler_arg,
    seed = args.sampling_seed,
    num_layers = args.num_layers,
    hidden_size = args.hidden_size,
    num_bins = args.num_bins
)

print("Start sampling")
key = jax.random.PRNGKey(args.sampling_seed)
jim.sample(key)
jim.print_summary()
samples = jim.get_samples()

# make output directory if not exists

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

print("Saving the samples now:")

for key in samples.keys():
    samples[key] = jnp.ravel(samples[key])
df = pd.DataFrame.from_dict(samples)
df.to_csv(f'./{args.output_path}/posterior_samples.dat', sep=' ', index=False)

# make corner plot

utils.corner_plot(samples, true_params, args.output_path)
