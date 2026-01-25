import numpy as np
import matplotlib.pyplot as plt
import json
from netpyne import sim
from t42_cfg import cfg
from t42_netParams import netParams

# ==========================================
# PHASE 1: Run the Biophysical Simulation
# ==========================================
print(">>> initializing CA1 Microcircuit (Ponzi et al., 2023)...")

# We use the configuration from the uploaded t42 files
# You might want to reduce duration for testing (e.g., to 1000ms)
cfg.duration = 1000  
cfg.verbose = False

# Create and Run the simulation
sim.createSimulateAnalyze(netParams=netParams, simConfig=cfg)

# ==========================================
# PHASE 2: The "Bridge" (Spikes -> Evidence)
# ==========================================
print(">>> Extracting Spikes for Decision Making...")

# 1. Retrieve data directly from memory (no need to parse JSON file)
all_spikes_time = np.array(sim.allSimData['spkt'])
all_spikes_gid = np.array(sim.allSimData['spkid'])

# 2. Define our "Place Cells"
# The PYR population usually starts at GID 0.
# We verify the GID range for PYR cells from the netParams
pyr_pop_name = 'PYR_pop'
pyr_gids = sim.net.pops[pyr_pop_name].cellGids
midpoint = len(pyr_gids) // 2

# Split PYR cells into two pools
left_gids = set(pyr_gids[:midpoint])   # Pool A: Evidence for Left
right_gids = set(pyr_gids[midpoint:])  # Pool B: Evidence for Right

# 3. Bin spikes to calculate instantaneous firing rate
dt_decision = 5.0  # ms (Time step for the DDM, simpler than simulation dt)
time_bins = np.arange(0, cfg.duration, dt_decision)
rate_left = []
rate_right = []

for t in time_bins:
    t_end = t + dt_decision
    # Find spikes in this time window
    mask = (all_spikes_time >= t) & (all_spikes_time < t_end)
    window_gids = all_spikes_gid[mask]
    
    # Count spikes for Left vs Right pools
    # We normalize by pool size to get Hz (spikes / N / seconds)
    count_L = sum(1 for gid in window_gids if gid in left_gids)
    count_R = sum(1 for gid in window_gids if gid in right_gids)
    
    hz_L = (count_L / len(left_gids)) * (1000/dt_decision)
    hz_R = (count_R / len(right_gids)) * (1000/dt_decision)
    
    rate_left.append(hz_L)
    rate_right.append(hz_R)

rate_left = np.array(rate_left)
rate_right = np.array(rate_right)

# ==========================================
# PHASE 3: The Drift Diffusion Model (DDM)
# ==========================================
print(">>> Running Cognitive DDM Integration...")

# DDM Parameters
decision_threshold = 10.0   # Arbitrary units
x = 0.0                     # Decision Variable (starts at 0)
trajectory = [0.0]
decision_time = None
choice = None

# Scaling factor: How strong is the neural evidence?
alpha = 0.10 
# Noise in the decision accumulator (separate from neural noise)
sigma_ddm = 0.0 

for i, t in enumerate(time_bins):
    # --- THE CORE MECHANISM ---
    # The Drift Rate (v) is NOT constant. It fluctuates with the 
    # Theta-Gamma rhythm of the biophysical model.
    # Agregamos un 'bias' que representa el estímulo sensorial real
    # Si bias > 0, favorece Izquierda. Si bias < 0, favorece Derecha.
    stimulus_bias = 5.0  # Hz extra para el lado izquierdo

    # v(t) = Neural_Left(t) - Neural_Right(t)
    drift_t = alpha * ((rate_left[i] + stimulus_bias) - rate_right[i])
    
    # Update DDM: dX = v*dt + sigma*dW
    noise = np.random.normal(0, sigma_ddm)
    dx = drift_t + noise
    x += dx
    trajectory.append(x)
    
    # Check crossing
    if x >= decision_threshold:
        choice = 'LEFT'
        decision_time = t
        break
    elif x <= -decision_threshold:
        choice = 'RIGHT'
        decision_time = t
        break

# ==========================================
# PHASE 4: Visualization (The "Money Plot")
# ==========================================
print(f">>> Outcome: Chosen {choice} at {decision_time} ms")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot 1: Neural Evidence (The "Rhythm")
ax1.plot(time_bins, rate_left, color='blue', alpha=0.6, label='Left Pool (PYR)')
ax1.plot(time_bins, rate_right, color='red', alpha=0.6, label='Right Pool (PYR)')
ax1.set_ylabel('Firing Rate (Hz)')
ax1.set_title('Biophysical Input (Theta-Gamma Oscillations)')
ax1.legend()

# Plot 2: Decision Variable (The "Integration")
time_axis = np.arange(0, len(trajectory)) * dt_decision
ax2.plot(time_axis, trajectory, color='black', linewidth=2)
ax2.axhline(decision_threshold, color='green', linestyle='--', label='Threshold')
ax2.axhline(-decision_threshold, color='green', linestyle='--')
ax2.set_ylabel('Decision Variable (X)')
ax2.set_xlabel('Time (ms)')
ax2.set_title('Accumulated Evidence (DDM)')

plt.tight_layout()
plt.savefig('lascon_project_results.png')
plt.show()