import math
import random
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from utils import grad_to_vector
from dataset import TrainDataset


class StegaPoison:
    """
    StegaPoison - Algorithm 1 Implementation (FIXED VERSION)

    Sequential Pipeline:
    1. Local Training: Train on D_i from CURRENT model to get Θ^(r), compute Δ^(r) = Θ^(r) - Θ^(current)
    2. Watermarking: Δ^(wm) ← Δ^(r) + w, where w_j = γ * sin(2πωj)
    3. LVDEP: Identify low-variance dims L = {j | Var_j(Δ^(r)) < τ_lvdep}, inject noise
    4. Mirror Shift: Δ^(mis) ← Δ^(lvdep) + βd = (1 + β) * Δ^(lvdep)
    5. Velocity-Based Sampling: v ← Δ^(mis)_momentum + Δ_i^(r), conditional noise if ||v|| > τ_vel
    6. Statistical Invisibility: Scale if ||Δ^(vel)|| > η||Θ^(r)|| * stealth_factor
    """

    def __init__(self, attacker_user_data, args, device):
        self.attacker_user_data = attacker_user_data
        self.attacker_id_list = list(self.attacker_user_data.keys())
        self.args = args
        self.device = device
        self.seed_rng = np.random.default_rng(self.args.SEED)

        # === Step 2: Watermarking Pattern ===
        gamma = getattr(args, 'WATERMARK_SCALE', 0.05)
        omega = getattr(args, 'K', 15.0)

        j = torch.arange(1, self.args.EMBDIM + 1, device=self.device, dtype=torch.float32)
        self.watermark_pattern = gamma * torch.sin(2.0 * math.pi * omega * j)

        # === Step 3: LVDEP threshold ===
        self.tau_lvdep = getattr(args, 'VADP_THRESHOLD', 0.5)
        self.epsilon_scale = getattr(args, 'VADP_SCALE', 1.0)

        # === Step 4: Mirror factor β ===
        self.beta = getattr(args, 'ALPHA', 1.0)

        # === Step 5: Velocity threshold τ_vel ===
        self.tau_vel = getattr(args, 'VELOCITY_THRESHOLD', 0.5)
        self.eta_noise = getattr(args, 'NOISE_VARIANCE', 0.35)

        # === Step 6: Statistical invisibility η ===
        self.eta_stat = getattr(args, 'INVISIBILITY_FACTOR', 0.3)

        # FIX: Store per-client velocity for momentum accumulation
        self.client_velocity = {}

        # FIX: Reduced base scale from 3.0 to 1.0
        self.base_scale = getattr(args, 'SCALE', 1.0)

    def prepare(self, server_model, step, *args):
        """
        Prepare attack: Execute Steps 1-4 to compute base malicious direction

        CRITICAL FIX: Use CURRENT server model as baseline, not initial model.
        This ensures attack remains relevant as the model evolves.
        """
        # === Step 1: Local Training and Δ^(r) Computation ===
        # FIX: Use CURRENT server model as baseline (updated every round)
        current_item_params = server_model.item_model.item_embedding.weight.clone().detach()

        sample_attacker = random.sample(
            self.attacker_id_list,
            k=min(self.args.ATTACKER_SAMPLE_NUM, len(self.attacker_id_list))
        )

        # Collect Δ^(r) from each sampled attacker
        delta_r_list = []

        for uid in sample_attacker:
            # FIX: Clone CURRENT model for local training (not initial model!)
            local_item_params = current_item_params.clone()
            server_model.item_model.item_embedding.weight.data.copy_(local_item_params)

            # Local training
            optimizer = optim.Adam(server_model.parameters(), lr=self.args.LR)
            client_data = self.attacker_user_data[uid]
            client_dataset = TrainDataset(uid, client_data)
            client_dataloader = DataLoader(
                client_dataset, shuffle=True, batch_size=self.args.BATCH_SIZE
            )

            for uid_batch, iid in client_dataloader:
                uid_batch = uid_batch.to(self.device, non_blocking=True)
                iid = iid.to(self.device, non_blocking=True)

                _, bz_loss = server_model(uid_batch, iid)
                optimizer.zero_grad()
                bz_loss.backward()
                optimizer.step()

            # Get trained parameters Θ^(r)
            theta_r = server_model.item_model.item_embedding.weight.clone().detach()

            # FIX: Compute Δ^(r) = Θ^(r) - Θ^(current), NOT Θ^(r) - Θ^(0)
            delta_r = theta_r - current_item_params
            delta_r_list.append(delta_r)  # Shape: (NUM_ITEMS, EMBDIM)

        with torch.no_grad():
            # Stack: (num_sampled_attackers, NUM_ITEMS, EMBDIM)
            delta_r_stack = torch.stack(delta_r_list, dim=0)

            # === Step 2: Watermarking Injection ===
            delta_mean = delta_r_stack.mean(dim=0)  # (NUM_ITEMS, EMBDIM)

            # Add watermark: Δ^(wm) = Δ^(mean) + w
            delta_wm = delta_mean + self.watermark_pattern.unsqueeze(0)

            # === Step 3: LVDEP (Low-Variance Dimension Embedding Perturbation) ===
            delta_var = delta_r_stack.var(dim=0)  # (NUM_ITEMS, EMBDIM)

            # Identify low-variance dimensions
            low_var_mask = (delta_var < self.tau_lvdep).float()

            # Generate Gaussian noise
            epsilon = torch.randn_like(delta_wm) * self.epsilon_scale

            # Apply LVDEP: Add noise to low-variance dimensions
            delta_lvdep = delta_wm + epsilon * low_var_mask

            # === Step 4: Mirror Shift ===
            delta_mis = (1.0 + self.beta) * delta_lvdep

            # Store for per-client refinement in update()
            self.base_delta_mis = delta_mis  # (NUM_ITEMS, EMBDIM)

            # FIX: Reset model to CURRENT state (not initial state)
            server_model.item_model.item_embedding.weight.data.copy_(current_item_params)

        return torch.tensor(0.0), None, 0

    def update(self, server_model, client_id):
        """
        Generate poisoned update for specific attacker client

        Implements Steps 5-6 per client:
        - Step 5: Velocity-Based Sampling with Momentum
        - Step 6: Statistical Invisibility
        """
        # First compute benign gradient for this client
        optimizer = optim.Adam(server_model.parameters(), lr=self.args.LR)
        client_data = self.attacker_user_data[client_id]
        client_dataset = TrainDataset(client_id, client_data)
        client_dataloader = DataLoader(
            client_dataset, shuffle=True, batch_size=self.args.BATCH_SIZE
        )
        client_sample_num = len(client_dataset)
        client_gradient_vec = 0

        for uid, iid in client_dataloader:
            uid = uid.to(self.device, non_blocking=True)
            iid = iid.to(self.device, non_blocking=True)

            _, bz_loss = server_model(uid, iid)
            optimizer.zero_grad()
            bz_loss.backward()

            batch_sample_num = len(uid)
            client_gradient_vec += grad_to_vector(server_model) * (
                batch_sample_num / client_sample_num
            )

        user_grad_param = self.args.NUM_USERS * self.args.EMBDIM
        item_grad_param = self.args.NUM_ITEMS * self.args.EMBDIM

        # Extract gradient components
        client_user_grad = client_gradient_vec[:user_grad_param].reshape(
            self.args.NUM_USERS, self.args.EMBDIM
        )
        user_grad_mask = torch.zeros(self.args.NUM_USERS, dtype=torch.float32).to(self.device)
        user_grad_mask[client_id] = 1.0
        client_user_grad = (client_user_grad * user_grad_mask.reshape(-1, 1)).reshape(-1)

        # Benign item gradient
        benign_item_grad = client_gradient_vec[
            user_grad_param:user_grad_param + item_grad_param
        ].reshape(self.args.NUM_ITEMS, self.args.EMBDIM)

        # Convert gradient to parameter update
        delta_i_r = -benign_item_grad

        client_other_grad = client_gradient_vec[user_grad_param + item_grad_param:]

        with torch.no_grad():
            # === Step 5: Velocity-Based Sampling with Momentum ===
            # FIX: Initialize and accumulate client velocity
            if client_id not in self.client_velocity:
                self.client_velocity[client_id] = torch.zeros_like(self.base_delta_mis)

            # FIX: Apply momentum accumulation
            momentum = getattr(self.args, 'MOMENTUM', 0.8)
            self.client_velocity[client_id] = (
                momentum * self.client_velocity[client_id] +
                (1 - momentum) * self.base_delta_mis
            )

            # Compute velocity: v ← momentum_velocity + Δ_i^(r)
            velocity = self.client_velocity[client_id] + delta_i_r

            # Compute velocity norm
            v_norm = torch.norm(velocity)

            # Threshold-based noise injection
            if v_norm > self.tau_vel:
                # Add Gaussian noise
                noise = torch.randn_like(self.base_delta_mis) * self.eta_noise
                delta_vel = self.client_velocity[client_id] + noise
            else:
                # Keep momentum velocity
                delta_vel = self.client_velocity[client_id]

            # === Step 6: Statistical Invisibility (Improved) ===
            # FIX: Better stealth calibration
            delta_vel_norm = torch.norm(delta_vel)

            # Get model norm as reference scale
            theta_r_norm = torch.norm(server_model.item_model.item_embedding.weight)

            # FIX: Use stealth factor to modulate invisibility
            stealth_factor = getattr(self.args, 'STEALTH_FACTOR', 1.5)
            max_allowed_norm = self.eta_stat * theta_r_norm * stealth_factor

            # Conditional scaling
            if delta_vel_norm > max_allowed_norm:
                scale_factor = max_allowed_norm / (delta_vel_norm + 1e-10)
                delta_scaled = scale_factor * delta_vel
            else:
                delta_scaled = delta_vel

            # Convert from parameter update to gradient
            # FIX: Reduced base_scale from 3.0 to 1.0
            attacker_item_grad = -delta_scaled.reshape(-1) * self.base_scale

        return client_user_grad, attacker_item_grad, client_other_grad, client_sample_num
