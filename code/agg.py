import math
import os
import torch
import pickle
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from sklearn.cluster import KMeans
from utils import vector_to_grad
from attacker import kmeans


class FedAdam():
    def __init__(self, server_model, args, device):
        self.server_model = server_model
        self.server_optimizer = optim.Adam(self.server_model.parameters(), lr=args.LR)
        self.args = args
        self.device = device
        self._reinit()

    def _reinit(self):
        self.client_user_grad = 0
        self.client_user_grad_list = []
        self.client_item_grad = []
        self.client_other_grad = []
        self.client_sample_num = []
        self.attacker_list = []
        self.client_id_list = []
        self.server_optimizer.zero_grad()

    @torch.no_grad()
    def collect_client_update(self, client_user_grad, client_item_grad, client_other_grad,
                              client_sample_num, is_attacker, client_id=None):
        # direct add for user embedding
        self.client_user_grad += client_user_grad
        self.client_user_grad_list.append(client_user_grad)
        self.client_item_grad.append(client_item_grad)
        self.client_other_grad.append(client_other_grad)
        self.client_sample_num.append(client_sample_num)
        self.attacker_list.append(is_attacker)
        self.client_id_list.append(client_id)

    @torch.no_grad()
    def agg(self):
        client_sample_num = torch.tensor(self.client_sample_num).to(self.device)
        client_weight = client_sample_num.float() / client_sample_num.sum()

        # In-place accumulation to avoid allocating a large stacked tensor on MPS
        agg_client_item_grad = torch.zeros_like(self.client_item_grad[0])
        for w, g in zip(client_weight, self.client_item_grad):
            agg_client_item_grad.add_(g, alpha=w.item())

        agg_client_other_grad = torch.zeros_like(self.client_other_grad[0])
        for w, g in zip(client_weight, self.client_other_grad):
            agg_client_other_grad.add_(g, alpha=w.item())

        vector_to_grad(self.client_user_grad, self.server_model.user_model)
        vector_to_grad(agg_client_item_grad, self.server_model.item_model)
        vector_to_grad(agg_client_other_grad, self.server_model.predictor)
        self.server_optimizer.step()
        self._reinit()


class TrimmedMean(FedAdam):
    """Trimmed Mean defense - removes top/bottom percentile before averaging"""
    @torch.no_grad()
    def agg(self):
        trim_ratio = getattr(self.args, 'TRIM_RATIO', 0.1)  # Default 10% trim
        
        client_item_grad = torch.stack(self.client_item_grad, dim=0)
        client_other_grad = torch.stack(self.client_other_grad, dim=0)
        client_grad = torch.cat([client_item_grad, client_other_grad], dim=-1)
        
        num_clients = client_grad.shape[0]
        num_trim = int(num_clients * trim_ratio)
        
        # Sort and trim along client dimension for each gradient element
        sorted_grads, _ = torch.sort(client_grad, dim=0)
        trimmed_grads = sorted_grads[num_trim:num_clients-num_trim]
        agg_grad = trimmed_grads.mean(dim=0)
        
        agg_client_item_grad = agg_grad[:client_item_grad.shape[1]]
        agg_client_other_grad = agg_grad[client_item_grad.shape[1]:]
        
        vector_to_grad(self.client_user_grad, self.server_model.user_model)
        vector_to_grad(agg_client_item_grad, self.server_model.item_model)
        vector_to_grad(agg_client_other_grad, self.server_model.predictor)
        self.server_optimizer.step()
        self._reinit()


class Krum(FedAdam):
    """Krum defense - selects the gradient closest to others"""
    @torch.no_grad()
    def agg(self):
        client_item_grad = torch.stack(self.client_item_grad, dim=0)
        client_other_grad = torch.stack(self.client_other_grad, dim=0)
        client_grad = torch.cat([client_item_grad, client_other_grad], dim=-1)
        
        num_clients = client_grad.shape[0]
        num_byzantine = self.args.ATTACKER_PER_ROUND
        
        # Compute pairwise distances
        scores = torch.zeros(num_clients, device=self.device)
        for i in range(num_clients):
            dists = ((client_grad - client_grad[i])**2).sum(dim=-1)
            dists[i] = float('inf')  # Exclude self
            # Sum of n-f-2 closest distances
            k = max(1, num_clients - num_byzantine - 2)
            scores[i] = torch.topk(dists, k=k, largest=False).values.sum()
        
        # Select client with minimum score (most representative)
        selected_idx = scores.argmin()
        agg_grad = client_grad[selected_idx]
        
        agg_client_item_grad = agg_grad[:client_item_grad.shape[1]]
        agg_client_other_grad = agg_grad[client_item_grad.shape[1]:]
        
        vector_to_grad(self.client_user_grad, self.server_model.user_model)
        vector_to_grad(agg_client_item_grad, self.server_model.item_model)
        vector_to_grad(agg_client_other_grad, self.server_model.predictor)
        self.server_optimizer.step()
        self._reinit()


class MultiKrum(FedAdam):
    """Multi-Krum defense - averages m clients with lowest Krum scores"""
    @torch.no_grad()
    def agg(self):
        client_item_grad = torch.stack(self.client_item_grad, dim=0)
        client_other_grad = torch.stack(self.client_other_grad, dim=0)
        client_grad = torch.cat([client_item_grad, client_other_grad], dim=-1)
        client_sample_num = torch.tensor(self.client_sample_num).to(self.device)
        
        num_clients = client_grad.shape[0]
        num_byzantine = self.args.ATTACKER_PER_ROUND
        
        # Compute Krum scores
        scores = torch.zeros(num_clients, device=self.device)
        for i in range(num_clients):
            dists = ((client_grad - client_grad[i])**2).sum(dim=-1)
            dists[i] = float('inf')
            k = max(1, num_clients - num_byzantine - 2)
            scores[i] = torch.topk(dists, k=k, largest=False).values.sum()
        
        # Select m = n - f clients with lowest scores
        m = num_clients - num_byzantine
        selected_indices = torch.topk(scores, k=m, largest=False).indices
        
        selected_grads = client_grad[selected_indices]
        selected_weights = client_sample_num[selected_indices]
        selected_weights = selected_weights.float() / selected_weights.sum()
        agg_grad = torch.matmul(selected_weights, selected_grads)
        
        agg_client_item_grad = agg_grad[:client_item_grad.shape[1]]
        agg_client_other_grad = agg_grad[client_item_grad.shape[1]:]
        
        vector_to_grad(self.client_user_grad, self.server_model.user_model)
        vector_to_grad(agg_client_item_grad, self.server_model.item_model)
        vector_to_grad(agg_client_other_grad, self.server_model.predictor)
        self.server_optimizer.step()
        self._reinit()


class NormBound(FedAdam):
    """Norm Bounding defense - clips gradient norms before aggregation"""
    @torch.no_grad()
    def agg(self):
        client_sample_num = torch.tensor(self.client_sample_num).to(self.device)
        client_weight = client_sample_num.float() / client_sample_num.sum()
        
        client_item_grad = torch.stack(self.client_item_grad, dim=0)
        client_other_grad = torch.stack(self.client_other_grad, dim=0)
        client_grad = torch.cat([client_item_grad, client_other_grad], dim=-1)
        
        # Clip gradient norms
        grad_norms = (client_grad**2).sum(dim=-1, keepdim=True).sqrt()
        clip_coef = torch.clamp(grad_norms / self.args.NORM_BOUND, min=1.0)
        client_grad = client_grad / clip_coef
        
        agg_grad = torch.matmul(client_weight, client_grad)
        
        agg_client_item_grad = agg_grad[:client_item_grad.shape[1]]
        agg_client_other_grad = agg_grad[client_item_grad.shape[1]:]
        
        vector_to_grad(self.client_user_grad, self.server_model.user_model)
        vector_to_grad(agg_client_item_grad, self.server_model.item_model)
        vector_to_grad(agg_client_other_grad, self.server_model.predictor)
        self.server_optimizer.step()
        self._reinit()


class FLWBC(FedAdam):
    """FL-WBC (Weighted Byzantine-resilient Clustering) defense"""
    @torch.no_grad()
    def agg(self):
        client_sample_num = torch.tensor(self.client_sample_num).to(self.device)
        client_item_grad = torch.stack(self.client_item_grad, dim=0)
        client_other_grad = torch.stack(self.client_other_grad, dim=0)
        client_grad = torch.cat([client_item_grad, client_other_grad], dim=-1)
        
        num_clients = client_grad.shape[0]
        
        # Compute pairwise cosine similarities
        grad_norms = client_grad.norm(dim=-1, keepdim=True)
        normalized_grads = client_grad / (grad_norms + 1e-10)
        cos_sim = torch.mm(normalized_grads, normalized_grads.t())
        
        # Compute trust scores based on average similarity
        trust_scores = cos_sim.mean(dim=1)
        trust_scores = torch.clamp(trust_scores, min=0)
        
        # Weight by trust scores and sample counts
        weights = trust_scores * client_sample_num.float()
        weights = weights / weights.sum()
        
        agg_grad = torch.matmul(weights, client_grad)
        
        agg_client_item_grad = agg_grad[:client_item_grad.shape[1]]
        agg_client_other_grad = agg_grad[client_item_grad.shape[1]:]
        
        vector_to_grad(self.client_user_grad, self.server_model.user_model)
        vector_to_grad(agg_client_item_grad, self.server_model.item_model)
        vector_to_grad(agg_client_other_grad, self.server_model.predictor)
        self.server_optimizer.step()
        self._reinit()


class UNION(FedAdam):
    def __init__(self, server_model, args, device):
        super().__init__(server_model, args, device)
        self.rng = np.random.default_rng(133)
        if not os.path.exists(args.GAP_CACHE):
            print('Preparing cache data for Gap Statistics')
            rng = np.random.default_rng()
            cache_data = {k: [] for k in range(2)}
            for k in range(1, 3):
                tmp = []
                for _ in tqdm(range(10000)):
                    random_samples = rng.uniform(low=0, high=1, size=(args.USER_SAMPLE_NUM, 1))
                    random_kmeans = KMeans(n_clusters=k, tol=1e-7).fit(random_samples)
                    tmp.append(random_kmeans.inertia_)
                tmp = np.log(np.array(tmp))
                cache_data[k] = tmp
            self.cache_data = cache_data
            with open(args.GAP_CACHE, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print('Dumping cache data to', args.GAP_CACHE)
        else:
            with open(args.GAP_CACHE, 'rb') as f:
                self.cache_data = pickle.load(f)
            print('Loading cache data from', args.GAP_CACHE)

    @torch.no_grad()
    def cal_uniformity(self, emb):
        '''
            emb: num_client, num_items, emb_dim
        '''
        num_client, num_items, _ = emb.shape
        cnt = num_items * (num_items - 1)
        pdist = torch.norm(emb.unsqueeze(dim=2) - emb.unsqueeze(dim=1), dim=-1, p=2)
        return pdist.reshape(num_client, -1).sum(dim=-1) / cnt

    @torch.no_grad()
    def agg(self):
        num_clients = len(self.attacker_list)
        sample_iid = self.rng.choice(self.args.NUM_ITEMS, self.args.AGG_SAMPLE_NUM)
        eps = 1e-8
        beta1, beta2 = (0.9, 0.999)

        current_item_emb = \
            self.server_model.item_model.item_embedding.weight.clone().detach()[sample_iid]
        total_item_emb = current_item_emb.unsqueeze(dim=0).expand(num_clients, -1, -1)
        total_item_grad = torch.stack(self.client_item_grad,
                                      dim=0).reshape(num_clients, self.args.NUM_ITEMS,
                                                     self.args.EMBDIM)[:, sample_iid]

        item_emb_param = self.server_model.item_model.item_embedding.weight
        item_emb_state = self.server_optimizer.state.get(item_emb_param, {})
        if len(item_emb_state) == 0:
            step = 0
            exp_avg = torch.zeros_like(current_item_emb, memory_format=torch.preserve_format)
            exp_avg_sq = torch.zeros_like(current_item_emb, memory_format=torch.preserve_format)
        else:
            step = item_emb_state['step']
            exp_avg = item_emb_state['exp_avg'][sample_iid]
            exp_avg_sq = item_emb_state['exp_avg_sq'][sample_iid]
        exp_avg = exp_avg.unsqueeze(dim=0).expand_as(total_item_grad)
        exp_avg_sq = exp_avg_sq.unsqueeze(dim=0).expand_as(total_item_grad)
        step += 1
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        exp_avg = exp_avg.mul(beta1).add(total_item_grad, alpha=1 - beta1)
        exp_avg_sq = exp_avg_sq.mul(beta2).addcmul(total_item_grad,
                                                   total_item_grad,
                                                   value=1 - beta2)
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add(eps)
        step_size = self.args.LR / bias_correction1
        total_item_emb = total_item_emb.addcdiv(exp_avg, denom, value=-step_size)

        total_uniformity = self.cal_uniformity(total_item_emb).reshape(-1, 1)
        more_than_one, two_cluster_labels = GapStatistics(total_uniformity, self.args, self.cache_data)

        cluster_idx = {
            0: torch.where(two_cluster_labels == 0)[0],
            1: torch.where(two_cluster_labels == 1)[0]
        }
        abnormal_idx = 0 if len(cluster_idx[0]) < len(cluster_idx[1]) else 1
        selected_idx = 1 - abnormal_idx

        if more_than_one or len(cluster_idx[abnormal_idx]) <= self.args.ATTACKER_PER_ROUND:
            filtered_clients = cluster_idx[abnormal_idx]
            selected_clients = cluster_idx[selected_idx]
        else:
            selected_clients = list(range(num_clients))
            filtered_clients = []

        attacker_list = torch.tensor(self.attacker_list).to(self.device)
        total_attacker_num = attacker_list.sum()
        filter_precision = attacker_list[filtered_clients].sum() / (len(filtered_clients) + 1e-12)
        filter_recall = attacker_list[filtered_clients].sum() / (total_attacker_num + 1e-12)

        client_sample_num = torch.tensor(self.client_sample_num).to(self.device)
        client_item_grad = torch.stack(self.client_item_grad, dim=0)
        client_other_grad = torch.stack(self.client_other_grad, dim=0)
        client_grad = torch.cat([client_item_grad, client_other_grad], dim=-1)
        selected_grad = client_grad[selected_clients]
        selected_client_sample_num = client_sample_num[selected_clients]
        selected_client_weight = \
            selected_client_sample_num.float() / selected_client_sample_num.sum()

        agg_client_grad = torch.matmul(selected_client_weight, selected_grad)
        agg_client_item_grad = agg_client_grad[:client_item_grad.shape[1]]
        agg_client_other_grad = agg_client_grad[client_item_grad.shape[1]:]

        vector_to_grad(self.client_user_grad, self.server_model.user_model)
        vector_to_grad(agg_client_item_grad, self.server_model.item_model)
        vector_to_grad(agg_client_other_grad, self.server_model.predictor)
        self.server_optimizer.step()
        self._reinit()

        return filter_precision, filter_recall, len(filtered_clients)


class MultiKrumUNION(UNION):
    @torch.no_grad()
    def agg(self):
        client_sample_num = torch.tensor(self.client_sample_num).to(self.device)
        client_item_grad = torch.stack(self.client_item_grad, dim=0)
        client_other_grad = torch.stack(self.client_other_grad, dim=0)
        client_grad = torch.cat([client_item_grad, client_other_grad], dim=-1)
        num_clients = len(self.attacker_list)
        sample_iid = self.rng.choice(self.args.NUM_ITEMS, self.args.AGG_SAMPLE_NUM)
        eps = 1e-8
        beta1, beta2 = (0.9, 0.999)

        # UNION
        current_item_emb = \
            self.server_model.item_model.item_embedding.weight.clone().detach()[sample_iid]
        total_item_emb = current_item_emb.unsqueeze(dim=0).expand(num_clients, -1, -1)
        total_item_grad = torch.stack(self.client_item_grad,
                                      dim=0).reshape(num_clients, self.args.NUM_ITEMS,
                                                     self.args.EMBDIM)[:, sample_iid]

        item_emb_param = self.server_model.item_model.item_embedding.weight
        item_emb_state = self.server_optimizer.state.get(item_emb_param, {})
        if len(item_emb_state) == 0:
            step = 0
            exp_avg = torch.zeros_like(current_item_emb, memory_format=torch.preserve_format)
            exp_avg_sq = torch.zeros_like(current_item_emb, memory_format=torch.preserve_format)
        else:
            step = item_emb_state['step']
            exp_avg = item_emb_state['exp_avg'][sample_iid]
            exp_avg_sq = item_emb_state['exp_avg_sq'][sample_iid]
        exp_avg = exp_avg.unsqueeze(dim=0).expand_as(total_item_grad)
        exp_avg_sq = exp_avg_sq.unsqueeze(dim=0).expand_as(total_item_grad)
        step += 1
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        exp_avg = exp_avg.mul(beta1).add(total_item_grad, alpha=1 - beta1)
        exp_avg_sq = exp_avg_sq.mul(beta2).addcmul(total_item_grad,
                                                   total_item_grad,
                                                   value=1 - beta2)
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add(eps)
        step_size = self.args.LR / bias_correction1
        total_item_emb = total_item_emb.addcdiv(exp_avg, denom, value=-step_size)

        total_uniformity = self.cal_uniformity(total_item_emb).reshape(-1, 1)
        more_than_one, two_cluster_labels = GapStatistics(total_uniformity, self.args, self.cache_data)

        cluster_idx = {
            0: torch.where(two_cluster_labels == 0)[0],
            1: torch.where(two_cluster_labels == 1)[0]
        }
        abnormal_idx = 0 if len(cluster_idx[0]) < len(cluster_idx[1]) else 1

        if more_than_one or len(cluster_idx[abnormal_idx]) <= self.args.ATTACKER_PER_ROUND:
            ours_filtered_clients = cluster_idx[abnormal_idx].cpu().tolist()
        else:
            ours_filtered_clients = []

        # MultiKrum
        client_scores = torch.zeros((num_clients, num_clients), device=self.device)
        for i in range(num_clients):
            client_scores[i] = ((client_grad - client_grad[i])**2).sum(dim=-1)
            client_scores[i, i] = client_scores[i].max() + 1

        topk_client_scores = torch.topk(client_scores,
                                        k=num_clients - self.args.ATTACKER_PER_ROUND - 2,
                                        dim=-1,
                                        largest=False).values
        sum_client_scores = torch.sum(topk_client_scores, dim=-1)
        mk_filtered_clients = torch.topk(sum_client_scores,
                                         k=self.args.ATTACKER_PER_ROUND,
                                         largest=True).indices.cpu().tolist()

        filtered_clients = list(set(ours_filtered_clients) | set(mk_filtered_clients))
        selected_clients = [i for i in range(num_clients) if i not in filtered_clients]

        attacker_list = torch.tensor(self.attacker_list).to(self.device)
        total_attacker_num = attacker_list.sum()
        filter_precision = attacker_list[filtered_clients].sum() / (len(filtered_clients) + 1e-12)
        filter_recall = attacker_list[filtered_clients].sum() / (total_attacker_num + 1e-12)

        selected_grad = client_grad[selected_clients]
        selected_client_sample_num = client_sample_num[selected_clients]
        selected_client_weight = \
            selected_client_sample_num.float() / selected_client_sample_num.sum()

        agg_client_grad = torch.matmul(selected_client_weight, selected_grad)
        agg_client_item_grad = agg_client_grad[:client_item_grad.shape[1]]
        agg_client_other_grad = agg_client_grad[client_item_grad.shape[1]:]

        vector_to_grad(self.client_user_grad, self.server_model.user_model)
        vector_to_grad(agg_client_item_grad, self.server_model.item_model)
        vector_to_grad(agg_client_other_grad, self.server_model.predictor)
        self.server_optimizer.step()
        self._reinit()

        return filter_precision, filter_recall, len(filtered_clients)


class NormBoundUNION(UNION):
    @torch.no_grad()
    def agg(self):
        client_sample_num = torch.tensor(self.client_sample_num).to(self.device)
        client_item_grad = torch.stack(self.client_item_grad, dim=0)
        client_other_grad = torch.stack(self.client_other_grad, dim=0)
        client_grad = torch.cat([client_item_grad, client_other_grad], dim=-1)
        num_clients = len(self.attacker_list)
        sample_iid = self.rng.choice(self.args.NUM_ITEMS, self.args.AGG_SAMPLE_NUM)
        eps = 1e-8
        beta1, beta2 = (0.9, 0.999)

        current_item_emb = \
            self.server_model.item_model.item_embedding.weight.clone().detach()[sample_iid]
        total_item_emb = current_item_emb.unsqueeze(dim=0).expand(num_clients, -1, -1)
        total_item_grad = torch.stack(self.client_item_grad,
                                      dim=0).reshape(num_clients, self.args.NUM_ITEMS,
                                                     self.args.EMBDIM)[:, sample_iid]

        item_emb_param = self.server_model.item_model.item_embedding.weight
        item_emb_state = self.server_optimizer.state.get(item_emb_param, {})
        if len(item_emb_state) == 0:
            step = 0
            exp_avg = torch.zeros_like(current_item_emb, memory_format=torch.preserve_format)
            exp_avg_sq = torch.zeros_like(current_item_emb, memory_format=torch.preserve_format)
        else:
            step = item_emb_state['step']
            exp_avg = item_emb_state['exp_avg'][sample_iid]
            exp_avg_sq = item_emb_state['exp_avg_sq'][sample_iid]
        exp_avg = exp_avg.unsqueeze(dim=0).expand_as(total_item_grad)
        exp_avg_sq = exp_avg_sq.unsqueeze(dim=0).expand_as(total_item_grad)
        step += 1
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        exp_avg = exp_avg.mul(beta1).add(total_item_grad, alpha=1 - beta1)
        exp_avg_sq = exp_avg_sq.mul(beta2).addcmul(total_item_grad,
                                                   total_item_grad,
                                                   value=1 - beta2)
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add(eps)
        step_size = self.args.LR / bias_correction1
        total_item_emb = total_item_emb.addcdiv(exp_avg, denom, value=-step_size)

        total_uniformity = self.cal_uniformity(total_item_emb).reshape(-1, 1)
        more_than_one, two_cluster_labels = GapStatistics(total_uniformity, self.args, self.cache_data)

        cluster_idx = {
            0: torch.where(two_cluster_labels == 0)[0],
            1: torch.where(two_cluster_labels == 1)[0]
        }
        abnormal_idx = 0 if len(cluster_idx[0]) < len(cluster_idx[1]) else 1
        selected_idx = 1 - abnormal_idx

        if more_than_one or len(cluster_idx[abnormal_idx]) <= self.args.ATTACKER_PER_ROUND:
            filtered_clients = cluster_idx[abnormal_idx]
            selected_clients = cluster_idx[selected_idx]
        else:
            selected_clients = list(range(num_clients))
            filtered_clients = []

        attacker_list = torch.tensor(self.attacker_list).to(self.device)
        total_attacker_num = attacker_list.sum()
        filter_precision = attacker_list[filtered_clients].sum() / (len(filtered_clients) + 1e-12)
        filter_recall = attacker_list[filtered_clients].sum() / (total_attacker_num + 1e-12)

        selected_grad = client_grad[selected_clients]
        selected_grad_norm = (selected_grad**2).sum(dim=-1, keepdim=True).sqrt()
        clip_coef = torch.clamp(selected_grad_norm / self.args.NORM_BOUND, min=1.0)
        selected_grad /= clip_coef
        selected_client_sample_num = client_sample_num[selected_clients]
        selected_client_weight = \
            selected_client_sample_num.float() / selected_client_sample_num.sum()

        agg_client_grad = torch.matmul(selected_client_weight, selected_grad)
        agg_client_item_grad = agg_client_grad[:client_item_grad.shape[1]]
        agg_client_other_grad = agg_client_grad[client_item_grad.shape[1]:]

        vector_to_grad(self.client_user_grad, self.server_model.user_model)
        vector_to_grad(agg_client_item_grad, self.server_model.item_model)
        vector_to_grad(agg_client_other_grad, self.server_model.predictor)
        self.server_optimizer.step()
        self._reinit()

        return filter_precision, filter_recall, len(filtered_clients)


@torch.no_grad()
def GapStatistics(metrics, args, cache_data):
    rng = np.random.default_rng()
    low, high = metrics.min(), metrics.max()
    normalized_metrics = (metrics - low) / (high - low)
    gap, s = [], []
    for k in range(1, 3):
        cluster_centroids, cluster_labels = kmeans(X=normalized_metrics,
                                                   num_clusters=k,
                                                   init='kmeans++',
                                                   tol=1e-7,
                                                   verbose=False,
                                                   seed=1)
        if k == 2:
            two_cluster_labels = cluster_labels
        V_k = ((normalized_metrics - cluster_centroids[cluster_labels])**2).sum().cpu().numpy()
        V_kb = rng.choice(cache_data[k], size=args.GAP_SAMPLE, replace=False)
        gap_k = V_kb.mean() - np.log(V_k)
        V_k_std = V_kb.std() * np.sqrt((1 + args.GAP_SAMPLE) / args.GAP_SAMPLE)
        gap.append(gap_k)
        s.append(V_k_std)
    return gap[0] < gap[1] - s[1], two_cluster_labels


class ECF(FedAdam):
    """
    Embedding Consistency Filtering (ECF) defense mechanism.
    Four interconnected components detect and counteract covert untargeted poisoning attacks:
    1. Temporal Embedding Drift Monitoring (TEDM)
    2. Interdimensional Consistency Check (IDC) via discrete KL-D and 1D EMD (Wasserstein distance)
    3. Compatibility Drop Estimator (CDE) via cosine similarity change simulation
    4. Credibility-Weighted Aggregation
    """
    def __init__(self, server_model, args, device):
        super().__init__(server_model, args, device)
        from collections import deque, OrderedDict
        self.last_client_update = OrderedDict()  # client_id -> item embedding update tensor
        self.max_clients_to_store = 5000
        self.drift_history = deque(maxlen=200)  # rolling history of drift norm
        self.hist_history = deque(maxlen=50)  # rolling history of client histograms
        
        # ECF Hyperparameters
        self.k_sensitivity = getattr(self.args, 'ECF_K', 1.5)
        self.gamma_tedm = getattr(self.args, 'ECF_GAMMA_TEDM', 1.0)
        self.bins_idc = getattr(self.args, 'ECF_BINS_IDC', 20)
        self.tau_idc = getattr(self.args, 'ECF_TAU_IDC', 2.0)
        self.alpha_idc = getattr(self.args, 'ECF_ALPHA_IDC', 2.0)
        self.lambda_cde = getattr(self.args, 'ECF_LAMBDA_CDE', 10.0)

    @torch.no_grad()
    def agg(self):
        num_clients = len(self.attacker_list)
        if num_clients == 0:
            self._reinit()
            return 0.0, 0.0, 0
            
        client_item_grad = torch.stack(self.client_item_grad, dim=0) # (num_clients, item_grad_dim)
        client_other_grad = torch.stack(self.client_other_grad, dim=0) # (num_clients, other_grad_dim)
        client_grad = torch.cat([client_item_grad, client_other_grad], dim=-1)
        client_sample_num = torch.tensor(self.client_sample_num).to(self.device)
        
        emb_dim = self.args.EMBDIM
        num_items = self.args.NUM_ITEMS
        
        # Reshape item updates for each client: (num_clients, num_items, emb_dim)
        client_item_updates = client_item_grad.reshape(num_clients, num_items, emb_dim)
        
        # ----------------------------------------------------
        # 1. Temporal Embedding Drift Monitoring (TEDM) - Offloaded to CPU to bound GPU memory
        # ----------------------------------------------------
        drift_norms = []
        for c in range(num_clients):
            uid = self.client_id_list[c]
            current_update_cpu = client_item_updates[c].cpu()
            
            if uid in self.last_client_update:
                prev_indices, prev_values = self.last_client_update[uid]
                diff = current_update_cpu.clone()
                diff[prev_indices] -= prev_values
                drift = torch.norm(diff, p=2)
            else:
                drift = torch.norm(current_update_cpu, p=2)
                
            drift_norms.append(drift)
            
            # Store active index-value representation on CPU to avoid memory exhaustion
            row_norms = torch.norm(current_update_cpu, dim=-1)
            # Filter out small gradients (weight decay) and bound to top 1000 items
            active_mask = row_norms > 1e-4
            active_indices = torch.where(active_mask)[0]
            if active_indices.shape[0] > 1000:
                _, top_indices = torch.topk(row_norms, k=1000, sorted=False)
                active_indices = top_indices.cpu()
                
            active_values = current_update_cpu[active_indices].clone()
            self.last_client_update[uid] = (active_indices, active_values)
            self.last_client_update.move_to_end(uid)
            
            # Bound dictionary size
            while len(self.last_client_update) > self.max_clients_to_store:
                self.last_client_update.popitem(last=False)
            
        drift_norms = torch.stack(drift_norms).to(self.device) # (num_clients,)
        
        for norm in drift_norms.cpu().tolist():
            self.drift_history.append(norm)
            
        if len(self.drift_history) > 1:
            drift_arr = np.array(self.drift_history)
            mu = float(drift_arr.mean())
            sigma = float(drift_arr.std()) + 1e-12
        else:
            mu = float(drift_norms.mean().cpu().item())
            sigma = float(drift_norms.std().cpu().item()) + 1e-12
            
        z_scores = torch.abs(drift_norms - mu) / sigma
        s_tedm = torch.exp(-self.gamma_tedm * torch.clamp(z_scores - self.k_sensitivity, min=0.0))
        
        # ----------------------------------------------------
        # 2. Interdimensional Consistency Check (IDC) - Offloaded to CPU to bypass MPS memory leaks
        # ----------------------------------------------------
        client_hists = []
        for c in range(num_clients):
            current_update = client_item_updates[c]
            
            # Optimization: Filter out zero-gradient items. We use a threshold of 1e-4
            # to filter out background gradients from weight decay, and limit to top 1000
            # items to bound memory consumption and avoid MPS out-of-memory errors.
            row_norms = torch.norm(current_update, dim=-1)
            active_mask = row_norms > 1e-4
            active_rows = current_update[active_mask]
            
            if active_rows.shape[0] > 1000:
                _, top_indices = torch.topk(row_norms, k=1000, sorted=False)
                active_rows = current_update[top_indices]
            elif active_rows.shape[0] == 0:
                active_rows = torch.zeros((1, emb_dim), device=self.device)
                
            # Perform pairwise differences and histogram extraction on CPU
            active_rows_cpu = active_rows.cpu()
            E_unsqueezed1 = active_rows_cpu.unsqueeze(2) # (N, emb_dim, 1)
            E_unsqueezed2 = active_rows_cpu.unsqueeze(1) # (N, 1, emb_dim)
            D = torch.abs(E_unsqueezed1 - E_unsqueezed2) # (N, emb_dim, emb_dim)
            
            triu = torch.triu_indices(emb_dim, emb_dim, offset=1, device='cpu')
            D_pairs = D[:, triu[0], triu[1]].flatten()
            
            max_val = max(0.01, float(D_pairs.max().item()))
            # Compute histc on CPU and move only the final micro-tensor (20 bins) to GPU device
            hist = torch.histc(D_pairs, bins=self.bins_idc, min=0.0, max=max_val).to(self.device)
            hist = hist / (hist.sum() + 1e-12)
            client_hists.append(hist)
            
        client_hists = torch.stack(client_hists) # (num_clients, bins_idc)
        
        mean_hist = client_hists.mean(dim=0)
        self.hist_history.append(mean_hist.cpu())
        
        hist_list = list(self.hist_history)
        Q = torch.stack(hist_list).mean(dim=0).to(self.device)
        Q = Q / (Q.sum() + 1e-12)
        
        eps = 1e-12
        kl_divs = []
        emd_divs = []
        for c in range(num_clients):
            P = client_hists[c]
            
            # Kullback-Leibler Divergence
            kl = torch.sum(P * torch.log((P + eps) / (Q + eps)))
            kl_divs.append(kl)
            
            # Earth Mover's Distance
            cdf_P = torch.cumsum(P, dim=0)
            cdf_Q = torch.cumsum(Q, dim=0)
            emd = torch.sum(torch.abs(cdf_P - cdf_Q))
            emd_divs.append(emd)
            
        kl_divs = torch.stack(kl_divs)
        emd_divs = torch.stack(emd_divs)
        idc_dists = kl_divs + emd_divs
        
        if num_clients > 1:
            mu_D = idc_dists.mean()
            sigma_D = idc_dists.std() + 1e-12
            z_scores_D = (idc_dists - mu_D) / sigma_D
        else:
            z_scores_D = torch.zeros(num_clients, device=self.device)
            
        s_idc = 1.0 / (1.0 + torch.exp(self.alpha_idc * (z_scores_D - self.tau_idc)))
        
        # ----------------------------------------------------
        # 3. Compatibility Drop Estimator (CDE)
        # ----------------------------------------------------
        cde_drops = []
        user_embedding_weight = self.server_model.user_model.user_embedding.weight
        item_embedding_weight = self.server_model.item_model.item_embedding.weight
        
        for c in range(num_clients):
            uid = self.client_id_list[c]
            current_update = client_item_updates[c]
            
            row_norms = torch.norm(current_update, dim=-1)
            active_iids = torch.where(row_norms > 1e-8)[0]
            
            if len(active_iids) == 0:
                cde_drops.append(torch.tensor(0.0, device=self.device))
                continue
                
            # Cosine similarity before update
            E_user = user_embedding_weight[uid] # (emb_dim,)
            E_items = item_embedding_weight[active_iids] # (num_active, emb_dim)
            
            E_user_norm = E_user / (E_user.norm(p=2) + eps)
            E_items_norm = E_items / (E_items.norm(p=2, dim=-1, keepdim=True) + eps)
            scores_before = torch.matmul(E_items_norm, E_user_norm)
            
            # Simulate updates
            g_user_c_flat = self.client_user_grad_list[c]
            g_user_c = g_user_c_flat.reshape(self.args.NUM_USERS, emb_dim)[uid]
            g_item_c = current_update[active_iids]
            
            E_user_new = E_user - self.args.LR * g_user_c
            E_items_new = E_items - self.args.LR * g_item_c
            
            # Cosine similarity after update
            E_user_new_norm = E_user_new / (E_user_new.norm(p=2) + eps)
            E_items_new_norm = E_items_new / (E_items_new.norm(p=2, dim=-1, keepdim=True) + eps)
            scores_after = torch.matmul(E_items_new_norm, E_user_new_norm)
            
            delta_S = (scores_after - scores_before).mean()
            cde_drops.append(delta_S)
            
        cde_drops = torch.stack(cde_drops)
        s_cde = torch.exp(-self.lambda_cde * torch.clamp(-cde_drops, min=0.0))
        
        # ----------------------------------------------------
        # 4. Credibility-Weighted Aggregation
        # ----------------------------------------------------
        credibility_scores = s_tedm * s_idc * s_cde # (num_clients,)
        
        # Calculate final aggregation weights
        weighted_sample_num = credibility_scores * client_sample_num.float()
        sum_weights = weighted_sample_num.sum()
        if sum_weights > eps:
            weights = weighted_sample_num / sum_weights
        else:
            weights = torch.ones(num_clients, device=self.device) / num_clients
            
        agg_grad = torch.matmul(weights, client_grad)
        agg_client_item_grad = agg_grad[:client_item_grad.shape[1]]
        agg_client_other_grad = agg_grad[client_item_grad.shape[1]:]
        
        vector_to_grad(self.client_user_grad, self.server_model.user_model)
        vector_to_grad(agg_client_item_grad, self.server_model.item_model)
        vector_to_grad(agg_client_other_grad, self.server_model.predictor)
        self.server_optimizer.step()
        # Calculate metrics for logging
        # Define filtered clients as those with low credibility score (< 0.5)
        filtered_clients = torch.where(credibility_scores < 0.5)[0].cpu().tolist()
        attacker_list = torch.tensor(self.attacker_list).to(self.device)
        total_attacker_num = attacker_list.sum()
        filter_precision = attacker_list[filtered_clients].sum() / (len(filtered_clients) + 1e-12)
        filter_recall = attacker_list[filtered_clients].sum() / (total_attacker_num + 1e-12)
        
        self._reinit()
        return filter_precision, filter_recall, len(filtered_clients)

