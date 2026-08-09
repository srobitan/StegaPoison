import torch
import numpy as np

from client import Client
from attacker import *
from agg import *


class Orchestra():
    def __init__(self, server_model, user_data, attacker_id_list, args, device):
        self.args = args
        self.device = device
        self.user_data = user_data
        self.attacker_id_list = attacker_id_list
        self.uid_list = list(self.user_data.keys())
        self.agg = eval(args.AGG_TYPE)(server_model, args, device)
        self.client = Client(self.args, self.device)
        self.rng = np.random.default_rng(self.args.SEED)
        if self.args.ATTACKER_STRAT is not None:
            self.attacker = eval(
                args.ATTACKER_STRAT)({_id: user_data[_id]
                                      for _id in self.attacker_id_list}, self.args, device)

    def update_one_round(self, step):
        total_client_loss, total_client_acc = [], []

        if self.args.ATTACKER_STRAT is not None and getattr(self.args, "ATTACKER_PER_ROUND", 0) > 0:
            num_attacker_one_round = self.args.ATTACKER_PER_ROUND
            benign_id_list = [uid for uid in self.uid_list if uid not in self.attacker_id_list]
            
            # Sample exactly the defined number of attackers and benign clients
            select_atk = self.rng.choice(self.attacker_id_list, size=num_attacker_one_round, replace=False).tolist()
            select_ben = self.rng.choice(benign_id_list, size=self.args.USER_SAMPLE_NUM - num_attacker_one_round, replace=False).tolist()
            select_uid = select_atk + select_ben
        else:
            if self.args.ATTACKER_STRAT is not None:
                select_uid = self.rng.choice(self.uid_list, size=self.args.USER_SAMPLE_NUM, replace=False).tolist()
                num_attacker_one_round = len(
                    [_id for _id in select_uid if _id in self.attacker_id_list])
            else:
                num_attacker_one_round = 0
                select_uid = self.rng.choice(self.uid_list, size=self.args.USER_SAMPLE_NUM, replace=False).tolist()

        if num_attacker_one_round > 0:
            attacker_loss = self.attacker.prepare(self.agg.server_model, step)
        else:
            attacker_loss = None

        attacker_grad_norm, benign_grad_norm = [], []
        attacker_updates, benign_updates = [], []

        for uid in select_uid:
            if num_attacker_one_round > 0 and uid in self.attacker_id_list:
                attacker_user_grad, attacker_item_grad, attacker_other_grad, \
                    attacker_sample_num = self.attacker.update(self.agg.server_model, uid)
                self.agg.collect_client_update(attacker_user_grad, attacker_item_grad,
                                               attacker_other_grad, attacker_sample_num, True, client_id=uid)

                grad_norm = ((attacker_item_grad**2).sum() + (attacker_other_grad**2).sum()).sqrt()
                attacker_grad_norm.append(grad_norm)
                # Save update vector for cosine similarity
                attacker_update_vec = torch.cat([attacker_item_grad.flatten(), attacker_other_grad.flatten()])
                attacker_updates.append(attacker_update_vec)
            else:
                client_user_grad, client_item_grad, client_other_grad, client_sample_num, \
                    client_loss, client_acc = self.client.update(self.agg.server_model,
                                                                 uid,
                                                                 self.user_data[uid])
                self.agg.collect_client_update(client_user_grad, client_item_grad,
                                               client_other_grad, client_sample_num, False, client_id=uid)
                total_client_loss.extend(client_loss)
                total_client_acc.extend(client_acc)
                client_grad_norm = ((client_item_grad**2).sum() + (client_other_grad**2).sum()).sqrt()
                benign_grad_norm.append(client_grad_norm)
                # Save update vector for cosine similarity
                client_update_vec = torch.cat([client_item_grad.flatten(), client_other_grad.flatten()])
                benign_updates.append(client_update_vec)

        filter_stat = self.agg.agg()
        average_client_loss = sum(total_client_loss) / len(total_client_loss)
        average_client_acc = sum(total_client_acc) / len(total_client_acc)

        # Calculate cosine similarity between each attacker update and the mean benign update
        mean_cosine_sim = None
        if len(attacker_updates) > 0 and len(benign_updates) > 0:
            mean_benign_update = torch.stack(benign_updates).mean(dim=0)
            cos_sims = []
            for atk_upd in attacker_updates:
                sim = torch.nn.functional.cosine_similarity(atk_upd, mean_benign_update, dim=0)
                cos_sims.append(sim)
            mean_cosine_sim = torch.stack(cos_sims).mean()

        if num_attacker_one_round > 0:
            attacker_grad_norm = torch.stack(attacker_grad_norm, dim=0).mean()
        else:
            attacker_grad_norm = None
            
        if len(benign_grad_norm) > 0:
            benign_grad_norm = torch.stack(benign_grad_norm, dim=0).mean()
        else:
            benign_grad_norm = None

        return average_client_loss, average_client_acc, num_attacker_one_round, \
            attacker_grad_norm, benign_grad_norm, mean_cosine_sim, filter_stat, attacker_loss
