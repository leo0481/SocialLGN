import time
from os.path import join

import torch

import Procedure
import register
import utils
import world
from register import dataset

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
torch.autograd.set_detect_anomaly(True)
Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        print(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")

best_ndcg, best_recall, best_pre = 0, 0, 0
best_ndcg_cold, best_recall_cold, best_pre_cold = 0, 0, 0
low_count, low_count_cold = 0, 0
try:
    for epoch in range(world.TRAIN_epochs + 1):
        print('======================')
        print(f'EPOCH[{epoch}/{world.TRAIN_epochs}]')
        start = time.time()
        if epoch % 10 == 1 or epoch == world.TRAIN_epochs:
            print("[TEST]")
            results = Procedure.Test(dataset, Recmodel, epoch, False)
            results_cold = Procedure.Test(dataset, Recmodel, epoch, True)
            if results['ndcg'][0] < best_ndcg:
                low_count += 1
                if low_count == 30:
                    if epoch > 1000:
                        break
                    else:
                        low_count = 0
            else:
                best_recall = results['recall'][0]
                best_ndcg = results['ndcg'][0]
                best_pre = results['precision'][0]
                low_count = 0

            if results_cold['ndcg'][0] > best_ndcg_cold:
                best_recall_cold = results_cold['recall'][0]
                best_ndcg_cold = results_cold['ndcg'][0]
                best_pre_cold = results_cold['precision'][0]
                low_count_cold = 0

        loss = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch)
        print(f'[saved][BPR aver loss{loss:.3e}]')
        torch.save(Recmodel.state_dict(), weight_file)
finally:
    print(f"\nbest recall at 10:{best_recall}")
    print(f"best ndcg at 10:{best_ndcg}")
    print(f"best precision at 10:{best_pre}")
    print(f"\nbest recall at 10:{best_recall_cold}")
    print(f"best ndcg at 10:{best_ndcg_cold}")
    print(f"best precision at 10:{best_pre_cold}")
