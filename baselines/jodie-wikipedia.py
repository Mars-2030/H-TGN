import os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import trange
import numpy as np

from library_data import *
import library_models as lib
from library_models import *
from sklearn.metrics import average_precision_score, roc_auc_score

# --------------------------
# SET ARGUMENTS HERE (NO argparse!)
# --------------------------
class Args:
    network = "wikipedia"        # change to "reddit" or "mooc" etc
    model = "jodie"
    gpu = -1                     # -1 = auto-select free GPU
    epochs = 50
    embedding_dim = 64
    train_proportion = 0.7       # 70% train, 15% val, 15% test
    state_change = True

args = Args()
args.datapath = f"data/{args.network}.csv"

# --------------------------
# GPU SETUP
# --------------------------
if args.gpu == -1:
    args.gpu = select_free_gpu()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

def evaluate_on_validation(model,
                           user_embeddings,
                           item_embeddings,
                           user_sequence_id,
                           item_sequence_id,
                           timestamp_sequence,
                           feature_sequence,
                           user_timediffs_sequence,
                           item_timediffs_sequence,
                           user_previous_itemid_sequence,
                           validation_start_idx,
                           test_start_idx,
                           item_embedding_static,
                          user_embedding_static):
    """
    Evaluate model performance on the validation set.
    Returns AP, AUC, and MRR.
    """
    model.eval()
    all_true_labels, all_pred_scores, val_ranks = [], [], []

    with torch.no_grad():
        for j in range(validation_start_idx, test_start_idx):
            userid = user_sequence_id[j]
            itemid = item_sequence_id[j]

            y_true_vec = np.zeros(item_embeddings.shape[0])
            y_true_vec[itemid] = 1

            user_embedding_input = user_embeddings[userid, :].unsqueeze(0)
            item_embedding_prev = item_embeddings[user_previous_itemid_sequence[j], :].unsqueeze(0)

            feature_tensor = torch.Tensor(feature_sequence[j]).unsqueeze(0).cuda()
            user_timediff_tensor = torch.Tensor([user_timediffs_sequence[j]]).unsqueeze(1).cuda()

            user_projected_embedding = model.forward(
                user_embedding_input, item_embedding_prev,
                timediffs=user_timediff_tensor,
                features=feature_tensor,
                select='project'
            )

            user_item_embedding = torch.cat([
                user_projected_embedding,
                item_embedding_prev,
                item_embedding_static[user_previous_itemid_sequence[j], :].unsqueeze(0),
                user_embedding_static[userid, :].unsqueeze(0)
            ], dim=1)

            pred_item_embedding = model.predict_item_embedding(user_item_embedding)
            scores = torch.mm(
                pred_item_embedding,
                torch.cat([item_embeddings, item_embedding_static], dim=1).t()
            )
            scores = scores.detach().cpu().numpy().flatten()

            rank = (scores.argsort()[::-1] == itemid).nonzero()[0][0] + 1

            all_true_labels.append(y_true_vec)
            all_pred_scores.append(scores)
            val_ranks.append(rank)

    all_true_labels = np.vstack(all_true_labels)
    all_pred_scores = np.vstack(all_pred_scores)

    AP = average_precision_score(all_true_labels.ravel(), all_pred_scores.ravel())
    AUC = roc_auc_score(all_true_labels.ravel(), all_pred_scores.ravel())
    MRR = np.mean([1.0 / r for r in val_ranks])

    model.train()
    return AP, AUC, MRR


def run_single_experiment(seed, args):
    # --------------------------
    # SEED FIXING
    # --------------------------
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --------------------------
    # LOAD DATA
    # --------------------------
    [user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
     item2id, item_sequence_id, item_timediffs_sequence, 
     timestamp_sequence, feature_sequence, y_true] = load_network(args)
    
    num_interactions = len(user_sequence_id)
    num_users = len(user2id) 
    num_items = len(item2id) + 1 # one extra item for "none-of-these"
    num_features = len(feature_sequence[0])
    true_labels_ratio = len(y_true)/(1.0+sum(y_true)) # +1 in denominator in case there are no state change labels, which will throw an error. 
    print(f"*** Network statistics:\n  {num_users} users\n  {num_items} items\n"
          f"  {num_interactions} interactions\n  {sum(y_true)}/{len(y_true)} true labels ***\n")

# SET TRAINING, VALIDATION, TESTING, and TBATCH BOUNDARIES
    train_end_idx = int(num_interactions * 0.7)
    validation_start_idx = train_end_idx
    validation_end_idx = int(num_interactions * 0.85)  # 0.7 + 0.15
    test_start_idx = validation_end_idx
    test_end_idx = num_interactions  # covers the rest (0.15)

    timespan = timestamp_sequence[-1] - timestamp_sequence[0]
    tbatch_timespan = timespan / 500 

    # --------------------------
    # MODEL SETUP
    # --------------------------
    # INITIALIZE MODEL AND PARAMETERS
    model = JODIE(args, num_features, num_users, num_items).cuda()
    weight = torch.Tensor([1,true_labels_ratio]).cuda()
    crossEntropyLoss = nn.CrossEntropyLoss(weight=weight)
    MSELoss = nn.MSELoss()
    
    # INITIALIZE EMBEDDING
    initial_user_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).cuda(), dim=0)) # the initial user and item embeddings are learned during training as well
    initial_item_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).cuda(), dim=0))
    model.initial_user_embedding = initial_user_embedding
    model.initial_item_embedding = initial_item_embedding
    
    user_embeddings = initial_user_embedding.repeat(num_users, 1) # initialize all users to the same embedding 
    item_embeddings = initial_item_embedding.repeat(num_items, 1) # initialize all items to the same embedding
    item_embedding_static = Variable(torch.eye(num_items).cuda()) # one-hot vectors for static embeddings
    user_embedding_static = Variable(torch.eye(num_users).cuda()) # one-hot vectors for static embeddings 

    # INITIALIZE MODEL
    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    print(f"*** Training the JODIE model for {args.epochs} epochs ***")
    
    best_val_ap = 0.0
    patience_counter = 0
    max_patience = 10
    best_model_path = f"saved_models/jodie_best_{args.network}_{seed}.pth"
    
    epoch_times = []
    
    # variables to help using tbatch cache between epochs
    is_first_epoch = True
    cached_tbatches_user = {}
    cached_tbatches_item = {}
    cached_tbatches_interactionids = {}
    cached_tbatches_feature = {}
    cached_tbatches_user_timediffs = {}
    cached_tbatches_item_timediffs = {}
    cached_tbatches_previous_item = {}
    
    with trange(args.epochs) as progress_bar1:
        for ep in progress_bar1:
            progress_bar1.set_description(f'Epoch {ep+1}/{args.epochs}')
    
            epoch_start_time = time.time()
            total_loss, loss, total_interaction_count = 0, 0, 0
    
            # initialize embedding trajectory storage
            user_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).cuda())
            item_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).cuda())
    
            optimizer.zero_grad()
            reinitialize_tbatches()
    
            tbatch_start_time = None
            tbatch_to_insert = -1
    
            # ------------------------------
            # TRAINING LOOP
            # ------------------------------
            with trange(train_end_idx) as progress_bar2:
                for j in progress_bar2:
                    progress_bar2.set_description(f'Processed {j}th interactions')
    
                    if is_first_epoch:
                        # read interaction j
                        userid = user_sequence_id[j]
                        itemid = item_sequence_id[j]
                        feature = feature_sequence[j]
                        user_timediff = user_timediffs_sequence[j]
                        item_timediff = item_timediffs_sequence[j]
    
                        # create tbatch
                        tbatch_to_insert = max(lib.tbatchid_user[userid], lib.tbatchid_item[itemid]) + 1
                        lib.tbatchid_user[userid] = tbatch_to_insert
                        lib.tbatchid_item[itemid] = tbatch_to_insert
    
                        lib.current_tbatches_user[tbatch_to_insert].append(userid)
                        lib.current_tbatches_item[tbatch_to_insert].append(itemid)
                        lib.current_tbatches_feature[tbatch_to_insert].append(feature)
                        lib.current_tbatches_interactionids[tbatch_to_insert].append(j)
                        lib.current_tbatches_user_timediffs[tbatch_to_insert].append(user_timediff)
                        lib.current_tbatches_item_timediffs[tbatch_to_insert].append(item_timediff)
                        lib.current_tbatches_previous_item[tbatch_to_insert].append(user_previous_itemid_sequence[j])
    
                    timestamp = timestamp_sequence[j]
                    if tbatch_start_time is None:
                        tbatch_start_time = timestamp
    
                    # if timespan exceeded, process tbatch
                    if timestamp - tbatch_start_time > tbatch_timespan:
                        tbatch_start_time = timestamp
    
                        # load cached tbatches in later epochs
                        if not is_first_epoch:
                            lib.current_tbatches_user = cached_tbatches_user[timestamp]
                            lib.current_tbatches_item = cached_tbatches_item[timestamp]
                            lib.current_tbatches_interactionids = cached_tbatches_interactionids[timestamp]
                            lib.current_tbatches_feature = cached_tbatches_feature[timestamp]
                            lib.current_tbatches_user_timediffs = cached_tbatches_user_timediffs[timestamp]
                            lib.current_tbatches_item_timediffs = cached_tbatches_item_timediffs[timestamp]
                            lib.current_tbatches_previous_item = cached_tbatches_previous_item[timestamp]

                        with trange(len(lib.current_tbatches_user)) as progress_bar3:
                            for i in progress_bar3:
                                progress_bar3.set_description('Processed %d of %d T-batches ' % (i, len(lib.current_tbatches_user)))
                                
                                total_interaction_count += len(lib.current_tbatches_interactionids[i])
    
                                # LOAD THE CURRENT TBATCH
                                if is_first_epoch:
                                    lib.current_tbatches_user[i] = torch.LongTensor(lib.current_tbatches_user[i]).cuda()
                                    lib.current_tbatches_item[i] = torch.LongTensor(lib.current_tbatches_item[i]).cuda()
                                    lib.current_tbatches_interactionids[i] = torch.LongTensor(lib.current_tbatches_interactionids[i]).cuda()
                                    lib.current_tbatches_feature[i] = torch.Tensor(lib.current_tbatches_feature[i]).cuda()
    
                                    lib.current_tbatches_user_timediffs[i] = torch.Tensor(lib.current_tbatches_user_timediffs[i]).cuda()
                                    lib.current_tbatches_item_timediffs[i] = torch.Tensor(lib.current_tbatches_item_timediffs[i]).cuda()
                                    lib.current_tbatches_previous_item[i] = torch.LongTensor(lib.current_tbatches_previous_item[i]).cuda()
    
                                tbatch_userids = lib.current_tbatches_user[i] # Recall "lib.current_tbatches_user[i]" has unique elements
                                tbatch_itemids = lib.current_tbatches_item[i] # Recall "lib.current_tbatches_item[i]" has unique elements
                                tbatch_interactionids = lib.current_tbatches_interactionids[i]
                                feature_tensor = Variable(lib.current_tbatches_feature[i]) # Recall "lib.current_tbatches_feature[i]" is list of list, so "feature_tensor" is a 2-d tensor
                                user_timediffs_tensor = Variable(lib.current_tbatches_user_timediffs[i]).unsqueeze(1)
                                item_timediffs_tensor = Variable(lib.current_tbatches_item_timediffs[i]).unsqueeze(1)
                                tbatch_itemids_previous = lib.current_tbatches_previous_item[i]
                                item_embedding_previous = item_embeddings[tbatch_itemids_previous,:]
    
                                # PROJECT USER EMBEDDING TO CURRENT TIME
                                user_embedding_input = user_embeddings[tbatch_userids,:]
                                user_projected_embedding = model.forward(user_embedding_input, item_embedding_previous, timediffs=user_timediffs_tensor, features=feature_tensor, select='project')
                                user_item_embedding = torch.cat([user_projected_embedding, item_embedding_previous, item_embedding_static[tbatch_itemids_previous,:], user_embedding_static[tbatch_userids,:]], dim=1)
    
                                # PREDICT NEXT ITEM EMBEDDING                            
                                predicted_item_embedding = model.predict_item_embedding(user_item_embedding)
    
                                # CALCULATE PREDICTION LOSS
                                item_embedding_input = item_embeddings[tbatch_itemids,:]
                                loss += MSELoss(predicted_item_embedding, torch.cat([item_embedding_input, item_embedding_static[tbatch_itemids,:]], dim=1).detach())
    
                                # UPDATE DYNAMIC EMBEDDINGS AFTER INTERACTION
                                user_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=user_timediffs_tensor, features=feature_tensor, select='user_update')
                                item_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=item_timediffs_tensor, features=feature_tensor, select='item_update')
    
                                item_embeddings[tbatch_itemids,:] = item_embedding_output
                                user_embeddings[tbatch_userids,:] = user_embedding_output  
    
                                user_embeddings_timeseries[tbatch_interactionids,:] = user_embedding_output
                                item_embeddings_timeseries[tbatch_interactionids,:] = item_embedding_output
    
                                # CALCULATE LOSS TO MAINTAIN TEMPORAL SMOOTHNESS
                                loss += MSELoss(item_embedding_output, item_embedding_input.detach())
                                loss += MSELoss(user_embedding_output, user_embedding_input.detach())
    
                                # CALCULATE STATE CHANGE LOSS
                                if args.state_change:
                                    loss += calculate_state_prediction_loss(model, tbatch_interactionids, user_embeddings_timeseries, y_true, crossEntropyLoss) 
    
    
                        # backprop after tbatch
                        total_loss += loss.item()
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
    
                        # detach to save memory
                        loss = 0
                        item_embeddings.detach_()
                        user_embeddings.detach_()
                        item_embeddings_timeseries.detach_()
                        user_embeddings_timeseries.detach_()
    
                        if is_first_epoch:
                            cached_tbatches_user[timestamp] = lib.current_tbatches_user
                            cached_tbatches_item[timestamp] = lib.current_tbatches_item
                            cached_tbatches_interactionids[timestamp] = lib.current_tbatches_interactionids
                            cached_tbatches_feature[timestamp] = lib.current_tbatches_feature
                            cached_tbatches_user_timediffs[timestamp] = lib.current_tbatches_user_timediffs
                            cached_tbatches_item_timediffs[timestamp] = lib.current_tbatches_item_timediffs
                            cached_tbatches_previous_item[timestamp] = lib.current_tbatches_previous_item
    
                            reinitialize_tbatches()
                            tbatch_to_insert = -1
    
            is_first_epoch = False
            #print(f"Epoch {ep+1} finished | Total loss = {total_loss:.4f}")
    
            # ------------------------------
            # VALIDATION + EARLY STOPPING
            # ------------------------------
            val_ap, val_auc, val_mrr = evaluate_on_validation(
                model, user_embeddings, item_embeddings,
                user_sequence_id, item_sequence_id,
                timestamp_sequence, feature_sequence,
                user_timediffs_sequence, item_timediffs_sequence,
                user_previous_itemid_sequence, validation_start_idx, test_start_idx, item_embedding_static, user_embedding_static
            )
            #print(f"Validation: AP={val_ap:.4f} | AUC={val_auc:.4f} | MRR={val_mrr:.4f}")
    
            if val_ap > best_val_ap:
                best_val_ap = val_ap
                patience_counter = 0
                #print(f"--- New best model found (AP={val_ap:.4f}), saving... ---")
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                #print(f"No improvement. Patience = {patience_counter}/{max_patience}")
                if patience_counter >= max_patience:
                    #print("Early stopping triggered.")
                    break
    
            epoch_times.append(time.time() - epoch_start_time)
    
    result = {
        "ap": val_ap,
        "auc": val_auc,
        "mrr": val_mrr,
        "avg_epoch_time": np.mean(epoch_times),
        "peak_memory_mb": torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
    }
    return result

def run_multiple_experiments(args, seeds=[42, 43, 44, 45, 46]):
    all_results = []
    for seed in seeds:
        print(f"\n===== Running experiment with seed {seed} =====")
        result = run_single_experiment(seed, args)
        all_results.append(result)

    test_aps = [r["ap"] for r in all_results]
    test_aucs = [r["auc"] for r in all_results]
    test_mrrs = [r["mrr"] for r in all_results]
    epoch_times = [r["avg_epoch_time"] for r in all_results]
    peak_mems = [r["peak_memory_mb"] for r in all_results]

    mean_ap, std_ap = np.mean(test_aps), np.std(test_aps)
    mean_auc, std_auc = np.mean(test_aucs), np.std(test_aucs)
    mean_mrr, std_mrr = np.mean(test_mrrs), np.std(test_mrrs)
    mean_time, std_time = np.mean(epoch_times), np.std(epoch_times)
    mean_mem, std_mem = np.mean(peak_mems), np.std(peak_mems)

    print("\n--- Final Performance Results (Mean ± Std Dev) ---")
    print(f"Test AP:  {mean_ap:.4f} ± {std_ap:.4f}")
    print(f"Test AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Test MRR: {mean_mrr:.4f} ± {std_mrr:.4f}")
    print("-" * 60)
    print("\n--- Final Efficiency Results (Mean ± Std Dev) ---")
    print(f"Avg. Runtime/Epoch: {mean_time:.2f}s ± {std_time:.2f}s")
    if torch.cuda.is_available() and mean_mem > 0:
        print(f"Peak Memory Usage:  {mean_mem:.2f} MB ± {std_mem:.2f} MB")
    else:
        print("Peak Memory Usage:  N/A (Not a CUDA device)")


if __name__ == "__main__":
    seeds = [123451, 123452, 123453, 123454, 123455]
    run_multiple_experiments(args, seeds=seeds)

