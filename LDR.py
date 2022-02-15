import numpy as np
import torch
import pdb
from sklearn.metrics import roc_auc_score
np.random.seed(2020)
torch.manual_seed(2020)
import pdb
from dataset import load_data
from Model import NCF, LDR_MCF,LDR_NCF
import arguments
from utils import gini_index, ndcg_func, get_user_wise_ctr, rating_mat_to_sample, binarize, shuffle, minU
mse_func = lambda x,y: np.mean((x-y)**2)
acc_func = lambda x,y: np.sum(x == y) / len(x)

dataset_name = "yahoo"
if dataset_name == "coat":
    biased_mat, unbiased_mat = load_data("coat")
    x_biased, y_biased = rating_mat_to_sample(biased_mat)
    x_test, y_test = rating_mat_to_sample(unbiased_mat)
    num_user = biased_mat.shape[0]
    num_item = unbiased_mat.shape[1]
elif dataset_name == "yahoo":
    x_biased, y_biased, x_test, y_test = load_data("yahoo")
    x_biased, y_unbiased = shuffle(x_biased, y_biased)
    num_user = x_biased[:,0].max() + 1
    num_item = x_biased[:,1].max() + 1

print("# user: {}, # item: {}".format(num_user, num_item))
# binarize
y_biased = binarize(y_biased)
y_test = binarize(y_test)
# print (len(x_biased))
# print (len(y_unbiased))
all_biased_id = np.arange(len(y_biased))
np.random.shuffle(all_biased_id)
y_validation = y_biased[all_biased_id[:int(0.05 * len(all_biased_id))]]
x_validation=x_biased[all_biased_id[:int(0.05 * len(all_biased_id))]]
y_train = y_biased[all_biased_id[int(0.05 * len(all_biased_id)):]]
x_train=x_biased[all_biased_id[int(0.05 * len(all_biased_id)):]]

args = arguments.parse_args()
learning_rateL = [1e-3,5e-3, 1e-2]
args.learning_rate = learning_rateL[args.learning_rate]
args.gamma = args.gamma/10
args.alpha=args.alpha/10

args.emb_dim = 2 ** args.emb_dim * 16
if args.model == 'MCF':
    LDR = LDR_MCF(num_user, num_item, embedding_k=args.emb_dim)
    modelName = 'LDR_MCF'
    print('model is %s'%modelName)
else:
    LDR = LDR_NCF(num_user, num_item, embedding_k=args.emb_dim)
    modelName = 'LDR_NCF'
    print('model is %s'%modelName)


LDR.fit(x_train, y_train,lr=args.learning_rate,
    alpha=args.alpha, gamma=args.gamma, lamb=1e-4, tol=1e-6,
    batch_size = 2048, verbose=1)
test_pred = LDR.predict(x_test)
ndcg_res = ndcg_func(LDR, x_test, y_test)
print("***"*5 + "[LDR]" + "***"*5)
print("ndcg@5:{:.6f}, ndcg@10:{:.6f}".format(
    np.mean(ndcg_res["ndcg_5"]), np.mean(ndcg_res["ndcg_10"])))
user_wise_ctr = get_user_wise_ctr(x_test,y_test,test_pred)
gi,gu = gini_index(user_wise_ctr)
print("***"*5 + "[LDR]" + "***"*5)

#search and save via validation
# checkpt_file = 'result/explicit_%s_AUC%.4f_' % (
#     args.dataset, auc_ncf) + 'mse%.4f_' % mse_ncf + "ndcg@5:{:.6f}, ndcg@10:{:.6f}".format(
#     np.mean(ndcg_res["ndcg_5"]), np.mean(ndcg_res["ndcg_10"])) \
#                + '_lamda1%s_lamda2%s_emb_dim%s_learning_rate%s_model%s' % (
#                    args.lamda1, args.lamda2, args.emb_dim, args.learning_rate, modelName) + '.csv'
# np.savetxt(checkpt_file, np.zeros((2)), delimiter=',')

