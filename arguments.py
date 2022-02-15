import argparse

# arguments setting
def parse_args(): 
    parser = argparse.ArgumentParser(description='learning framework for RS')
    parser.add_argument('--dataset', type=str, default='yahooR3', help='Choose from {yahooR3, coat, simulation}')
    # parser.add_argument('--base_model_args', type=dict, default={'emb_dim': 10, 'learning_rate': 0.01, 'imputaion_lambda': 0.01, 'weight_decay': 1},
    #             help='base model arguments.')
    # parser.add_argument('--weight1_model_args', type=dict, default={'learning_rate': 0.1, 'weight_decay': 0.001},
    #             help='weight model arguments.')
    # parser.add_argument('--weight2_model_args', type=dict, default={'learning_rate': 1e-3, 'weight_decay': 1e-2},
    #             help='imputation model arguments.')
    # parser.add_argument('--imputation_model_args', type=dict, default= {'learning_rate': 1e-1, 'weight_decay': 1e-4},
    #             help='imputation model arguments.')
    parser.add_argument('--training_args', type=dict, default = {'batch_size': 1024, 'epochs': 500, 'patience': 60, 'block_batch': [20, 500]}, 
                help='training arguments.')
    parser.add_argument('--uniform_ratio', type=float, default=0.05, help='the ratio of uniform set in the unbiased dataset.')
    parser.add_argument('--seed', type=int, default=0, help='global general random seed.')
    parser.add_argument('--gamma', type=int, default=1 ) # lamda1/10
    parser.add_argument('--alpha', type=int, default=2)  # lamda2/10
    parser.add_argument('--emb_dim', type=int, default=1 ) # 16* 2^n 16-256 0-4
    parser.add_argument('--learning_rate', type=int, default=1) # 0-4 [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    parser.add_argument('--model', type=str, default='NCF', help='Choose from {NCF, MF}')
    parser.add_argument('--D_C', type=str, default='C', help='Choose from {NCF, MF}')

    return parser.parse_args()
