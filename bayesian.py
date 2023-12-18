import torch
import torch.nn as nn
import scipy.stats as stats
import numpy as np
import plots
import os
from displays import get_experiment, ChangeDetection
from cnn import get_cnn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
def get_likelihood(embedding, params) -> stats.rv_continuous:
    dim = embedding.shape[0]
    cov = stats.Covariance.from_diagonal(np.ones(dim) * params['sigma'] ** 2)
    rv = stats.multivariate_normal(embedding, cov)
    return rv

def get_embedding(cnn, imgs) -> np.ndarray:
    if len(imgs.shape) == 3:
        imgs = imgs.unsqueeze(0)
    embedding = cnn(imgs.to(device))
    if embedding.shape[0] == 1:
        embedding = embedding.squeeze(0)
    return embedding

gaussian_samples = np.random.randn(5000, 5000)

def get_predictions(
    params: dict, 
    embeddings: list,
    sampling_points: int = 1000, 
):

    # get predictions
    predictions = []
    n = sampling_points
    m = embeddings[0][0].shape[0]

    for i in range(len(embeddings)):
        sample_embedding, test_embedding, changed_embedding = embeddings[i]
        memory_samples = gaussian_samples[:n, :m] * params['sigma'] + sample_embedding

        likelihood_uc = get_likelihood(test_embedding, params) # p(S | D1 = D2)
        likelihood_c = [get_likelihood(changed_embedding[i], params) for i in range(changed_embedding.shape[0])] # p(S | D) for D not equal to D2        
        
        likelihood_uc_sample = likelihood_uc.pdf(memory_samples)
        likelihood_c_sample = [x.pdf(memory_samples) for x in likelihood_c]
        # print(likelihood_uc_sample, np.mean(likelihood_c_sample, axis=0))
        uc_p = 1 - params['change_prior']

        posterior_uc = uc_p * likelihood_uc_sample / (uc_p * likelihood_uc_sample + (1 - uc_p) * np.mean(likelihood_c_sample, axis=0))
        posterior_uc = np.mean(posterior_uc)
        predictions.append(posterior_uc)
    
    return predictions

def run(
    exp_name: str, 
    init_params: dict, 
    cnn_config: dict,
    optimize=True):

    # load experiment
    exp = get_experiment(exp_name)
    cnn = get_cnn(**cnn_config)
    cnn.to(device)

    behaviors = []
    embeddings = []
    dists = []

    for i in range(len(exp)):
        trial = exp.get_trial(i)
        features = trial['features']
        displays = trial['displays']
        behaviors.append(exp.behaviors[i])

        sample_embedding = get_embedding(cnn, displays[0]) # D1
        test_embedding = get_embedding(cnn, displays[1]) # D2
        changed_embedding = get_embedding(cnn, exp.feature_to_changed_displays(features[1]))
        embeddings.append([sample_embedding, test_embedding, changed_embedding])

        if trial['changed']:
            print(((sample_embedding - test_embedding) ** 2).sum() ** 0.5, (sample_embedding ** 2).sum() ** 0.5)
            dist = ((sample_embedding - test_embedding) ** 2).sum() ** 0.5
            # dist = torch.nn.functional.cosine_similarity(torch.from_numpy(sample_embedding), torch.from_numpy(test_embedding), dim=0)
            dists.append(dist)

    if optimize:
        def loss(param):
            params = {}
            params['sigma'] = param[0]
            params['change_prior'] = param[1]
            predictions = get_predictions(params, embeddings)
            loss = np.mean((np.array(predictions) - np.array(behaviors)) ** 2)
            return loss
        
        from scipy.optimize import minimize
        res = minimize(loss, [init_params['sigma'], init_params['change_prior']], method='L-BFGS-B', 
               options={'gtol': 1e-6, 'ftol': 1e-6}, bounds=[(0.05, 20), (0.1, 0.9)])
        assert res.success

        params = {}
        params['sigma'] = res.x[0]
        params['change_prior'] = res.x[1]
        print(res.x)
    else:
        params = init_params

    predictions = get_predictions(params, embeddings)

    save_path='figures/{}_{}'.format(exp_name, cnn_config['model_name'])
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    d_prime = plots.compare_behavior_vs_prediction(behaviors, predictions, save_path=save_path)
    plots.compare_behavior_vs_distance(behaviors[1::2], dists, save_path=save_path)

    print(list(zip(range(24), d_prime[0], d_prime[1])))
    return params

def check_set_size_effect(
    params,
    cnn_config: dict
):
    
    exp_name = 'ColoredSquares_SetSize'
    exp = get_experiment(exp_name)
    cnn = get_cnn(**cnn_config)
    cnn.to(device)

    embeddings = []
    dists = []

    for i in range(len(exp)):
        trial = exp.get_trial(i)
        features = trial['features']
        displays = trial['displays']

        sample_embedding = get_embedding(cnn, displays[0]) # D1
        test_embedding = get_embedding(cnn, displays[1]) # D2
        changed_embedding = get_embedding(cnn, exp.feature_to_changed_displays(features[1]))
        embeddings.append([sample_embedding, test_embedding, changed_embedding])

        if trial['changed']:
            print(((sample_embedding - test_embedding) ** 2).sum() ** 0.5, (sample_embedding ** 2).sum() ** 0.5)
            dist = ((sample_embedding - test_embedding) ** 2).sum() ** 0.5
            dists.append(dist)

    predictions = get_predictions(params, embeddings)
    save_path='figures/{}_{}_{}'.format(exp_name, int(params['sigma'] * 100), cnn_config['model_name'])
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # calculate accuracy for each set size
    # 100 trials for n = 1, 2, ..., 12
    hit_rates = []
    false_alarm_rates = []
    accs = []
    avg_dists = []
    set_sizes = [1, 2, 3, 4, 8, 12]
    for set_size in range(len(set_sizes)):
        pred = predictions[set_size * 400: (set_size + 1) * 400]
        change_pred = pred[1 :: 2]
        no_change_pred = pred[:: 2]
        accs.append((np.mean(no_change_pred) + (1 - np.mean(change_pred))) * 0.5)
        hit_rates.append(1 - np.mean(change_pred))
        false_alarm_rates.append(1 - np.mean(no_change_pred))
        dist = dists[set_size * 200: (set_size + 1) * 200]
        avg_dists.append(np.mean(dist))

    plots.compare_acc_vs_set_size(set_sizes, accs, 'Accuracy', save_path=save_path)
    plots.compare_acc_vs_set_size(set_sizes, hit_rates, 'Hit Rate', save_path=save_path)
    plots.compare_acc_vs_set_size(set_sizes, false_alarm_rates, 'False Alarm Rate', save_path=save_path)
    plots.compare_acc_vs_set_size(set_sizes, avg_dists, 'Distance', save_path=save_path)

if __name__ == '__main__':
    params = {
        'sigma': 0.2,
        'change_prior': 0.5
    }

    cnn_config = {
        'cnn_archi': 'ResNet-18',
        'cnn_pret': 'Classification_ImageNet',
        'pca_dim': 16,
        'cnn_layer': 'last'
    }

    for exp in ['RedBlue', 'BlackWhite', 'ColoredSquares']:
        for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
            cnn_config['cnn_layer'] = layer
            cnn_config['model_name'] = '{}_{}_{}_{}'.format(cnn_config['cnn_archi'], cnn_config['cnn_layer'], cnn_config['pca_dim'], cnn_config['cnn_pret'][:5])
            params = run(exp, params, cnn_config, optimize=False)
    exit(0)

    cnn_config['model_name'] = '{}_{}_{}_{}'.format(cnn_config['cnn_archi'], cnn_config['cnn_layer'], cnn_config['pca_dim'], cnn_config['cnn_pret'][:5])
    for sigma in [0.2, 0.5, 1]:
        params = {
            'sigma': sigma,
            'change_prior': 0.5
        }
        params = check_set_size_effect(params, cnn_config)