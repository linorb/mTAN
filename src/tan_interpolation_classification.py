"""This script first train the encoder by optimizing the interpolation problem (in unsupervised manner)
then, it trains the classifier with the trained encoder."""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

from random import SystemRandom
import models
import utils

def get_parsed_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--niters', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--std', type=float, default=0.01)
    parser.add_argument('--latent-dim', type=int, default=32)
    parser.add_argument('--rec-hidden', type=int, default=32)
    parser.add_argument('--gen-hidden', type=int, default=50)
    parser.add_argument('--embed-time', type=int, default=128)
    parser.add_argument('--k-iwae', type=int, default=10)
    parser.add_argument('--save', type=int, default=1)
    parser.add_argument('--enc', type=str, default='mtan_rnn')
    parser.add_argument('--dec', type=str, default='mtan_rnn')
    parser.add_argument('--fname', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n', type=int, default=8000)
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--quantization', type=float, default=0.016,
                        help="Quantization on the physionet dataset.")
    parser.add_argument('--classif', action='store_true',
                        help="Include binary classification loss")
    parser.add_argument('--norm', action='store_true')
    parser.add_argument('--kl', action='store_true')
    parser.add_argument('--learn-emb', action='store_true')
    parser.add_argument('--enc-num-heads', type=int, default=1)
    parser.add_argument('--dec-num-heads', type=int, default=1)
    parser.add_argument('--length', type=int, default=20)
    parser.add_argument('--num-ref-points', type=int, default=128)
    parser.add_argument('--dataset', type=str, default='toy')
    parser.add_argument('--enc-rnn', action='store_false')
    parser.add_argument('--dec-rnn', action='store_false')
    parser.add_argument('--sample-tp', type=float, default=1.0)
    parser.add_argument('--only-periodic', type=str, default=None)
    parser.add_argument('--dropout', type=float, default=0.0)
    args = parser.parse_args()

    return args

def train_encoder_by_interpolation(args, experiment_id):
    # Set the device to GPU
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Load the data - Change this section according to the data you want to train the model.
    # The data is Tensor with dims: [n_samples, n_timestamps, 3]
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    if args.dataset == 'toy':
        data_obj = utils.kernel_smoother_data_gen(args, alpha=100., seed=0)
    elif args.dataset == 'physionet':
        data_obj = utils.get_physionet_data(args, 'cpu', args.quantization)

    train_loader = data_obj["train_dataloader"]
    test_loader = data_obj["test_dataloader"]
    dim = data_obj["input_dim"]

    # Set the model for the encoder and the decoder.
    # mtan_rnn is the multi-time-attention RNN
    # Set model for encoder (rec)
    if args.enc == 'enc_rnn3':
        rec = models.enc_rnn3(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim,
            args.rec_hidden, 128, learn_emb=args.learn_emb).to(device)
    elif args.enc == 'mtan_rnn':
        rec = models.enc_mtan_rnn(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, args.rec_hidden,
            embed_time=128, learn_emb=args.learn_emb, num_heads=args.enc_num_heads).to(device)

    # Set model for decoder (dec)
    if args.dec == 'rnn3':
        dec = models.dec_rnn3(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim,
            args.gen_hidden, 128, learn_emb=args.learn_emb).to(device)
    elif args.dec == 'mtan_rnn':
        dec = models.dec_mtan_rnn(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, args.gen_hidden,
            embed_time=128, learn_emb=args.learn_emb, num_heads=args.dec_num_heads).to(device)

    params = (list(dec.parameters()) + list(rec.parameters()))
    optimizer = optim.Adam(params, lr=args.lr)
    print('parameters:', utils.count_parameters(rec), utils.count_parameters(dec))

    # Print test evaluations - for debug
    if args.fname is not None:
        checkpoint = torch.load(args.fname)
        rec.load_state_dict(checkpoint['rec_state_dict'])
        dec.load_state_dict(checkpoint['dec_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('loading saved weights', checkpoint['epoch'])
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 1))
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 3))
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 10))
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 20))
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 30))
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 50))

    # Train the model
    for itr in range(1, args.niters + 1):
        train_loss = 0
        train_n = 0
        avg_reconst, avg_kl, mse = 0, 0, 0
        if args.kl:
            wait_until_kl_inc = 10
            if itr < wait_until_kl_inc:
                kl_coef = 0.
            else:
                kl_coef = (1 - 0.99 ** (itr - wait_until_kl_inc))
        else:
            kl_coef = 1

        for train_batch in train_loader:
            train_batch = train_batch.to(device)
            batch_len = train_batch.shape[0]
            observed_data = train_batch[:, :, :dim]
            observed_mask = train_batch[:, :, dim:2 * dim]
            observed_tp = train_batch[:, :, -1]

            # Subsample data for training
            if args.sample_tp and args.sample_tp < 1:
                subsampled_data, subsampled_tp, subsampled_mask = utils.subsample_timepoints(
                    observed_data.clone(), observed_tp.clone(), observed_mask.clone(), args.sample_tp)
            else:
                subsampled_data, subsampled_tp, subsampled_mask = \
                    observed_data, observed_tp, observed_mask

            # Apply encoder on subsampled data to extract the latent space
            out = rec(torch.cat((subsampled_data, subsampled_mask), 2), subsampled_tp)
            qz0_mean = out[:, :, :args.latent_dim]
            qz0_logvar = out[:, :, args.latent_dim:]
            # epsilon = torch.randn(qz0_mean.size()).to(device)
            epsilon = torch.randn(
                args.k_iwae, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]
            ).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
            z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])

            # Apply decoder on the latent space
            pred_x = dec(
                z0,
                observed_tp[None, :, :].repeat(args.k_iwae, 1, 1).view(-1, observed_tp.shape[1])
            )
            # nsample, batch, seqlen, dim
            pred_x = pred_x.view(args.k_iwae, batch_len, pred_x.shape[1], pred_x.shape[2])

            # compute loss
            logpx, analytic_kl = utils.compute_losses(
                dim, train_batch, qz0_mean, qz0_logvar, pred_x, args, device)
            loss = -(torch.logsumexp(logpx - kl_coef * analytic_kl, dim=0).mean(0) - np.log(args.k_iwae))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_len
            train_n += batch_len
            avg_reconst += torch.mean(logpx) * batch_len
            avg_kl += torch.mean(analytic_kl) * batch_len
            mse += utils.mean_squared_error(
                observed_data, pred_x.mean(0), observed_mask) * batch_len

        print('Iter: {}, avg elbo: {:.4f}, avg reconst: {:.4f}, avg kl: {:.4f}, mse: {:.6f}'
            .format(itr, train_loss / train_n, -avg_reconst / train_n, avg_kl / train_n, mse / train_n))
        if itr % 10 == 0:
            print('Test Mean Squared Error', utils.evaluate(dim, rec, dec, test_loader, args, 1))
        if itr % 10 == 0 and args.save:
            trained_model_filename = args.dataset + '_' + args.enc + '_' + args.dec + '_' + \
                                     str(experiment_id) + '.h5'
            torch.save({
                'args': args,
                'epoch': itr,
                'rec_state_dict': rec.state_dict(),
                'dec_state_dict': dec.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': -loss,
            }, trained_model_filename)

    return trained_model_filename

def train_full_with_classifier(trained_model_filename, experiment_id, new_args):
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Load the checkpoint file
    checkpoint = torch.load(trained_model_filename,
                            map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Extract stored values
    args = checkpoint['args']
    epoch = checkpoint['epoch']
    rec_state_dict = checkpoint['rec_state_dict']
    dec_state_dict = checkpoint['dec_state_dict']
    print(f"Loaded model from epoch {epoch}")

    # Add for classification because the args were taken from interpolation task
    args.classif = True
    args.classify_pertp = False
    args.alpha = new_args['alpha']
    args.lr = new_args['lr']
    args.niters = new_args['niters']

    # Load the data - change this part if you want to use different data to train the model
    if args.dataset == 'physionet':
        data_obj = utils.get_physionet_data(args, 'cpu', args.quantization)
    elif args.dataset == 'mimiciii':
        data_obj = utils.get_mimiciii_data(args)

    train_loader = data_obj["train_dataloader"]
    test_loader = data_obj["test_dataloader"]
    val_loader = data_obj["val_dataloader"]
    dim = data_obj["input_dim"]

    # Reconstruct the trained model, we define the architecture of rec (the encoder) before loading the state dictionary
    rec = models.enc_mtan_rnn(
        dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, args.rec_hidden,
        embed_time=128, learn_emb=args.learn_emb, num_heads=args.enc_num_heads)
    rec.load_state_dict(rec_state_dict)
    rec.train()
    rec.to(device)

    dec = models.dec_mtan_rnn(
        dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, args.gen_hidden,
        embed_time=128, learn_emb=args.learn_emb, num_heads=args.dec_num_heads)
    dec.load_state_dict(dec_state_dict)
    dec.train()
    dec.to(device)

    # Set the classifier - If you want to use a different classifier change here the architecture
    classifier = models.create_classifier(args.latent_dim, args.rec_hidden).to(device)

    params = (list(rec.parameters()) + list(dec.parameters()) + list(classifier.parameters()))
    print('parameters:', utils.count_parameters(rec), utils.count_parameters(dec), utils.count_parameters(classifier))
    optimizer = optim.Adam(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    if args.fname is not None:
        checkpoint = torch.load(args.fname)
        rec.load_state_dict(checkpoint['rec_state_dict'])
        dec.load_state_dict(checkpoint['dec_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('loading saved weights', checkpoint['epoch'])

    best_val_loss = float('inf')
    total_time = 0.
    for itr in range(1, args.niters + 1):
        train_recon_loss, train_ce_loss = 0, 0
        mse = 0
        train_n = 0
        train_acc = 0
        # avg_reconst, avg_kl, mse = 0, 0, 0
        if args.kl:
            wait_until_kl_inc = 10
            if itr < wait_until_kl_inc:
                kl_coef = 0.
            else:
                kl_coef = (1 - 0.99 ** (itr - wait_until_kl_inc))
        else:
            kl_coef = 1
        start_time = time.time()
        for train_batch, label in train_loader:
            train_batch, label = train_batch.to(device), label.to(device)
            batch_len = train_batch.shape[0]
            observed_data, observed_mask, observed_tp \
                = train_batch[:, :, :dim], train_batch[:, :, dim:2 * dim], train_batch[:, :, -1]

            # Apply encoder on observed_data to extract the latent space
            out = rec(torch.cat((observed_data, observed_mask), 2), observed_tp)
            qz0_mean, qz0_logvar = out[:, :, :args.latent_dim], out[:, :, args.latent_dim:]
            epsilon = torch.randn(args.k_iwae, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
            z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])

            # Apply the classifier on the latent space
            pred_y = classifier(z0)

            # Apply the decoder on the latent space
            pred_x = dec(
                z0, observed_tp[None, :, :].repeat(args.k_iwae, 1, 1).view(-1, observed_tp.shape[1]))
            pred_x = pred_x.view(args.k_iwae, batch_len, pred_x.shape[1],
                                 pred_x.shape[2])  # nsample, batch, seqlen, dim

            # Compute decoder loss for the reconstruction
            logpx, analytic_kl = utils.compute_losses(
                dim, train_batch, qz0_mean, qz0_logvar, pred_x, args, device)
            recon_loss = -(torch.logsumexp(logpx - kl_coef * analytic_kl, dim=0).mean(0) - np.log(args.k_iwae))

            # Compute the classification loss for the label
            label = label.unsqueeze(0).repeat_interleave(args.k_iwae, 0).view(-1)
            ce_loss = criterion(pred_y, label)

            # Compute the total loss as a combination of decoder and classification loss
            loss = recon_loss + args.alpha * ce_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_ce_loss += ce_loss.item() * batch_len
            train_recon_loss += recon_loss.item() * batch_len
            train_acc += (pred_y.argmax(1) == label).sum().item() / args.k_iwae
            train_n += batch_len
            mse += utils.mean_squared_error(observed_data, pred_x.mean(0),
                                            observed_mask) * batch_len
        total_time += time.time() - start_time
        val_loss, val_acc, val_auc = utils.evaluate_classifier(
            rec, val_loader, args=args, classifier=classifier, reconst=True, num_sample=1, dim=dim)
        if val_loss <= best_val_loss:
            best_val_loss = min(best_val_loss, val_loss)
            rec_state_dict = rec.state_dict()
            dec_state_dict = dec.state_dict()
            classifier_state_dict = classifier.state_dict()
            optimizer_state_dict = optimizer.state_dict()
        test_loss, test_acc, test_auc = utils.evaluate_classifier(
            rec, test_loader, args=args, classifier=classifier, reconst=True, num_sample=1, dim=dim)
        print(
            'Iter: {}, recon_loss: {:.4f}, ce_loss: {:.4f}, acc: {:.4f}, mse: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, test_acc: {:.4f}, test_auc: {:.4f}'
            .format(itr, train_recon_loss / train_n, train_ce_loss / train_n,
                    train_acc / train_n, mse / train_n, val_loss, val_acc, test_acc, test_auc))

        if itr % 100 == 0 and args.save:
            trained_model_filename = args.dataset + '_' + args.enc + '_' + args.dec + '_' + str(experiment_id) + '.h5'
            torch.save({
                'args': args,
                'epoch': itr,
                'rec_state_dict': rec_state_dict,
                'dec_state_dict': dec_state_dict,
                'optimizer_state_dict': optimizer_state_dict,
                'classifier_state_dict': classifier_state_dict,
                'loss': -loss,
            }, trained_model_filename)
    print(best_val_loss)
    print(total_time)

    return trained_model_filename


def main():
    args = get_parsed_arguments()
    experiment_id = int(SystemRandom().random() * 100000)
    print(args, experiment_id)

    trained_model_filename = train_encoder_by_interpolation(args, experiment_id)

    new_args = {'lr': 0.00001, 'alpha': 100, 'niters': 10}
    full_model_filename = train_full_with_classifier(trained_model_filename, experiment_id, new_args)

if __name__ == '__main__':
    trained_model_filename = r"C:\Users\user\Documents\dev\mTAN\src\physionet_mtan_rnn_mtan_rnn_49317.h5"
    experiment_id = 49317
    new_args = {'lr': 0.00001, 'alpha': 100, 'niters': 10}
    train_full_with_classifier(trained_model_filename, experiment_id, new_args)