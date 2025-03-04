import argparse
import numpy as np
import torch
import torch.optim as optim

from random import SystemRandom
import models
import utils
import matplotlib.pyplot as plt


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

def main():
    args = get_parsed_arguments()
    experiment_id = int(SystemRandom().random() * 100000)
    print(args, experiment_id)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    if args.dataset == 'toy':
        data_obj = utils.kernel_smoother_data_gen(args, alpha=100., seed=0)

    train_loader = data_obj["train_dataloader"]
    test_loader = data_obj["test_dataloader"]
    dim = data_obj["input_dim"]

    # Look at the toy data
    raw_data = data_obj['dataset_obj']
    example = raw_data[:, 0, :]
    plt.figure()
    plt.plot(example[:, 0])
    plt.plot(example[:, 1])
    plt.plot(example[:, 2])
    plt.show()

    print()


def plot_prediction(full_sample, full_timestamp, predicted_part, predicted_timestamp):
    for i in range(full_sample.shape[0]):
        # Convert tensors to numpy for plotting
        orig_np = full_sample[i, :, :].cpu().detach().numpy()  # (128, 1)
        pred_np = predicted_part[i, :, :].cpu().detach().numpy()  # (128, 1)
        orig_time = full_timestamp[i, :].cpu().detach().numpy()
        pred_time = predicted_timestamp[i, :].cpu().detach().numpy()

        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(orig_time, orig_np, label="Original", alpha=0.7)
        plt.plot(pred_time, pred_np, label="Predicted", linestyle="dashed", alpha=0.7)

        plt.legend()
        plt.xlabel("Samples")
        plt.ylabel("Value")
        plt.title(f"Comparison of Original vs Predicted for sample {i}")
        plt.show()


def evaluate_forward(dim, rec, dec, test_loader, args, num_sample=10, device="cuda", num_future_sample=2):
    mse, test_n = 0.0, 0.0
    with torch.no_grad():
        for test_batch in test_loader:
            test_batch = test_batch.to(device)
            observed_data, observed_mask, observed_tp = (
                test_batch[:, :-num_future_sample, :dim],
                test_batch[:, :-num_future_sample, dim: 2 * dim],
                test_batch[:, :-num_future_sample, -1],
            )
            future_data, future_mask, future_tp = (
                test_batch[:, -num_future_sample:, :dim],
                test_batch[:, -num_future_sample:, dim: 2 * dim],
                test_batch[:, -num_future_sample:, -1],
            )
            if args.sample_tp and args.sample_tp < 1:
                subsampled_data, subsampled_tp, subsampled_mask = utils.subsample_timepoints(
                    observed_data.clone(), observed_tp.clone(), observed_mask.clone(), args.sample_tp)
            else:
                subsampled_data, subsampled_tp, subsampled_mask = \
                    observed_data, observed_tp, observed_mask
            out = rec(torch.cat((subsampled_data, subsampled_mask), 2), subsampled_tp)
            qz0_mean, qz0_logvar = (
                out[:, :, : args.latent_dim],
                out[:, :, args.latent_dim:],
            )
            epsilon = torch.randn(
                num_sample, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]
            ).to(device)
            z0 = epsilon * torch.exp(0.5 * qz0_logvar) + qz0_mean
            z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
            batch, seqlen = future_tp.size()

            time_steps = future_tp[None, :, :].repeat(num_sample, 1, 1).view(-1, seqlen)

            pred_x = dec(z0, time_steps)
            pred_x = pred_x.view(num_sample, -1, pred_x.shape[1], pred_x.shape[2])
            pred_x = pred_x.mean(0)

            full_data, full_tp = (
                test_batch[:, :, :dim],
                test_batch[:, :, -1],
            )
            plot_prediction(full_data, full_tp, pred_x, time_steps)

            mse += utils.mean_squared_error(future_data, pred_x, future_mask) * batch
            test_n += batch

    return mse / test_n

def load_model():
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    filename = r"C:\Users\user\Documents\dev\mTAN\src\toy_mtan_rnn_mtan_rnn_5875.h5"

    # Load the checkpoint file
    checkpoint = torch.load(filename,
                            map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Extract stored values
    args = checkpoint['args']
    epoch = checkpoint['epoch']
    rec_state_dict = checkpoint['rec_state_dict']
    dec_state_dict = checkpoint['dec_state_dict']
    optimizer_state_dict = checkpoint['optimizer_state_dict']
    loss = checkpoint['loss']
    # Taken from the dim = data_obj["input_dim"]
    dim = 1
    print(f"Loaded model from epoch {epoch} with loss {loss}")

    # In order to reconstruct the model, we define the architecture of rec (the encoder) before loading the state dictionary
    rec = models.enc_mtan_rnn(
        dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, args.rec_hidden,
        embed_time=128, learn_emb=args.learn_emb, num_heads=args.enc_num_heads)
    rec.load_state_dict(rec_state_dict)
    rec.eval()
    rec.to(device)

    dec = models.dec_mtan_rnn(
        dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, args.gen_hidden,
        embed_time=128, learn_emb=args.learn_emb, num_heads=args.dec_num_heads)
    dec.load_state_dict(dec_state_dict)
    dec.eval()
    dec.to(device)

    # Create more toy data and test the model on it
    data_obj = utils.kernel_smoother_data_gen(args, alpha=100., seed=0)
    train_loader = data_obj["train_dataloader"]
    test_loader = data_obj["test_dataloader"]
    dim = data_obj["input_dim"]

    # evaluate the model
    # print('Test MSE', evaluate_forward(dim, rec, dec, test_loader, args, 1))
    # print('Test MSE', evaluate_forward(dim, rec, dec, test_loader, args, 3))
    print('Test MSE', evaluate_forward(dim, rec, dec, test_loader, args, 10))
    # print('Test MSE', evaluate_forward(dim, rec, dec, test_loader, args, 20))
    # print('Test MSE', evaluate_forward(dim, rec, dec, test_loader, args, 30))
    # print('Test MSE', evaluate_forward(dim, rec, dec, test_loader, args, 50))


if __name__ == '__main__':
    load_model()