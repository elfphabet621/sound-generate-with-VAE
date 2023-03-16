import torch

def calculate_combined_loss(y_target, y_predicted, mu, log_variance, reconstruction_loss_weight):
    reconstruction_loss = calculate_reconstruction_loss(y_target, y_predicted)
    kl_loss = calculate_kl_loss(mu, log_variance)
    combined_loss = reconstruction_loss_weight * reconstruction_loss\
                                                        + kl_loss
    
    return combined_loss

def calculate_reconstruction_loss( y_target, y_predicted):
    error = y_target - y_predicted
    reconstruction_loss = torch.mean(torch.square(error))
    return reconstruction_loss

def calculate_kl_loss(mu, log_variance):
    kl_loss = -0.5 * torch.sum(1 + log_variance - torch.square(mu) -
                            torch.exp(log_variance))
    return kl_loss