import torch
import torch.nn.functional as F


def mf_loss_fun(pred_mf, target_mf):
    loss = F.l1_loss(pred_mf.clone(), target_mf.clone(), reduction='sum')
    return loss


def KL_loss(current_dist, target_dist):
    current_dist = current_dist.view(1, -1)/torch.sum(current_dist.view(1, -1))
    target_dist = target_dist.view(1, -1)/torch.sum(target_dist.view(1, -1))

    ratio_dist = current_dist/target_dist
    ratio_dist[torch.isnan(ratio_dist)] = 0.0

    kl_div_vec = current_dist * torch.log(current_dist) - current_dist * torch.log(target_dist)
    kl_div_vec[torch.isnan(kl_div_vec)] = 0.0
    return torch.sum(kl_div_vec)


def sparsity_loss_tanh(pop_activations):
    #return torch.sum(torch.tanh(pop_activations))
    return torch.tanh(pop_activations)


# this should generate a sharp sigmoid
def sparsity_loss_gen_sigmoid(pop_activations):
    sig_response = 1/(1 + 25 * torch.exp(-10*(pop_activations-0.0)))
    return torch.sum(sig_response)
    #return sig_response

# this should generate a sharp sigmoid
def sparsity_loss_gen_sigmoid_zero(pop_activations):
    sig_response = 1/(1 + 10 * torch.exp(-10*(pop_activations-0.00)))
    #return torch.sum(sig_response)
    return sig_response

# this should generate a sigmoid org
def sparsity_loss_gen_sigmoid_org(pop_activations):
    sig_response = 1/((1 + torch.exp(-5*(pop_activations-1))) ** (1))
    #return torch.sum(sig_response)
    return sig_response

def continuation_loss_fun(pop_activations):
    pop_activations = torch.squeeze(pop_activations)
    act_tplus1 = pop_activations[1:pop_activations.size(0), :]
    act_t = pop_activations[0:pop_activations.size(0)-1, :]
    return F.mse_loss(act_tplus1, act_t, reduction='sum')


def squared_tanh_activation(input):
    return torch.tanh(input)**2.0 # * (input**2)


def squared_activation(input):
    return input**2.0


if __name__=="__main__":
    import matplotlib.pyplot as plt
    input = torch.arange(0, 1, 0.01)
    # output = sparsity_loss_gen_sigmoid(input)
    # plt.plot(input.numpy(), output.numpy(), 'r.')

    # output_zero = sparsity_loss_gen_sigmoid_zero(input)
    # plt.plot(input.numpy(), output_zero.numpy(), 'r*')

    output_tanh = sparsity_loss_tanh(input)
    plt.plot(input.numpy(), output_tanh.numpy(), 'b.')

    output_sq_tanh = squared_tanh_activation(input)
    plt.plot(input.numpy(), output_sq_tanh.numpy(), 'r.')

    output_sq = squared_activation(input)
    plt.plot(input.numpy(), output_sq.numpy(), 'g.')

    # output_tanh_org = sparsity_loss_gen_sigmoid_org(input)
    # plt.plot(input.numpy(), output_tanh_org.numpy(), 'g.')

    plt.show()
