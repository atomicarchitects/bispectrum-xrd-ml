from imports import * 

def analyze_element_wise_loss(model, val_loader):
    """
    compute L1 loss and normalized loss for each element in the 10x13 output matrix and compute
    """
    model.cuda()
    model.eval()

    # initialize lists to store losses and values for each position
    all_losses = []
    all_values = []

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)

            # calculate absolute error for each element (batch_size, 10, 13)
            batch_losses = torch.abs(output - target)
            all_losses.append(batch_losses)
            all_values.append(torch.abs(target))  # store absolute values for normalization

    all_losses = torch.cat(all_losses, dim=0)  # (total_samples, 10, 13)
    all_values = torch.cat(all_values, dim=0)

    element_wise_losses = torch.mean(all_losses, dim=0)  # (10, 13)
    element_wise_values = torch.mean(all_values, dim=0)

    # normalized losses (L1 loss / average value)
    epsilon = 1e-8

    zero_mask = (torch.abs(element_wise_values) <= epsilon)
    normalized_losses = element_wise_losses / (element_wise_values + epsilon)

    return element_wise_losses, normalized_losses

def visualize_element_wise_loss(element_wise_losses, normalized_losses):
    """
    heatmaps of the element-wise losses and their normalized values
    """
    plt.figure(figsize=(15, 5))

    # plot absolute L1 losses
    plt.subplot(1, 2, 1)
    loss_max = max(abs(element_wise_losses.max()), abs(element_wise_losses.min()))
    im1 = plt.imshow(element_wise_losses.cpu().numpy(),
                     cmap='RdBu',
                     vmax=loss_max,
                     vmin=-loss_max)
    plt.yticks(np.linspace(0, element_wise_losses.shape[0] - 1, num=10))
    plt.xticks([])
    plt.tick_params(axis='y', labelsize=18)
    plt.ylabel('Radial Basis Function', fontsize=18)
    plt.colorbar(im1)
    plt.title('L1 Loss', fontsize=18)

    # plot normalized losses (L1 loss / average value)
    plt.subplot(1, 2, 2)
    norm_max = max(abs(normalized_losses.max()), abs(normalized_losses.min()))
    im2 = plt.imshow(normalized_losses.cpu().numpy(),
                     cmap='RdBu',
                     vmax=norm_max,
                     vmin=-norm_max)
    plt.yticks(np.linspace(0, normalized_losses.shape[0] - 1, num=10))
    plt.xticks([])
    plt.tick_params(axis='y', labelsize=18)
    plt.ylabel('Radial Basis Function', fontsize=18)
    plt.colorbar(im2)
    plt.title('L1 Loss/Average Value', fontsize=18)

    plt.tight_layout()
    plt.show()



def plot_comparison(val_structure, val_xrd, val_ground_truth, val_predictions, idx=0):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 4))

    key = list(val_structure.keys())[idx]
    xrd_struct = val_structure[key]
    mineral_name = xrd_struct.metadata['mineral_name'].iloc[0]
    fig.suptitle(f'Analysis for {mineral_name}', fontsize=16, fontweight='bold')

    two_theta = np.arange(5, 90, 0.010)

    ax1.plot(two_theta[:len(val_xrd[idx])], val_xrd[idx])
    ax1.set_title('XRD Pattern')
    ax1.set_xlabel('2θ (degrees)')
    ax1.set_ylabel('Intensity (a.u.)')

    gt = np.array(val_ground_truth[idx])
    pred = np.array(val_predictions[idx])
    
    common_vmax = max(np.abs(pred).max(), np.abs(gt).max())

    # plot the ground truth as a heatmap
    im_gt = ax2.imshow(gt, cmap='RdBu', vmax = common_vmax, vmin = -common_vmax)
    ax2.set_title('Ground Truth Bispectrum')
    plt.colorbar(im_gt, ax=ax2)

    im_pred = ax3.imshow(pred, cmap='RdBu', vmax = common_vmax, vmin = -common_vmax)
    ax3.set_title('Prediction Bispectrum')
    plt.colorbar(im_pred, ax=ax3)

    plt.tight_layout()
    plt.show()

    fig.savefig(f'plots/comparison/bispec_comparison_rruff_{idx}.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    with open('pickles/validation_dict_rruff_transformer_sixteen_32.pkl', 'rb') as f:
        val_dict = pickle.load(f)
    for i in range(10):
        plot_comparison(val_dict['structure'], val_dict['xrd_pattern'], val_dict['ground_truth'], val_dict['prediction'], i)
