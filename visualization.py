import json
import matplotlib.pyplot as plt
import os

def read_data(log_path):
    stats = []
    with open(log_path, "r") as file:
        for line in file:
            line = line.strip()
            if line:
                stats.append(json.loads(line))
    
    loss = [entry['test_loss'] for entry in stats]
    max_gpu_mem = [entry['max_mem_gpu'] for entry in stats]
    epoch = [entry['epoch'] for entry in stats]
    
    return epoch, loss, max_gpu_mem


def visualization(log_path):
    x, loss, mem = read_data(log_path)
    
    plt.subplot(1, 2, 1)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(x, loss, 'r-')


    path = os.path.join(dir_path, "loss_curve.png")
    plt.savefig(path)
    
    plt.subplot(1, 2, 2)
    plt.xlabel('Epoch')
    plt.ylabel('GPU mem')
    plt.plot(x, mem, 'r-')
    dir_path = 'output/vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2'
    os.makedirs(dir_path, exist_ok=True)

    plt.close()

if __name__ == '__main__':
    log_path = 'output/vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2/log_batch_size_20.txt'
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--log_path', default='output/vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2/log_batch_size_20.txt', type=str, required=True)
    # args = parser.parse_args()
    # visualization(log_path)

    visualization(log_path)
