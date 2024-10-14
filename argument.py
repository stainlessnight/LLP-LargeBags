from tap import Tap


class SimpleArgumentParser(Tap):
    
    # Training
    seed: int = 0                      # seed value
    epochs: int = 50                # the number of epochs
    batch_size: int = 512              # batch size for training
    mini_batch: int = 8                # mini batch size for training (the number of labeled bags)
    patience: int = 30                  # patience of early stopping
    lr: float = 1e-4                    # learning rate
    confidence_interval: float = 0.005  # 0.005 means 99% confidential interval
    num_sampled_instances: int = 25  # the number of sampled instances

    # Dataset
    bag_size: int = 200                 # bag size
    bags_num: int = 250                 # the number of labeled bags
    minibags_num: int = 50               # the number of minibags created from a population bag
    val_ratio: float = 0.2      # validation ratio
    num_workers: int = 4                # number of workers for dataloader

    # Model
    pretrained: bool = True
    classes: int = 10
    channels: int = 3                   # input image's channel
    dataset: str = 'cifar10'            # dataset name cifar10, mnist, NCT_CRC_HE_100K
    output_path: str = 'result/'        # output file name
    device: str = 'cuda:0'              # device

    # LLP Method
    label_sampling_method: str = 'hypergeometric' # 'population', 'hypergeometric', 'uniform', 'normal', 'mode'
    llp_method: str = 'pl'  # 'pl', 'llpvat', 'llpfc',
    apply_loss_weights: bool = True  # True, False
    minibags_instances: str = 'unfixed'  # 'unfixed', 'fixed'
    noisy_prior_choice: str = 'uniform'  # 'approx', 'uniform'
    weights: str = 'uniform'  # 'uniform', 'ch_vol', weights for LLPFC

    # Toy or Application
    toy: bool = False  # True, False