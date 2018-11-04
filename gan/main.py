import sys
from utils.misc import pp, visualize
import yaml
import tensorflow as tf
import argparse
import os


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()


def make_flags(args=None):
    FLAGS = parser.parse_args(args)
    if FLAGS.num_gpus is None:
        vis_devs = os.environ.get('CUDA_VISIBLE_DEVICES', "")
        FLAGS.num_gpus = len(vis_devs.split(','))

    if FLAGS.config_file:
        config = yaml.load(open(FLAGS.config_file))
        dic = vars(FLAGS)
        all(map(dic.pop, config))
        dic.update(config)
    return FLAGS


def add_arg(name, **kwargs):
    "Convenience to handle reasonable names for args as well as crappy ones."
    assert name[0] == '-'
    assert '-' not in name[1:]
    nice_name = '--' + name[1:].replace('_', '-')
    return parser.add_argument(name, nice_name, **kwargs)


# Optimizer
add_arg('-max_iteration',               default=150000,         type=int,       help='Epoch to train [%(default)s]')
add_arg('-beta1',                       default=0.5,            type=float,     help='Momentum term of adam [%(default)s]')
add_arg('-beta2',                       default=0.9,            type=float,     help='beta2 in adam [%(default)s]')
add_arg('-learning_rate',               default=0.0001,         type=float,     help='Learning rate [%(default)s]')
add_arg('-learning_rate_D',             default=-1,             type=float,     help='Learning rate for discriminator, if negative same as generator [%(default)s]')
add_arg('-dsteps',                      default=5,              type=int,       help='Number of discriminator steps in a row [%(default)s]')
add_arg('-gsteps',                      default=1,              type=int,       help='Number of generator steps in a row [%(default)s]')
add_arg('-start_dsteps',                default=10,             type=int,       help='Number of discrimintor steps in a row during first 20 steps and every 100th step [%(default)s]')

add_arg('-clip_grad',                   default=True,           type=str2bool,  help='Use gradient clipping [%(default)s]')
add_arg('-batch_norm',                  default=False,          type=str2bool,  help='Use of batch norm; overridden off if gradient penalty is used [%(default)s]')

# Initalization params
add_arg('-init',                        default=0.02,           type=float,     help='Initialization value [%(default)s]')
# dimensions
add_arg('-batch_size',                  default=64,             type=int,       help='The size of batch images [%(default)s]')
add_arg('-real_batch_size',             default=-1,             type=int,       help='The size of batch images for real samples. If -1 then same as batch_size [%(default)s]')
add_arg('-output_size',                 default=128,            type=int,       help='The size of the output images to produce [%(default)s]')
add_arg('-c_dim',                       default=3,              type=int,       help='Dimension of image color. [%(default)s]')
add_arg('-z_dim',                       default=128,            type=int,       help='Dimension of latent noise [%(default)s]')
add_arg('-df_dim',                      default=64,             type=int,       help='Discriminator no of channels at first conv layer [%(default)s]')
add_arg('-dof_dim',                     default=1,              type=int,       help='No of discriminator output features [%(default)s]')
add_arg('-gf_dim',                      default=64,             type=int,       help='No of generator channels [%(default)s]')
# Directories
add_arg('-dataset',                     default="cifar10",      type=str,       help='The name of the dataset [celebA, mnist, lsun, *cifar10*, imagenet]')
add_arg('-name',                        default="",             type=str,       help='The name of the experiment for saving purposes ')
add_arg('-checkpoint_dir',              default="checkpoint",   type=str,       help='Directory name to save the checkpoints [%(default)s]')
add_arg('-sample_dir',                  default="sample",       type=str,       help='Directory name to save the image samples [%(default)s]')
add_arg('-log_dir',                     default="log",          type=str,       help='Directory name to save the image samples [%(default)s]')
add_arg('-data_dir',                    default="data",         type=str,       help='Directory containing datasets [%(default)s]')
add_arg('-out_dir',                     default="",             type=str,       help='Directory name to save the outputs of the experiment (log, sample, checkpoints) [.]')
add_arg('-config_file',                 default="",             type=str,       help='path to a YAML config file overriding arguments')

# models
add_arg('-architecture',                default="dcgan",        type=str,       help='The name of the architecture [*dcgan*, g-resnet5, dcgan5]')
add_arg('-kernel',                      default="mix_rq_dot",   type=str,       help="The name of the kernel ['', 'mix_rbf', 'mix_rq', 'distance', 'dot', 'mix_rq_dot']")
add_arg('-model',                       default="mmd",          type=str,       help='The model type [*mmd*, smmd, swgan, wgan_gp]')

# training options
add_arg('-is_train',                    default=True,           type=str2bool,  help='True for training, False for testing [%(default)s]')
add_arg('-visualize',                   default=False,          type=str2bool,  help='True for visualizing, False for nothing [%(default)s]')
add_arg('-is_demo',                     default=False,          type=str2bool,  help='For testing [%(default)s]')

add_arg('-log',                         default=True,           type=str2bool,  help='Whether to write log to a file in samples directory [%(default)s]')
add_arg('-compute_scores',              default=True,           type=str2bool,  help='Compute scores [%(default)s]')
add_arg('-print_pca',                   default=False,          type=str2bool,  help='Print the PCA [%(default)s]')
add_arg('-suffix',                      default="",             type=str,       help="For additional settings ['', '_tf_records']")
add_arg('-gpu_mem',                     default=.9,             type=float,     help="GPU memory fraction limit [%(default)s]")
add_arg('-no_of_samples',               default=100000,         type=int,       help="number of samples to produce [%(default)s]")
add_arg('-save_layer_outputs',          default=0,              type=int,       help="Whether to save_layer_outputs. If == 2, saves outputs at exponential steps: 1, 2, 4, ..., 512 and every 1000. [*0*, 1, 2]")
add_arg('-ckpt_name',                   default="",             type=str,       help="Name of the checkpoint to load [none]")

# Decay rates
add_arg('-decay_rate',                  default=.8,             type=float,     help='Decay rate [%(default)s]')
add_arg('-gp_decay_rate',               default=.8,             type=float,     help='Decay rate of the gradient penalty [%(default)s]')
add_arg('-sc_decay_rate',               default=1.,             type=float,     help='Decay of the scaling factor [%(default)s]')
add_arg('-restart_lr',                  default=False,          type=str2bool,  help='Whether to use lr scheduler based on 3-sample test [%(default)s]')
add_arg('-restart_sc',                  default=False,          type=str2bool,  help='Ensures the discriminator network is injective by adding the input to the feature [%(default)s]')
add_arg('-MMD_lr_scheduler',            default=True,           type=str2bool,  help='Whether to use lr scheduler based on 3-sample test [%(default)s]')
add_arg('-MMD_sdlr_past_sample',        default=10,             type=int,       help='lr scheduler: number of past iterations to keep [%(default)s]')
add_arg('-MMD_sdlr_num_test',           default=3,              type=int,       help='lr scheduler: number of failures to decrease KID score [%(default)s]')
add_arg('-MMD_sdlr_freq',               default=2000,           type=int,       help='lr scheduler: frequency of scoring the model [%(default)s]')

# discriminator penalties
add_arg('-gradient_penalty',            default=0.0,            type=float,     help='Use gradient penalty if > 0 [%(default)s]')
add_arg('-L2_discriminator_penalty',    default=0.0,            type=float,     help="Use L2 penalty on discriminator features if > 0 [%(default)s]")

# scaled MMD
add_arg('-with_scaling',                default=False,          type=str2bool,  help='Use scaled MMD [%(default)s]')
add_arg('-scaling_coeff',               default=10.,            type=float,     help='coeff of scaling [%(default)s]')
add_arg('-scaling_variant',             default='grad',         type=str,       help='The variant of the scaled MMD   [value_and_grad, *grad*]')
add_arg('-use_gaussian_noise',          default=False,          type=str2bool,  help='Add N(0, 10^2) noise to images in scaling [%(default)s]')

# spectral normalization
add_arg('-with_sn',                     default=False,          type=str2bool,  help='use spectral normalization [%(default)s]')
add_arg('-with_learnable_sn_scale',     default=False,          type=str2bool,  help='train the scale of normalized weights [%(default)s]')

# incomplete cholesky options for sobolevmmd
add_arg('-use_incomplete_cho',          default=True,           type=str2bool,  help="whether to use incomplete Cholesky for sobolevmmd [%(default)s]")
add_arg('-incho_eta',                   default=1e-3,           type=float,     help="stopping criterion for incomplete cholesky [%(default)s]")
add_arg('-incho_max_steps',             default=1000,           type=int,       help="iteration cap for incomplete cholesky [%(default)s]")

# multi-gpu training
add_arg('-multi_gpu',                   default=False,          type=str2bool,  help='Train accross multiple gpus in a multi-tower fashion [%(default)s]')
add_arg('-num_gpus',                    default=None,           type=int,       help='Number of GPUs to use [len(CUDA_VISIBLE_DEVICES)]')
# conditional gan, only for imagenet
add_arg('-with_labels',                 default=False,          type=str2bool,  help='Conditional GAN [%(default)s]')


def main(_):
    global FLAGS
    pp.pprint(vars(FLAGS))

    sess_config = tf.ConfigProto(
        device_count={"CPU": 3},
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    if FLAGS.dataset == 'mnist':
        FLAGS.output_size = 28
        FLAGS.c_dim = 1
    elif FLAGS.dataset == 'cifar10':
        FLAGS.output_size = 32
        FLAGS.c_dim = 3
    elif FLAGS.dataset in ['celebA', 'lsun', 'imagenet']:
        FLAGS.c_dim = 3

    from core import model_class
    Model = model_class(FLAGS.model)

    with tf.Session(config=sess_config) as sess:
        #sess = tf_debug.tf_debug.TensorBoardDebugWrapperSession(sess,'localhost:6064')
        #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        gan = Model(sess, config=FLAGS)

        if FLAGS.is_train:
            gan.train()
        elif FLAGS.print_pca:
            gan.print_pca()
        elif FLAGS.visualize:
            gan.load_checkpoint()
            visualize(sess, gan, FLAGS, 2)
        else:
            gan.get_samples(FLAGS.no_of_samples, layers=[-1])

        if FLAGS.log:
            sys.stdout = gan.old_stdout
            gan.log_file.close()
        gan.sess.close()


if __name__ == '__main__':
    FLAGS = make_flags()
    tf.app.run()
