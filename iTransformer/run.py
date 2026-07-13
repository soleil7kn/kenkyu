import argparse
import torch
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from experiments.exp_long_term_forecasting_partial import Exp_Long_Term_Forecast_Partial
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='iTransformer')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='iTransformer',
                        help='model name, options: [iTransformer, iInformer, iReformer, iFlowformer, iFlashformer]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='electricity.csv', help='data csv file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length') # no longer needed in inverted Transformers
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size') # applicable on arbitrary number of variates in inverted Transformers
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # iTransformer
    parser.add_argument('--exp_name', type=str, required=False, default='MTSF',
                        help='experiemnt name, options:[MTSF, partial_train]')
    parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--target_root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='data file')
    parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient_training (exp_name should be partial train)') # See Figure 8 of our paper for the detail
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
    parser.add_argument('--partial_start_index', type=int, default=0, help='the start index of variates for partial training, '
                                                                           'you can select [partial_start_index, min(enc_in + partial_start_index, N)]')

    # 追加したparser
    parser.add_argument(
        '--skip_rates',
        type=str,
        default='2',
        help='skip rates for Multi-Skip Token, e.g. 2 or 1,2,4'
    )

    parser.add_argument(
        '--use_skip_weight',
        type=int,
        default=1,
        help='whether to use learnable skip weights'
    )

    parser.add_argument(
        '--use_skip_interaction',
        type=int,
        default=0,
        help='whether to use Skip-Time Interaction'
    )

    parser.add_argument(
        '--skip_interaction_layers',
        type=int,
        default=0,
        help='number of Skip-Time Interaction layers'
    )

    parser.add_argument(
        '--use_sticln',
        type=int,
        default=0,
        help='whether to use STICLN'
    )

    parser.add_argument(
    '--task_name',
    type=str,
    default='long_term_forecast',
    help='task name'
    )

    # TimeMixer用の設定
    parser.add_argument(
        '--down_sampling_layers',
        type=int,
        default=3,
        help='number of down-sampling layers'
    )

    parser.add_argument(
        '--down_sampling_method',
        type=str,
        default='avg',
        choices=['avg', 'max', 'conv'],
        help='down-sampling method'
    )

    parser.add_argument(
        '--down_sampling_window',
        type=int,
        default=2,
        help='down-sampling window size'
    )

    parser.add_argument(
        '--decomp_method',
        type=str,
        default='moving_avg',
        choices=['moving_avg', 'dft_decomp'],
        help='series decomposition method'
    )

    parser.add_argument(
        '--top_k',
        type=int,
        default=5,
        help='number of selected frequencies in DFT decomposition'
    )

    parser.add_argument(
        '--use_future_temporal_feature',
        type=int,
        default=0,
        choices=[0, 1],
        help='whether to use future temporal features'
    )

    # TimeMixer + iTransformer refinement用
    parser.add_argument(
        '--refiner_d_model',
        type=int,
        default=64
    )

    parser.add_argument(
        '--refiner_n_heads',
        type=int,
        default=4
    )

    parser.add_argument(
        '--refiner_d_ff',
        type=int,
        default=128
    )

    parser.add_argument(
        '--refiner_e_layers',
        type=int,
        default=1
    )

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    if args.exp_name == 'partial_train': # See Figure 8 of our paper, for the detail
        Exp = Exp_Long_Term_Forecast_Partial
    else: # MTSF: multivariate time series forecasting
        Exp = Exp_Long_Term_Forecast


    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                args.class_strategy, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            model_for_weight = exp.model.module if hasattr(exp.model, "module") else exp.model

            if hasattr(model_for_weight, "get_skip_weights"):

                weights = model_for_weight.get_skip_weights().detach().cpu()

                print("skip weights:")
                print(weights)

                idx = 0
                for skip in model_for_weight.skip_rates:
                    for offset in range(skip):
                        print(
                            f"skip={skip}, offset={offset}: {weights[idx].item():.4f}"
                        )
                        idx += 1

            if hasattr(model_for_weight, "weighted_pooling"):

                weights = torch.softmax(
                    model_for_weight.weighted_pooling.skip_logits.detach().cpu(),
                    dim=0
                )

                print("skip weights:")
                print(weights)

                if hasattr(model_for_weight, "skip_rates"):
                    idx = 0

                    for skip in model_for_weight.skip_rates:
                        for offset in range(skip):
                            print(
                                f"skip={skip}, offset={offset}: {weights[idx].item():.4f}"
                            )
                            idx += 1

            if hasattr(model_for_weight, "stif_gate"):

                raw_gate = model_for_weight.stif_gate.detach().cpu()

                gate = torch.tanh(raw_gate)

                print("raw stif gate parameter:")
                print(raw_gate)

                print("stif gate after tanh:")
                print(gate)

            if hasattr(model_for_weight, "sticln_gate"):

                raw_sticln_gate = model_for_weight.sticln_gate.detach().cpu()

                sticln_gate = torch.tanh(raw_sticln_gate)

                print("raw sticln gate parameter:")
                print(raw_sticln_gate)

                print("sticln gate after tanh:")
                print(sticln_gate)

            if hasattr(model_for_weight, "main_gate"):

                raw_main_gate = model_for_weight.main_gate.detach().cpu()
                main_gate = torch.tanh(raw_main_gate)

                print("raw main gate parameter:")
                print(raw_main_gate)

                print("main gate after tanh:")
                print(main_gate)


            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            args.class_strategy,
            args.skip_rates,
            args.use_skip_weight,
            args.use_skip_interaction,
            args.use_sticln, 
            ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
