import time
import torch
import torch.optim as optim
import argparse
import csv
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import MobileVitClassifier
from utils import train_one_epoch, evaluate, test
from data import Splite_view_data, Mul_dataset
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--lambda-epochs', type=int, default=22, metavar='N',
                        help='gradually increase the value of lambda from 0 to 1')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='LR',
                        help='learning rate')
    parser.add_argument('--views',type=int, default=3, metavar='view',
                        help='learning view')
    parser.add_argument('--classes', type=int, default=12, metavar='class',
                        help='class')
    parser.add_argument('--data_name', type=str, default='tmc_256_16.txt',
                        help='known classes dataset')
    parser.add_argument('--unknown_file', type=str, default='cic_256_16.txt',  #
                        help='Unknown classes dataset')
    parser.add_argument('--transform', type=bool, default=True, metavar='transform',
                        help='Not used')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--is_open', type=bool, default=True, metavar='open set or not',
                        help='open or close')
    parser.add_argument('--threshold', type=float, default=0.002, metavar='threshold')
    parser.add_argument('--is_train', type=bool, default=False,
                        help='Train or Test')
    parser.add_argument('--byte_num', type=int, default=128,
                        help='Used to adjust the number of bytes')
    parser.add_argument('--packet_num', type=int, default=8,
                        help='Used to adjust the number of packets')

    args = parser.parse_args()
    views = args.views

    datasets_idx = args.data_name.split('_')[0]

    current_time = time.strftime("%Y%m_%d-%H_%M")
    log_dir = f'./runs/{datasets_idx}_experiment_byte_{args.byte_num}_packet_{args.packet_num}_{current_time}'


    tb_writer = SummaryWriter(log_dir=log_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    args.data_path = './datasets/' + args.data_name
    unknown_path = './datasets/' + args.unknown_file
    is_train = args.is_train

    if args.is_open:
        class_model = args.classes + 1
    else:
        class_model = args.classes
    train_data, train_label, test_data, test_label, val_data, val_label = Splite_view_data(args.data_path, args.is_open, unknown_path, args.classes, args.byte_num, args.packet_num).split_mul_data()


    train_dataset = Mul_dataset(images_path=train_data,
                                images_class=train_label,
                                views = views)

    val_dataset = Mul_dataset(images_path=val_data,
                              images_class=val_label,
                              views = views)

    test_dataset = Mul_dataset(images_path=test_data,
                               images_class=test_label,
                               views = views)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = args.batch_size,
                                               shuffle = True
                                               # ,num_workers = 4
                                               # ,collate_fn=train_dataset.collate_fn
                                               )
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size = args.batch_size,
                                               shuffle = False
                                               # ,num_workers = 4
                                               # ,collate_fn=val_dataset.collate_fn
                                              )
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True
                                              )



    model = MobileVitClassifier(class_model, args.views, args.lambda_epochs)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    model.to(device)

    if is_train:
        best_acc = 0.0

        for epoch in range(args.epochs+1):
            # train
            train_loss, train_acc = train_one_epoch(model = model,
                                                    optimizer = optimizer,
                                                    train_loader = train_loader,
                                                    device = device,
                                                    epoch = epoch,
                                                    views= views)

            # validate
            val_acc = evaluate(model = model,
                               data_loader = val_loader,
                               device = device,
                               epoch = epoch,
                               views= views)


            # writer.writerow([epoch, train_loss, train_acc])
            # writer.writerow([epoch,val_acc])
            #
            tags = ["train_loss", "train_acc", "val_acc", "learning_rate"]
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_acc, epoch)
            # tb_writer.add_scalar(tags[3], test_acc, epoch)
            tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"], epoch)

            if os.path.exists("./Tmc_dataset_save_model") is False:
                os.makedirs("./Tmc_dataset_save_model")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), "./Tmc_dataset_save_model/ONE_Tmc_{}_{}_best_model.pth".format(args.byte_num, args.packet_num))

            torch.save(model.state_dict(), "./Tmc_dataset_save_model/ONE_Tmc_{}_{}_latest_model.pth".format(args.byte_num, args.packet_num))


    test(model, test_loader, device, class_model, args.byte_num, args.packet_num, args.unknown_file, threshold=args.threshold, is_open=args.is_open, epoch = 1, views = views)





    threshold = args.threshold
    print('-------threshold: ',threshold)

