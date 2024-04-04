import argparse
import torch
import sys

sys.path.append("/kaggle/input/newmnistsam/pytorch/2828/1/sam-tud-main/sam-tud-main/example")
from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.MNIST import MNIST
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
from model.neural_net import SimpleNN

import sam

learning_rates = [0.01, 0.05, 0.1, 0.2, 0.3]
rhos = [0.01, 0.05, 0.1]
accuracies = {}  # Dictionary to hold accuracy for each (lr, rho) pair

if __name__ == "__main__":
    for lr in learning_rates:
        for rho in rhos:
            parser = argparse.ArgumentParser()
            parser.add_argument("--adaptive", default=True, type=bool)
            parser.add_argument("--batch_size", default=128, type=int)
            parser.add_argument("--depth", default=16, type=int)
            parser.add_argument("--dropout", default=0.0, type=float)
            parser.add_argument("--epochs", default=2, type=int)
            parser.add_argument("--label_smoothing", default=0.1, type=float)
            parser.add_argument("--learning_rate", default=lr, type=float)
            parser.add_argument("--momentum", default=0.9, type=float)
            parser.add_argument("--threads", default=2, type=int)
            parser.add_argument("--rho", default=rho, type=float)
            parser.add_argument("--weight_decay", default=0.0005, type=float)
            parser.add_argument("--width_factor", default=8, type=int)
            args, unknown = parser.parse_known_args()

            initialize(args, seed=42)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            dataset = MNIST(args.batch_size, args.threads)
            log = Log(log_each=10)
            model = SimpleNN().to(device)

            base_optimizer = torch.optim.SGD
            optimizer = sam.SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)
            scheduler = StepLR(optimizer, args.learning_rate, args.epochs)

            best_accuracies = []

            for epoch in range(args.epochs):
                model.train()
                log.train(len_dataset=len(dataset.train))

                for batch in dataset.train:
                    inputs, targets = (b.to(device) for b in batch)
                    enable_running_stats(model)
                    predictions = model(inputs)
                    loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
                    loss.mean().backward()
                    optimizer.first_step(zero_grad=True)

                    disable_running_stats(model)
                    smooth_crossentropy(model(inputs), targets, smoothing=args.label_smoothing).mean().backward()
                    optimizer.second_step(zero_grad=True)

                    with torch.no_grad():
                        correct = torch.argmax(predictions.data, 1) == targets
                        log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                        scheduler(epoch)

                model.eval()
                log.eval(len_dataset=len(dataset.test))

                with torch.no_grad():
                    for batch in dataset.test:
                        inputs, targets = (b.to(device) for b in batch)
                        predictions = model(inputs)
                        loss = smooth_crossentropy(predictions, targets)
                        correct = torch.argmax(predictions, 1) == targets
                        log(model, loss.cpu(), correct.cpu())

                best_accuracies.append(log.epoch_state["accuracy"] / 100.0)
            accuracies[(lr, rho)] = max(best_accuracies) / 100.0
            log.flush()
        print(accuracies)
        
# Remove SAM import and SAM-specific arguments from argparse

# import argparse
# import torch
# import sys
#
# sys.path.append("/kaggle/input/sam_mnist/pytorch/sammn/1/sam-tud-main/example")
# from model.wide_res_net import WideResNet
# from model.smooth_cross_entropy import smooth_crossentropy
# from data.cifar import Cifar
# from data.MNIST import MNIST
# from utility.log import Log
# from utility.initialize import initialize
# from utility.step_lr import StepLR
# from model.neural_net import SimpleNN
# from utility.bypass_bn import enable_running_stats, disable_running_stats
#
# # No import for SAM needed
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     # Remove the SAM-specific arguments from here
#     parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
#     parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
#     parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
#     parser.add_argument("--epochs", default=100, type=int, help="Total number of epochs.")
#     parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
#     parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
#     parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
#     parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
#     parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
#     parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
#     args, unknown = parser.parse_known_args()
#
#     initialize(args, seed=42)
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     dataset = MNIST(args.batch_size, args.threads)
#     log = Log(log_each=10)
#     # model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)
#     model = SimpleNN().to(device)
#     # Use SGD directly
#     optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
#     scheduler = StepLR(optimizer, args.learning_rate, args.epochs)
#
#     test_accuracy = []
#
#     for epoch in range(args.epochs):
#         model.train()
#         log.train(len_dataset=len(dataset.train))
#
#         for batch in dataset.train:
#             inputs, targets = (b.to(device) for b in batch)
#
#             # Standard training pass with SGD
#             optimizer.zero_grad()
#             predictions = model(inputs)
#             loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
#             loss.mean().backward()
#             optimizer.step()
#
#             with torch.no_grad():
#                 correct = torch.argmax(predictions.data, 1) == targets
#                 log(model, loss.cpu(), correct.cpu(), scheduler.lr())
#                 scheduler(epoch)
#
#         model.eval()
#         log.eval(len_dataset=len(dataset.test))
#
#         with torch.no_grad():
#             for batch in dataset.test:
#                 inputs, targets = (b.to(device) for b in batch)
#
#                 predictions = model(inputs)
#                 loss = smooth_crossentropy(predictions, targets)
#                 correct = torch.argmax(predictions, 1) == targets
#                 log(model, loss.cpu(), correct.cpu())
#
#         test_accuracy.append(log.epoch_state["accuracy"] / 100.0)
#
#     log.flush()
#     print(test_accuracy)

# import argparse
# import torch
# import sys
#
# sys.path.append("/kaggle/input/sam_mnist/pytorch/sammn/1/sam-tud-main/example")
# from model.wide_res_net import WideResNet
# from model.smooth_cross_entropy import smooth_crossentropy
# from data.cifar import Cifar
# from data.MNIST import MNIST
# from utility.log import Log
# from utility.initialize import initialize
# from utility.step_lr import StepLR
# from model.neural_net import SimpleNN
# from utility.bypass_bn import enable_running_stats, disable_running_stats
#
# learning_rates = [0.01, 0.05, 0.1, 0.2, 0.3]
# accuracies = {}  # Dictionary to hold accuracy for each learning rate
#
# if __name__ == "__main__":
#     for lr in learning_rates:
#         parser = argparse.ArgumentParser()
#         parser.add_argument("--batch_size", default=128, type=int)
#         parser.add_argument("--depth", default=16, type=int)
#         parser.add_argument("--dropout", default=0.0, type=float)
#         parser.add_argument("--epochs", default=100, type=int)
#         parser.add_argument("--label_smoothing", default=0.1, type=float)
#         parser.add_argument("--learning_rate", default=lr, type=float)  # Use current lr from the loop
#         parser.add_argument("--momentum", default=0.9, type=float)
#         parser.add_argument("--threads", default=2, type=int)
#         parser.add_argument("--weight_decay", default=0.0005, type=float)
#         parser.add_argument("--width_factor", default=8, type=int)
#         args, unknown = parser.parse_known_args()
#
#         initialize(args, seed=42)
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#         dataset = MNIST(args.batch_size, args.threads)
#         log = Log(log_each=10)
#         model = SimpleNN().to(device)
#         optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
#         scheduler = StepLR(optimizer, args.learning_rate, args.epochs)
#
#         best_accuracies = []
#
#         for epoch in range(args.epochs):
#             model.train()
#             log.train(len_dataset=len(dataset.train))
#
#             for batch in dataset.train:
#                 inputs, targets = (b.to(device) for b in batch)
#                 optimizer.zero_grad()
#                 predictions = model(inputs)
#                 loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
#                 loss.mean().backward()
#                 optimizer.step()
#
#                 with torch.no_grad():
#                     correct = torch.argmax(predictions.data, 1) == targets
#                     log(model, loss.cpu(), correct.cpu(), scheduler.lr())
#                     scheduler(epoch)
#
#             model.eval()
#             log.eval(len_dataset=len(dataset.test))
#
#             with torch.no_grad():
#                 for batch in dataset.test:
#                     inputs, targets = (b.to(device) for b in batch)
#                     predictions = model(inputs)
#                     loss = smooth_crossentropy(predictions, targets)
#                     correct = torch.argmax(predictions, 1) == targets
#                     log(model, loss.cpu(), correct.cpu())
#
#             # Record the final accuracy for the current learning rate
#             best_accuracies.append(log.epoch_state["accuracy"] / 100.0)
#         accuracies[args.learning_rate] = max(best_accuracies)
#         log.flush()
#     print(accuracies)

